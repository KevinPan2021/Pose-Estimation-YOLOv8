import math
import torch
import time
import torchvision
import numpy as np

from summary import Summary

    
# Generate anchors from features
def make_anchors(x, strides, offset=0.5):
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, dtype=x[i].dtype, device=x[i].device) + offset  # shift x
        sy = torch.arange(end=h, dtype=x[i].dtype, device=x[i].device) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=x[i].dtype, device=x[i].device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


# convert from [x_center, y_center, width, height] to [x_min, y_min, x_max, y_max]
@torch.no_grad()
def wh2xy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


# convert from [x_min, x_max, x_max, y_max] to [centerX, centerY, width, height]
@torch.no_grad()
def xy2wh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y



@torch.no_grad()
def non_max_suppression(
        outputs, 
        nc,
        conf_threshold=0.25, 
        iou_threshold=0.45,
        max_wh = 7680, # (pixels) maximum box width and height
        max_det = 300,  # the maximum number of boxes to keep after NMS
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    ):
    bs = outputs.shape[0]  # batch size
    nc = nc or (outputs.shape[1] - 4)  # number of classes
    nm = outputs.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = outputs[:, 4:mi].amax(1) > conf_threshold  # candidates

    # Settings
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=outputs.device)] * bs
    for index, x in enumerate(outputs):  # image index, image inference
        x = x.transpose(0, -1)[xc[index]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)
        box = wh2xy(box)  # (center_x, center_y, width, height) to (center_x, center_y, x2, y2)
        if nc > 1:
            i, j = (cls > conf_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_threshold]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        i = i[:max_det]  # limit detections

        output[index] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output



def pad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv



class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, pad(k, p, d), d, g, False)
        self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.03)
        self.relu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, ch, add=True):
        super().__init__()
        self.add_m = add
        self.res_m = torch.nn.Sequential(Conv(ch, ch, 3),
                                         Conv(ch, ch, 3))

    def forward(self, x):
        return self.res_m(x) + x if self.add_m else self.res_m(x)


# cross stage partial networks
class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2)
        self.conv2 = Conv(in_ch, out_ch // 2)
        self.conv3 = Conv((2 + n) * out_ch // 2, out_ch)
        self.res_m = torch.nn.ModuleList(Residual(out_ch // 2, add) for _ in range(n))

    def forward(self, x):
        y = [self.conv1(x), self.conv2(x)]
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv3(torch.cat(y, dim=1))


# spatial pyramid pooling
class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2)
        self.conv2 = Conv(in_ch * 2, out_ch)
        self.res_m = torch.nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.res_m(y2)], 1))


class DarkNet(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        p1 = [Conv(width[0], width[1], 3, 2)]
        p2 = [Conv(width[1], width[2], 3, 2),
              CSP(width[2], width[2], depth[0])]
        p3 = [Conv(width[2], width[3], 3, 2),
              CSP(width[3], width[3], depth[1])]
        p4 = [Conv(width[3], width[4], 3, 2),
              CSP(width[4], width[4], depth[2])]
        p5 = [Conv(width[4], width[5], 3, 2),
              CSP(width[5], width[5], depth[0]),
              SPP(width[5], width[5])]

        self.p1 = torch.nn.Sequential(*p1)
        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


# Dark Feature Pyramid Network
class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.up = torch.nn.Upsample(None, 2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[0], False)
        self.h2 = CSP(width[3] + width[4], width[3], depth[0], False)
        self.h3 = Conv(width[3], width[3], 3, 2)
        self.h4 = CSP(width[3] + width[4], width[4], depth[0], False)
        self.h5 = Conv(width[4], width[4], 3, 2)
        self.h6 = CSP(width[4] + width[5], width[5], depth[0], False)

    def forward(self, x):
        p3, p4, p5 = x
        h1 = self.h1(torch.cat([self.up(p5), p4], 1))
        h2 = self.h2(torch.cat([self.up(h1), p3], 1))
        h4 = self.h4(torch.cat([self.h3(h2), h1], 1))
        h6 = self.h6(torch.cat([self.h5(h4), p5], 1))
        return h2, h4, h6


# Distribution Focal Loss
class DFL(torch.nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)



class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=1, kpt_shape=(17, 3), ch=()):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.kpt_shape = kpt_shape  # number of kpt, number of dims (2 for x,y or 3 for x,y,visible)
        self.num_kpt = kpt_shape[0] * kpt_shape[1]  # number of kpt total
        
        c1 = max((16, ch[0] // 4, self.ch * 4))
        c2 = max(ch[0], self.nc)
        c3 = max(ch[0] // 4, self.num_kpt)

        self.dfl = DFL(self.ch) if self.ch > 1 else torch.nn.Identity()
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(
            Conv(x, c2, 3),
            Conv(c2, c2, 3),
            torch.nn.Conv2d(c2, self.nc, 1)) for x in ch)
        self.box = torch.nn.ModuleList(torch.nn.Sequential(
            Conv(x, c1, 3),
            Conv(c1, c1, 3),
            torch.nn.Conv2d(c1, 4 * self.ch, 1)) for x in ch)
        self.kpt = torch.nn.ModuleList(torch.nn.Sequential(
            Conv(x, c3, 3),
            Conv(c3, c3, 3),
            torch.nn.Conv2d(c3, self.num_kpt, 1)) for x in ch)
    
    
    def forward(self, x, dummy):
        x_det, x_det_copy = self.detect_box(x, dummy)
        x_kpt = self.detect_kpt(x)
        
        if dummy:
            return x_det, x_kpt
        
        return torch.cat([x_det, self.decode_kpt(x_kpt)], dim=1), (x_det_copy, x_kpt)

    
    def detect_box(self, x, dummy):
        x_det = []
        shape = x[0].shape[0]
        for i in range(self.nl):
            x_det.append(torch.cat((self.box[i](x[i]), self.cls[i](x[i])), dim=1))
        
        if dummy:
            return x_det, None
        
        x_det_copy = list(x_det)
        
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x_det, self.stride, 0.5))

        x_cat = torch.cat([i.view(shape, self.no, -1) for i in x_det], dim=2)
        box, cls = x_cat.split((self.ch * 4, self.nc), 1)
        a, b = self.dfl(box).chunk(2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(((a + b) / 2, b - a), dim=1)
        return torch.cat((box * self.strides, cls.sigmoid()), dim=1), x_det_copy
    
    
    def detect_kpt(self, x):
        x_kpt = []
        shape = x[0].shape[0]
        for i in range(self.nl):
            kpt = self.kpt[i](x[i])
            kpt = kpt.view(shape, self.num_kpt, -1)
            x_kpt.append(kpt)
        return torch.cat(x_kpt, dim=-1)


    def decode_kpt(self, x):
        y = x.clone()
        dim = self.kpt_shape[1]
        if dim == 3:
            y[:, 2::3].sigmoid_()  # inplace sigmoid
        y[:, 0::dim] = (y[:, 0::dim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
        y[:, 1::dim] = (y[:, 1::dim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
        return y


    def initialize_biases(self):
        # Initialize Detect() biases,
        # WARNING: requires stride availability
        for a, b, s in zip(self.box, self.cls, self.stride):
            # box
            a[-1].bias.data[:] = 1.0
            # cls (.01 objects, 80 classes, 640 img)
            b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)



# main yolo class
class YOLO(torch.nn.Module):
    def __init__(self, size='m', num_classes=1):
        super().__init__()
        
        # model size parameters
        if size == 'n':
            depth = [1, 2, 2]
            width = [3, 16, 32, 64, 128, 256]
            
        elif size == 's':
            depth = [1, 2, 2]
            width = [3, 32, 64, 128, 256, 512]
        
        elif size == 'm':
            depth = [2, 4, 4]
            width = [3, 48, 96, 192, 384, 576]
            
        elif size == 'l':
            depth = [3, 6, 6]
            width = [3, 64, 128, 256, 512, 512]
            
        elif size == 'x':
            depth = [3, 6, 6]
            width = [3, 80, 160, 320, 640, 640]
            
            
        self.size = size
        
        self.net = DarkNet(width, depth)
        self.fpn = DarkFPN(width, depth)

        img_dummy = torch.zeros(1, 3, 256, 256)
        self.head = Head(num_classes, (17, 3), (width[3], width[4], width[5]))
        self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy, dummy=True)[0]])
        self.stride = self.head.stride
        self.head.initialize_biases()
    
    
    def forward(self, x, dummy=False):
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x), dummy)
    
    
    def name(self):
        return 'yolo_v8_' + self.size
    
    
    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self
    
    
    
def main():    
    device = 'cuda'
    
    # Creating model and testing output shapes 
    model = YOLO(size='s', num_classes=1) 
    model = model.to(device)
    Summary(model)

    
    
if __name__ == "__main__": 
    main()
    