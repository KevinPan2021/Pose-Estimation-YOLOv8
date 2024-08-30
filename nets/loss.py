import numpy as np
import torch
import math
from torch.nn.functional import cross_entropy

from nn import make_anchors, wh2xy, xy2wh


KPT_SIGMA = np.array([
    .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07,
    1.07, .87, .87, .89, .89
]) / 10.0


def compute_iou(box1, box2, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
    # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
    v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)  # CIoU


# Task-aligned One-stage Object Detection assigner
class Assigner(torch.nn.Module):
    def __init__(self, top_k=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.top_k = top_k
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        size = pd_scores.size(0)
        max_boxes = gt_bboxes.size(1)

        if max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device),
                    torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))
        # get in_gts mask, (b, max_num_obj, h*w)
        n_anchors = anc_points.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((anc_points[None] - lt, rb - anc_points[None]), dim=2)
        bbox_deltas = bbox_deltas.view(bs, n_boxes, n_anchors, -1)
        mask_in_gts = bbox_deltas.amin(3).gt_(1e-9)
        # get anchor_align metric, (b, max_num_obj, h*w)
        na = pd_bboxes.shape[-2]
        true_mask = (mask_in_gts * mask_gt).bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([size, max_boxes, na],
                               dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([size, max_boxes, na],
                                  dtype=pd_scores.dtype, device=pd_scores.device)
        index = torch.zeros([2, size, max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        index[0] = torch.arange(end=size).view(-1, 1).repeat(1, max_boxes)  # b, max_num_obj
        index[1] = gt_labels.long().squeeze(-1)  # b, max_num_obj
        # get the scores of each grid for each gt cls
        bbox_scores[true_mask] = pd_scores[index[0], :, index[1]][true_mask]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).repeat(1, max_boxes, 1, 1)[true_mask]
        gt_boxes = gt_bboxes.unsqueeze(2).repeat(1, 1, na, 1)[true_mask]
        overlaps[true_mask] = compute_iou(gt_boxes, pd_boxes).squeeze(-1).clamp(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        # get top_k_metric mask, (b, max_num_obj, h*w)
        #num_anchors = align_metric.shape[-1]  # h*w
        top_k_mask = mask_gt.repeat([1, 1, self.top_k]).bool()
        # (b, max_num_obj, top_k)
        top_k_metrics, top_k_indices = torch.topk(align_metric, self.top_k, dim=-1, largest=True)
        if top_k_mask is None:
            top_k_mask = (top_k_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(top_k_indices)
        # (b, max_num_obj, top_k)
        top_k_indices.masked_fill_(~top_k_mask, 0)
        # (b, max_num_obj, top_k, h*w) -> (b, max_num_obj, h*w)
        count = torch.zeros(align_metric.shape, dtype=torch.int8, device=top_k_indices.device)
        ones = torch.ones_like(top_k_indices[:, :, :1], dtype=torch.int8, device=top_k_indices.device)
        for k in range(self.top_k):
            count.scatter_add_(-1, top_k_indices[:, :, k:k + 1], ones)
        # filter invalid bboxes
        count.masked_fill_(count > 1, 0)
        mask_top_k = count.to(align_metric.dtype)
        # merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_top_k * mask_in_gts * mask_gt
        # (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, max_boxes, 1])  # (b, n_max_boxes, h*w)
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
            fg_mask = mask_pos.sum(-2)
        # find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)

        # assigned target, assigned target labels, (b, 1)
        batch_index = torch.arange(end=size, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_idx = target_gt_idx + batch_index * max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_idx]  # (b, h*w)

        # assigned target boxes, (b, max_num_obj, 4) -> (b, h*w)
        target_bboxes = gt_bboxes.view(-1, 4)[target_idx]

        # assigned target scores
        target_labels.clamp(0)
        target_scores = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.num_classes),
                                    dtype=torch.int64,
                                    device=target_labels.device)  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        # normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps))
        target_scores = target_scores * (norm_align_metric.amax(-2).unsqueeze(-1))

        return target_bboxes, target_scores, fg_mask.bool(), target_gt_idx



# bounding box loss
class BoxLoss(torch.nn.Module):
    def __init__(self, dfl_ch):
        super().__init__()
        self.dfl_ch = dfl_ch

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # IoU loss
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = compute_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        a, b = target_bboxes.chunk(2, -1)
        target = torch.cat((anchor_points - a, b - anchor_points), -1)
        target = target.clamp(0, self.dfl_ch - 0.01)
        loss_dfl = self.df_loss(pred_dist[fg_mask].view(-1, self.dfl_ch + 1), target[fg_mask])
        loss_dfl = (loss_dfl * weight).sum() / target_scores_sum

        return loss_iou, loss_dfl
    
    # # Return sum of left and right Distribution Focal Loss (DFL) loss
    @staticmethod
    def df_loss(pred_dist, target):
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        left_loss = cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape)
        right_loss = cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape)
        return (left_loss * wl + right_loss * wr).mean(-1, keepdim=True)


# keypoints loss
class PointLoss(torch.nn.Module):
    def __init__(self, sigmas):
        super().__init__()
        self.sigmas = sigmas
    
    # Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints.
    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]) ** 2 + (pred_kpts[..., 1] - gt_kpts[..., 1]) ** 2
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        e = d / (2 * self.sigmas) ** 2 / (area + 1e-9) / 2  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()

    
    
class ComputeLoss:
    def __init__(self, model, cls_gain=0.5, box_gain=7.5, dfl_gain=1.5, kpt_gain=12.0, kpt_obj_gain=1.0):
        super().__init__()
        if hasattr(model, 'module'):
            model = model.module

        device = next(model.parameters()).device  # get model device

        m = model.head  # Head() module
        self.no = m.no
        self.nc = m.nc  # number of classes
        self.dfl_ch = m.ch
        
        # gains
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.dfl_gain = dfl_gain
        self.kpt_gain = kpt_gain
        self.kpt_obj_gain = kpt_obj_gain
        
        self.device = device
        self.stride = m.stride  # model strides

        self.kpt_shape = model.head.kpt_shape
        if self.kpt_shape == [17, 3]:
            sigmas = torch.from_numpy(KPT_SIGMA.copy()).to(self.device)
        else:
            sigmas = torch.ones(self.kpt_shape[0], device=self.device) / self.kpt_shape[0]

        self.assigner = Assigner(top_k=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.box_loss = BoxLoss(m.ch - 1).to(device)
        self.kpt_loss = PointLoss(sigmas=sigmas)

        self.box_bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.kpt_bce = torch.nn.BCEWithLogitsLoss()
        self.project = torch.arange(m.ch, dtype=torch.float, device=device)


    def __call__(self, outputs, targets):
        x_det, x_kpt = outputs
        shape = x_det[0].shape
        loss = torch.zeros(5, device=self.device)  # cls, box, dfl, kpt_location, kpt_visibility

        x_cat = torch.cat([i.view(shape[0], self.no, -1) for i in x_det], 2)
        pred_distri, pred_scores = x_cat.split((self.dfl_ch * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        x_kpt = x_kpt.permute(0, 2, 1).contiguous()

        size = torch.tensor(shape[2:], device=self.device, dtype=pred_scores.dtype)
        size = size * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(x_det, self.stride, 0.5)

        # targets
        indices = targets['idx'].view(-1, 1)
        batch_size = pred_scores.shape[0]
        
        box_targets = torch.cat((indices, targets['cls'].view(-1, 1), targets['box']), 1)
        box_targets = box_targets.to(self.device)
        if box_targets.shape[0] == 0:
            gt = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = box_targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            gt = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    gt[j, :n] = box_targets[matches, 1:]
            gt[..., 1:5] = wh2xy(gt[..., 1:5].mul_(size[[1, 0, 1, 0]]))
            
        gt_labels, gt_bboxes = gt.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = self.box_decode(anchor_points, pred_distri, self.project)  # xyxy, (b, h*w, 4)
        x_kpt = self.kpt_decode(anchor_points, x_kpt.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        loss[0] = self.box_bce(pred_scores, target_scores.to(pred_scores.dtype)).sum()  # BCE
        loss[0] /= target_scores_sum

        if fg_mask.sum():
            # box loss
            target_bboxes /= stride_tensor
            loss[1], loss[2] = self.box_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask
            )
            
            # keypoint loss
            kpt = targets['kpt'].to(self.device).float().clone()
            
            kpt[..., 0] *= size[1]
            kpt[..., 1] *= size[0]
            
            for i in range(batch_size):
                if fg_mask[i].sum():
                    idx = target_gt_idx[i][fg_mask[i]]
                    gt_kpt = kpt[indices.view(-1) == i][idx]  # (n, 51)
                    gt_kpt[..., 0] /= stride_tensor[fg_mask[i]]
                    gt_kpt[..., 1] /= stride_tensor[fg_mask[i]]
                    # calculate the area based on weight and height
                    area = xy2wh(target_bboxes[i][fg_mask[i]])[:, 2:].prod(1, keepdim=True)
                    pred_kpt = x_kpt[i][fg_mask[i]]
                    kpt_mask = gt_kpt[..., 2] != 0

                    # kpt loss
                    loss[3] += self.kpt_loss(pred_kpt, gt_kpt, kpt_mask, area)
                    if pred_kpt.shape[-1] == 3:
                        loss[4] += self.kpt_bce(pred_kpt[..., 2], kpt_mask.float())  # kpt obj loss
        
        # scale loss by gain
        loss[0] *= self.cls_gain
        loss[1] *= self.box_gain
        loss[2] *= self.dfl_gain
        loss[3] *= self.kpt_gain / batch_size
        loss[4] *= self.kpt_obj_gain / batch_size

        return loss.sum()

    @staticmethod
    def box_decode(anchor_points, pred_dist, project):
        b, a, c = pred_dist.shape  # batch, anchors, channels
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3)
        pred_dist = pred_dist.matmul(project.type(pred_dist.dtype))
        a, b = pred_dist.chunk(2, -1)
        a = anchor_points - a
        b = anchor_points + b
        return torch.cat((a, b), -1)

    @staticmethod
    def kpt_decode(anchor_points, pred_kpt):
        y = pred_kpt.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y
    
    

