import torch
from tqdm import tqdm
import numpy as np
import math
import copy

from nets.nn import non_max_suppression, wh2xy, xy2wh
from nets.loss import ComputeLoss, KPT_SIGMA
from visualization import plot_training_curves
from torch.cuda.amp import GradScaler, autocast



class ComputeMAP():
    def __init__(self, pred, target, nc, shape):
        # extract x_det (bounding box) from predict and bbox for target
        self.pred = pred
        self.target = target
        self.nc = nc
        _, _, self.w, self.h = shape
    
    
    def compute_metric(self, output, target, iou_v, pred_kpt=None, true_kpt=None):
        if pred_kpt is not None and true_kpt is not None:
            # `0.53` is from https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384
            area = xy2wh(target[:, 1:])[:, 2:].prod(1) * 0.53
            # (N, M, 17)
            d_x = (true_kpt[:, None, :, 0] - pred_kpt[..., 0]) ** 2
            d_y = (true_kpt[:, None, :, 1] - pred_kpt[..., 1]) ** 2
            sigma = torch.tensor(KPT_SIGMA.copy(), device=true_kpt.device, dtype=true_kpt.dtype)  # (17, )
            kpt_mask = true_kpt[..., 2] != 0  # (N, 17)
            e = (d_x + d_y) / (2 * sigma) ** 2 / (area[:, None, None] + 1e-7) / 2  # from coco-eval
            iou = (torch.exp(-e) * kpt_mask[:, None]).sum(-1) / (kpt_mask.sum(-1)[:, None] + 1e-7)
        else:
            # intersection(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
            (a1, a2) = target[:, 1:].unsqueeze(1).chunk(2, 2)
            (b1, b2) = output[:, :4].unsqueeze(0).chunk(2, 2)
            intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
            # IoU = intersection / (area1 + area2 - intersection)
            iou = intersection / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - intersection + 1e-7)
    
        correct = np.zeros((output.shape[0], iou_v.shape[0]), dtype=bool)
        for i in range(len(iou_v)):
            # IoU > threshold and classes match
            x = torch.where((iou >= iou_v[i]) & (target[:, 0:1] == output[:, 5]))
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1),
                                     iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=output.device)


    
    # compute average precision
    def compute_ap(self, tp, conf, pred_cls, target_cls, eps=1e-16):
        # Sort by object-ness
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    
        # Find unique classes
        unique_classes, nt = np.unique(target_cls, return_counts=True)
        nc = unique_classes.shape[0]  # number of classes, number of detections
    
        # Create Precision-Recall curve and compute AP for each class
        p = np.zeros((nc, 1000))
        r = np.zeros((nc, 1000))
        ap = np.zeros((nc, tp.shape[1]))
        px = np.linspace(0, 1, 1000)
        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            nl = nt[ci]  # number of labels
            no = i.sum()  # number of outputs
            if no == 0 or nl == 0:
                continue
    
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)
    
            # Recall
            recall = tpc / (nl + eps)  # recall curve
            # negative x, xp because xp decreases
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)
    
            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score
    
            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                m_rec = np.concatenate(([0.0], recall[:, j], [1.0]))
                m_pre = np.concatenate(([1.0], precision[:, j], [0.0]))
    
                # Compute the precision envelope
                m_pre = np.flip(np.maximum.accumulate(np.flip(m_pre)))
    
                # Integrate area under curve
                x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
                ap[ci, j] = np.trapz(np.interp(x, m_rec, m_pre), x)  # integrate
    
        mean_ap = ap.mean()
        return mean_ap

    
    
    # compute mAP
    def mAP(self):
        box_metrics = []
        kpt_metrics = []
        
        # Configure
        iou_v = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
        n_iou = iou_v.numel()
        scale = torch.tensor((self.w, self.h, self.w, self.h)).cuda()

        # NMS
        outputs = non_max_suppression(self.pred, self.nc, conf_threshold=0.001, iou_threshold=0.7)
    
        # Metrics
        for i, output in enumerate(outputs):            
            idx = self.target['idx'] == i
            cls = self.target['cls'][idx]
            box = self.target['box'][idx]
            kpt = self.target['kpt'][idx]

            cls = cls.cuda()
            box = box.cuda()
            kpt = kpt.cuda()
            
            correct_box = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()  # init
            correct_kpt = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()  # init

            if output.shape[0] == 0:
                if cls.shape[0]:
                    box_metrics.append((correct_box,
                                        *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                    kpt_metrics.append((correct_kpt,
                                        *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                continue
    
            pred = output.clone()
            p_kpt = pred[:, 6:].view(output.shape[0], kpt.shape[1], -1)
            
            # Evaluate
            if cls.shape[0]:
                t_box = wh2xy(box) # (x1, y1, x2, y2)
                
                t_kpt = kpt.clone()
                t_kpt[..., 0] *= self.w
                t_kpt[..., 1] *= self.h

                target = torch.cat((cls, t_box * scale), 1)  # native-space labels

                correct_box = self.compute_metric(pred[:, :6], target, iou_v)
                correct_kpt = self.compute_metric(pred[:, :6], target, iou_v, p_kpt, t_kpt)
            # Append
            box_metrics.append((correct_box, output[:, 4], output[:, 5], cls.squeeze(-1)))
            kpt_metrics.append((correct_kpt, output[:, 4], output[:, 5], cls.squeeze(-1)))
        
        # Compute metrics
        box_metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*box_metrics)]  # to numpy
        kpt_metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*kpt_metrics)]  # to numpy
        if len(box_metrics) and box_metrics[0].any():
            box_mean_ap = self.compute_ap(*box_metrics)
        else:
            box_mean_ap = np.array([0])
        if len(kpt_metrics) and kpt_metrics[0].any():
            kpt_mean_ap = self.compute_ap(*kpt_metrics)
        else:
            kpt_mean_ap = np.array([0])
        
        return box_mean_ap, kpt_mean_ap
    

def learning_rate(epochs, lrf):
    def fn(x):
        return (1 - x / epochs) * (1.0 - lrf) + lrf

    return fn

    
def clip_gradients(model, max_norm=10.0):
    parameters = model.parameters()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
    
    
# Exponential Moving Average (EMA)
# Keeps a moving average of everything in the model state_dict
class EMA:
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        if hasattr(model, 'module'):
            model = model.module
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()
                     

@torch.no_grad()
def feedforward(data_loader, model):
    model.eval()
    
    criterion = ComputeLoss(model)
    device = next(model.parameters()).device
    epoch_loss = 0.0
    epoch_bbox_mAP = 0.0
    epoch_keypoint_mAP = 0.0
    
    with tqdm(total=len(data_loader)) as pbar:
        # Iterate over the dataset
       for i, (X, Y) in enumerate(data_loader):
           # move to device
           X = X.to(device)
            
           # Forward
           with autocast(dtype=torch.float16):
               output, decoded = model(X)  # forward
               loss_score = criterion(decoded, Y)
               
               bbox_mAP, keypoints_mAP = ComputeMAP(output, Y, model.head.nc, X.shape).mAP()
               
           # Add the loss to the list
           epoch_loss += loss_score.item()
           
           # compute mAP
           epoch_bbox_mAP += bbox_mAP.item()
           epoch_keypoint_mAP += keypoints_mAP.item()
           
           # Update tqdm description with loss and accuracy
           pbar.set_postfix({
                'Loss': f'{epoch_loss/(i+1):.3f}', 
                'bbox_mAP': f'{epoch_bbox_mAP/(i+1):.3f}',
                'kpt_mAP': f'{epoch_keypoint_mAP/(i+1):.3f}'
           })
           pbar.update(1)
           
           torch.cuda.empty_cache()
           
    # averaging over all batches
    epoch_loss /= len(data_loader)
    epoch_bbox_mAP /= len(data_loader)
    epoch_keypoint_mAP /= len(data_loader)
    
    return epoch_bbox_mAP, epoch_keypoint_mAP, epoch_loss



def backpropagation(data_loader, model, optimizer, scaler, ema, mAP_skip=1):
    model.train()
    
    criterion = ComputeLoss(model)
    device = next(model.parameters()).device
    epoch_loss = 0.0
    epoch_bbox_mAP = 0.0
    epoch_keypoint_mAP = 0.0
    
    with tqdm(total=len(data_loader)) as pbar:
        # Iterate over the dataset
       for i, (X, Y) in enumerate(data_loader):
           # move to device
           X = X.to(device)
  
           # Forward
           with autocast(dtype=torch.float16):
               output, decoded = model(X)  # forward
               loss_score = criterion(decoded, Y)
               
               bbox_mAP, keypoints_mAP = torch.tensor([0]), torch.tensor([0])
               if i % mAP_skip == 0: # only compute mAP when necessary
                   bbox_mAP, keypoints_mAP = ComputeMAP(output, Y, model.head.nc, X.shape).mAP()
                  
           # Add the loss to the list
           epoch_loss += loss_score.item()
           
           # compute mAP
           epoch_bbox_mAP += bbox_mAP.item()
           epoch_keypoint_mAP += keypoints_mAP.item()
           
           # Reset gradients
           optimizer.zero_grad()
           
           # Backpropagate the loss
           scaler.scale(loss_score).backward()
           
           # clip gradients
           #clip_gradients(model)  
           
           # Optimization step
           scaler.step(optimizer)
           
           # Updates the scale for next iteration.
           scaler.update()
           
           # update EMA
           ema.update(model)
                        
           # Update tqdm description with loss and accuracy
           pbar.set_postfix({
                'Loss': f'{epoch_loss/(i+1):.3f}', 
                'bbox_mAP': f'{epoch_bbox_mAP/(i+1)*mAP_skip:.3f}',
                'kpt_mAP': f'{epoch_keypoint_mAP/(i+1)*mAP_skip:.3f}'
           })
           pbar.update(1)
           
           torch.cuda.empty_cache()
    
    # averaging over all batches
    epoch_loss /= len(data_loader)
    epoch_bbox_mAP /= len(data_loader) / mAP_skip
    epoch_keypoint_mAP /= len(data_loader) / mAP_skip
    
    return epoch_bbox_mAP, epoch_keypoint_mAP, epoch_loss


# model training loop
def model_training(train_loader, valid_loader, model):
    # Define hyperparameters
    n_epochs = 100
    warmup_steps = 3
    
    # Learning rate for training
    lr0 = 1e-2 # initial learning rate (SGD=1E-2, Adam=1E-3)
    lrf = 1e-2 # final OneCycleLR learning rate (lr0 * lrf)

    warmup_bias_lr = 0.10   # warmup initial bias lr
    momentum = 0.937
    warmup_momentum = 0.8
    
    # l2 regularization
    weight_decay = 5e-4
    
    # create optimizer
    p = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
            p[2].append(v.bias)
        if isinstance(v, torch.nn.BatchNorm2d):
            p[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
            p[0].append(v.weight)

    optimizer = torch.optim.SGD(p[2], lr0, momentum, nesterov=True)

    optimizer.add_param_group({'params': p[0], 'weight_decay': weight_decay})
    optimizer.add_param_group({'params': p[1]})
    del p
    
    # Create a learning scheduler
    lr = learning_rate(n_epochs, lrf)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr, last_epoch=-1)
    
    # EMA
    ema = EMA(model)
    
    # Creates a GradScaler
    scaler = GradScaler()

    # Early Stopping criteria
    best_valid_loss = float('inf')
    
    # Training loop
    train_loss_curve, train_bbox_mAP_curve, train_kpt_mAP_curve = [], [], []
    valid_loss_curve, valid_bbox_mAP_curve, valid_kpt_mAP_curve = [], [], []
    for epoch in range(n_epochs):
        print(f"Epoch: {epoch+1}/{n_epochs}")
        
        # Warmup epoch
        if epoch <= warmup_steps:
            xp = [0, warmup_steps]
            for j, y in enumerate(optimizer.param_groups):
                if j == 0:
                    y['lr'] = np.interp(epoch, xp, [warmup_bias_lr, y['initial_lr'] * lr(epoch)])
                else:
                    y['lr'] = np.interp(epoch, xp, [0.0, y['initial_lr'] * lr(epoch)])
                if 'momentum' in y:
                    y['momentum'] = np.interp(epoch, xp, [warmup_momentum, momentum])
        
        
        # turning off the mosaic augmentation during last 10 epochs
        if epoch == n_epochs - 10:
            train_loader.dataset.close_mosaic()
        
        # back and forward propagation
        train_bbox_mAP, train_kpt_mAP, train_loss = backpropagation(train_loader, model, optimizer, scaler, ema, mAP_skip=8)
        valid_bbox_mAP, valid_kpt_mAP, valid_loss = feedforward(valid_loader, model)

        # Step the scheduler after each epoch
        scheduler.step()
        
        train_loss_curve.append(train_loss)
        train_bbox_mAP_curve.append(train_bbox_mAP)
        train_kpt_mAP_curve.append(train_kpt_mAP)
        
        valid_loss_curve.append(valid_loss)
        valid_bbox_mAP_curve.append(valid_bbox_mAP)
        valid_kpt_mAP_curve.append(valid_kpt_mAP)

        # evaluate the current preformance
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            
            # save the best model in float16
            model.half()
            torch.save(model.state_dict(), model.name() + '_pose.pth')
            model.float()

            
    plot_training_curves(
        train_bbox_mAP_curve, train_kpt_mAP_curve, train_loss_curve, 
        valid_bbox_mAP_curve, valid_kpt_mAP_curve, valid_loss_curve
    )
    