import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

from nets.nn import ComputeLoss, non_max_suppression, wh2xy
from visualization import plot_training_curves
from torch.cuda.amp import GradScaler, autocast



# compute average precision
def compute_ap(tp, conf, pred_cls, target_cls, eps=1e-16):
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


# compute the iou for 2 boxes
def box_iou(box1, box2):
    # intersection(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = intersection / (area1 + area2 - intersection)
    box1 = box1.T
    box2 = box2.T

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / (area1[:, None] + area2 - intersection)


# compute mAP
def mAP(pred, target, scale):
    metrics = []
    
    # Configure
    iou_v = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()
    
    # NMS
    target[:, 2:] *= scale
    pred = non_max_suppression(pred, 0.001, 0.65)

    # Metrics
    for i, output in enumerate(pred):
        labels = target[target[:, 0] == i, 1:]
        correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

        if output.shape[0] == 0:
            if labels.shape[0]:
                metrics.append((correct, *torch.zeros((3, 0)).cuda()))
            continue

        detections = output.clone()

        # Evaluate
        if labels.shape[0]:
            tbox = labels[:, 1:5].clone()  # target boxes
            tbox = wh2xy(tbox)

            correct = np.zeros((detections.shape[0], iou_v.shape[0])).astype(bool)

            t_tensor = torch.cat((labels[:, 0:1], tbox), 1)
            iou = box_iou(t_tensor[:, 1:], detections[:, :4])
            correct_class = t_tensor[:, 0:1] == detections[:, 5]
            for j in range(len(iou_v)):
                x = torch.where((iou >= iou_v[j]) & correct_class)
                if x[0].shape[0]:
                    matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                    matches = matches.cpu().numpy()
                    if x[0].shape[0] > 1:
                        matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), j] = True
            correct = torch.tensor(correct, dtype=torch.bool, device=iou_v.device)
        metrics.append((correct, output[:, 4], output[:, 5], labels[:, 0]))
     # Compute metrics
    metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        return compute_ap(*metrics)

    return np.array([0])



# Lambda function for learning rate warm-up
def lr_lambda(step, warmup_steps):
    if (step+1) < warmup_steps:
        return (step+1) / warmup_steps  # Gradual increase from 0 to 1
    return 1  # Continue with normal learning rate after warm-up


@torch.no_grad()
def feedforward(data_loader, model):
    model.eval()
    
    criterion = ComputeLoss(model)
    device = next(model.parameters()).device
    epoch_loss = 0.0
    epoch_mAP = 0.0
    
    with tqdm(total=len(data_loader)) as pbar:
        # Iterate over the dataset
       for i, (X, Y) in enumerate(data_loader):
           # move to device
           X = X.to(device)
           Y = Y.to(device)
            
           # Forward
           with autocast(dtype=torch.float16):
               output, box = model(X)  # forward
               loss_score = criterion(output, Y)
               
               map_score = mAP(box, Y, X.shape[-1])
               
           # Add the loss to the list
           epoch_loss += loss_score.item()
           
           # compute mAP
           epoch_mAP += map_score.item()
           
           # Update tqdm description with loss and accuracy
           pbar.set_postfix({
                'Loss': f'{epoch_loss/(i+1):.3f}', 
                'mAP': f'{epoch_mAP/(i+1):.3f}'
           })
           pbar.update(1)
           
           torch.cuda.empty_cache()
           
    # averaging over all batches
    epoch_loss /= len(data_loader)
    epoch_mAP /= len(data_loader)
    
    return epoch_mAP, epoch_loss



def backpropagation(data_loader, model, optimizer, scaler, mAP_skip=1):
    model.train()
    
    criterion = ComputeLoss(model)
    device = next(model.parameters()).device
    epoch_loss = 0.0
    epoch_mAP = 0.0
    
    with tqdm(total=len(data_loader)) as pbar:
        # Iterate over the dataset
       for i, (X, Y) in enumerate(data_loader):
           # move to device
           X = X.to(device)
           Y = Y.to(device)

           # Forward
           with autocast(dtype=torch.float16):
               output, box = model(X)  # forward
               loss_score = criterion(output, Y)
               
               map_score = torch.tensor([0])
               if i % mAP_skip == 0: # only compute mAP when necessary
                   map_score = mAP(box, Y, X.shape[-1])
               
           # Add the loss to the list
           epoch_loss += loss_score.item()
           
           # compute mAP
           epoch_mAP += map_score.item()
           
           # Reset gradients
           optimizer.zero_grad()
           
           # Backpropagate the loss
           scaler.scale(loss_score).backward()
           
           # Optimization step
           scaler.step(optimizer)
           
           # Updates the scale for next iteration.
           scaler.update()
           
           # Update tqdm description with loss and accuracy
           pbar.set_postfix({
                'Loss': f'{epoch_loss/(i+1):.3f}', 
                'mAP': f'{epoch_mAP/(i+1)*mAP_skip:.3f}'
           })
           pbar.update(1)
           
           torch.cuda.empty_cache()
    
    # averaging over all batches
    epoch_loss /= len(data_loader)
    epoch_mAP /= len(data_loader) / mAP_skip
    
    return epoch_mAP, epoch_loss


# model training loop
def model_training(train_loader, valid_loader, model):
    # Define hyperparameters
    n_epochs = 100
    warmup_steps = 3
    
    # Learning rate for training
    learning_rate = 1e-4
    
    # l2 regularization
    weight_decay = 5e-4
    
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=learning_rate)
    
    # Create a learning scheduler
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, warmup_steps))

    # Creates a GradScaler
    scaler = GradScaler()

    # Early Stopping criteria
    patience = 3
    not_improved = 0
    threshold = 0.01
    best_valid_loss = float('inf')
    
    # Training loop
    train_loss_curve, train_mAP_curve = [], []
    valid_loss_curve, valid_mAP_curve = [], []
    for epoch in range(1, n_epochs):
        print(f"Epoch {epoch}/{n_epochs}")
        train_mAP, train_loss = backpropagation(train_loader, model, optimizer, scaler, mAP_skip=16)
        valid_mAP, valid_loss = feedforward(valid_loader, model)
        
        # Step the scheduler after each epoch
        scheduler.step()
        
        train_loss_curve.append(train_loss)
        train_mAP_curve.append(train_mAP)
        valid_loss_curve.append(valid_loss)
        valid_mAP_curve.append(valid_mAP)

        # evaluate the current preformance
        if valid_loss - threshold < best_valid_loss:
            best_valid_loss = valid_loss
            not_improved = 0
            
            # save the best model in float16
            model.half()
            torch.save(model.state_dict(), model.name() + '.pth')
            model.float()
            
        else:
            not_improved += 1
            if not_improved >= patience:
                print('Early Stopping Activated')
                break
            
    plot_training_curves(train_mAP_curve, train_loss_curve, valid_mAP_curve, valid_loss_curve)
    