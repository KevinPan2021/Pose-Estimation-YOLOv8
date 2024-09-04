import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import sys
import os
import numpy as np
import pickle
import random
#import torch_tensorrt
sys.path.append(os.path.join(os.path.dirname(__file__), 'nets'))

from nn import YOLO, non_max_suppression, xy2wh
from dataset import Pose_Dataset
from visualization import plot_image, plot_images
from training import model_training


def compute_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
    

# get the keypoint and skeleton color
def get_color():
    # Define color palettes and skeleton
    palette = np.array(
        [[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
         [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
         [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
         [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]], dtype=np.uint8
    )

    COCO_SKELETON = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
        [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
    ]
    kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    return palette, kpt_color, limb_color, COCO_SKELETON


# Setup random seed.
def setup_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    

# the official model from ultralytics is in .pt format, convert it to .pth
def convert_pt_to_pth(path):
    model = torch.load(path, map_location='cpu')['model'].float()
    model.half()
    torch.save(model.state_dict(), path.split('/')[-1] + 'h')

    

# model inference
@torch.no_grad
def inference(model, img):
    model.eval()
    
    # move to device
    img = img.to(compute_device())
    
    # unsqueeze batch dimension
    img = img.unsqueeze(0)
    
    with autocast(dtype=torch.float16):
        output, _ = model(img)

    # apply nms
    # pred shape -> batch_size * torch.Size([(num_bbox, 6)])
    output = non_max_suppression(output, model.head.nc, 0.6, 0.65)[0]
    
    if output.size():
        box_output = output[:, :6]
        kps_output = output[:, 6:].view(len(output), *model.head.kpt_shape)
    else:
        box_output = output[:, :6]
        kps_output = output[:, 6:]

    _, _, w, h = img.shape

    # clip to range and rescale
    box_output = xy2wh(box_output[:, :4])
    box_output[:, [0, 2]] = box_output[:, [0, 2]].clip(0, w - 1E-3) / w
    box_output[:, [1, 3]] = box_output[:, [1, 3]].clip(0, h - 1E-3) / h
    
    kps_output[..., 0] = kps_output[..., 0].clip(0, w - 1E-3) / w
    kps_output[..., 1] = kps_output[..., 1].clip(0, w - 1E-3) / h
    
    # convert to numpy
    box_output = box_output.cpu().numpy()
    kps_output = kps_output.cpu().numpy()

    return box_output, kps_output
    

def main():    
    setup_seed()
    data_path = '../Datasets/MS-COCO/'
    
    image_size = 640
    
    # Creating train dataset object 
    train_dataset = Pose_Dataset( 
        image_dir = data_path + "train2017/",
        label_dir = data_path + "annotations_trainval2017/person_keypoints_train2017.json",
        augment = True,
        input_size = image_size
    )
    
    # Creating valid dataset object
    valid_dataset = Pose_Dataset( 
        image_dir = data_path + "val2017/",
        label_dir = data_path + "annotations_trainval2017/person_keypoints_val2017.json",
        augment = False,
        input_size = image_size
    )
    
    # get the color and skeleton
    colors = get_color()
    with open('categories.pkl', 'rb') as f:
        categories = pickle.load(f)
    
    
    # visualize some train examples (with augmentation)
    for i in range(0, len(train_dataset), len(train_dataset)//6):
        image, classes, bboxes, keypoints, _ = train_dataset[i]
        
        # Plotting the image with the bounding boxes 
        plot_image(image.permute(1,2,0), bboxes, keypoints, categories, colors)
    
    
    # visualize some valid examples (without augmentation)
    for i in range(0, len(valid_dataset), len(valid_dataset)//6):
        image, classes, bboxes, keypoints, _ = valid_dataset[i]

        # Plotting the image with the bounding boxes 
        plot_image(image.permute(1,2,0), bboxes, keypoints, categories, colors)
    
    
    # Create data loaders for training and validation sets
    train_loader = DataLoader(
        train_dataset, batch_size=32, num_workers=4, pin_memory=True,
        persistent_workers=True, shuffle=True, collate_fn=Pose_Dataset.collate_fn
    )
    
    val_loader = DataLoader(
        valid_dataset, batch_size=64, num_workers=4, pin_memory=True,
        persistent_workers=True, shuffle=False, collate_fn=Pose_Dataset.collate_fn
    )  
    
    
    # converting the pretrain .pt model to .pth
    #convert_pt_to_pth('../pretrained_models/YOLO_v8/v8_m_pose.pt')
    
    model = YOLO(size='n', num_classes=1)
    model = model.to(compute_device())
    
    #model.load_state_dict(torch.load(f'v8_{model.size}_pose.pth')) 
    
    # model training
    model_training(train_loader, val_loader, model)
    
    # load the trained model
    model.load_state_dict(torch.load(model.name() + '_pose.pth'))
    # load the converted official model
    #model.load_state_dict(torch.load(f'v8_{model.size}_pose.pth')) 
    
    # inference using validation dataset
    for i in range(0, len(valid_dataset), len(valid_dataset)//6):
        # extract x, y
        image, classes, bboxes, keypoints, _ = valid_dataset[i]
        
        # inferece
        pred_bboxes, pred_keypoints = inference(model, image)
        
        # Plotting the image with the bounding boxes 
        plot_images(image.permute(1,2,0), bboxes, keypoints, pred_bboxes, pred_keypoints, categories, colors)
    
    
if __name__ == "__main__": 
    main()
