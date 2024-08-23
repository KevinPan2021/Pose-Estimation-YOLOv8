import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'nets'))

from nn import YOLO, non_max_suppression, xy2wh
from dataset import Pose_Estimation_Dataset
from visualization import plot_image, plot_images
from training import model_training


def compute_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
    

# bidirectional dictionary
class BidirectionalMap:
    def __init__(self):
        self.key_to_value = {}
        self.value_to_key = {}
    
    def __len__(self):
        return len(self.key_to_value)
    
    def add_mapping(self, key, value):
        self.key_to_value[key] = value
        self.value_to_key[value] = key

    def get_value(self, key):
        return self.key_to_value.get(key)

    def get_key(self, value):
        return self.value_to_key.get(value)


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
    output = non_max_suppression(output, model.head.nc, 0.4, 0.65)[0]
    
    if output.size():
        box_output = output[:, :6]
        kps_output = output[:, 6:].view(len(output), *model.head.kpt_shape)
    else:
        box_output = output[:, :6]
        kps_output = output[:, 6:]

    _, _, w, h = img.shape
    
    # convert to numpy
    box_output = box_output.cpu().numpy()
    kps_output = kps_output.cpu().numpy()

    # clip to range and rescale
    box_output = xy2wh(box_output[:, :4])
    box_output[:, [0, 2]] = box_output[:, [0, 2]].clip(0, w - 1E-3) / w
    box_output[:, [1, 3]] = box_output[:, [1, 3]].clip(0, h - 1E-3) / h
    
    kps_output[..., 0] = kps_output[..., 0].clip(0, w - 1E-3) / w
    kps_output[..., 1] = kps_output[..., 1].clip(0, w - 1E-3) / h
    
    return box_output, kps_output
    

def main():    
    data_path = '../Datasets/MS-COCO/'
    
    image_size = 640
    
    # Creating train dataset object 
    train_dataset = Pose_Estimation_Dataset( 
        image_dir = data_path + "train2017/",
        label_dir = data_path + "annotations_trainval2017/person_keypoints_train2017.json",
        augment = True,
        input_size = image_size
    )
    
    # Creating valid dataset object
    valid_dataset = Pose_Estimation_Dataset( 
        image_dir = data_path + "val2017/",
        label_dir = data_path + "annotations_trainval2017/person_keypoints_val2017.json",
        augment = False,
        input_size = image_size
    )
    
    # get the color and skeleton
    colors = valid_dataset.get_color()
    
    
    # visualize some train examples (with augmentation)
    for i in range(0, len(train_dataset), len(train_dataset)//6):
        image, classes, bboxes, keypoints, categories, _ = train_dataset[i]
        
        # Plotting the image with the bounding boxes 
        plot_image(image.permute(1,2,0), bboxes, keypoints, categories, colors)
    
    
    # visualize some valid examples (without augmentation)
    for i in range(0, len(valid_dataset), len(valid_dataset)//6):
        image, classes, bboxes, keypoints, categories, _ = valid_dataset[i]

        # Plotting the image with the bounding boxes 
        plot_image(image.permute(1,2,0), bboxes, keypoints, categories, colors)
        
    
    # Create data loaders for training and validation sets
    train_loader = DataLoader(
        train_dataset, batch_size=32, num_workers=4, pin_memory=True,
        persistent_workers=True, shuffle=True, collate_fn=Pose_Estimation_Dataset.collate_fn
    )
    
    val_loader = DataLoader(
        valid_dataset, batch_size=64, num_workers=4, pin_memory=True,
        persistent_workers=True, shuffle=False, collate_fn=Pose_Estimation_Dataset.collate_fn
    )  

    
    # converting the pretrain .pt model to .pth
    #convert_pt_to_pth('../pretrained_models/YOLO_v8/v8_m_pose.pt')
    
    model = YOLO(size='n', num_classes=1)
    model = model.to(compute_device())
    
    # model training
    model_training(train_loader, val_loader, model)
    
    # load the best model
    #model.load_state_dict(torch.load(model.name() + '.pth'))
    # load the converted official model
    model.load_state_dict(torch.load(f'v8_{model.size}_pose.pth')) 
    
    # inference using validation dataset
    for i in range(0, len(valid_dataset), len(valid_dataset)//6):
        # extract x, y
        image, classes, bboxes, keypoints, categories, _ = valid_dataset[i]
        
        # inferece
        pred_bboxes, pred_keypoints = inference(model, image)
        
        # Plotting the image with the bounding boxes 
        plot_images(image.permute(1,2,0), bboxes, keypoints, pred_bboxes, pred_keypoints, categories, colors)
    
    
if __name__ == "__main__": 
    main()
