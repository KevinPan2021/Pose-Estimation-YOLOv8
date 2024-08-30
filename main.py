import torch
from torch.utils.data import DataLoader
import pickle
from torch.cuda.amp import autocast
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'nets'))

from nn import YOLO, non_max_suppression, xy2wh
from dataset import Object_Detection_Dataset
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
        _, box = model(img)
    
    # apply nms
    # pred shape -> batch_size * torch.Size([(num_bbox, 6)])
    pred = non_max_suppression(box, 0.4, 0.65)[0]
    # convert from xyxy to xywh
    pred = pred.cpu().numpy()
    pred = xy2wh(pred)
    
    # normalize to [0, 1]
    # inplace clip
    _, _, w, h = img.shape
    pred[:, [0, 2]] = pred[:, [0, 2]].clip(0, w - 1E-3) / w
    pred[:, [1, 3]] = pred[:, [1, 3]].clip(0, h - 1E-3) / h
    
    # convert from [x,y,w,h,score,class] to [score,class,x,y,w,h]
    pred = pred[:, [4, 5, 0, 1, 2, 3]]
    return pred
    

def main():    
    data_path = '../Datasets/MS-COCO/'
    
    image_size = 640
    
    # Class labels 
    classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
        'remote', 'keyboard','cell phone', 'microwave', 'oven', 'toaster', 'sink', 
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
        'hair drier', 'toothbrush'
    ]
    class_labels = BidirectionalMap()
    for i in range(len(classes)):
        class_labels.add_mapping(i, classes[i])
    
    # Save the instance to a pickle file
    with open("class_ind_pair.pkl", "wb") as f:
        pickle.dump(class_labels, f)
    
    # Creating train dataset object 
    train_dataset = Object_Detection_Dataset( 
        image_dir = data_path + "train2017/",
        label_dir = data_path + "annotations_trainval2017/instances_train2017.json",
        class_labels_map = class_labels,
        augment = True,
        input_size = image_size
    )
    
    # Creating valid dataset object
    valid_dataset = Object_Detection_Dataset( 
        image_dir = data_path + "val2017/",
        label_dir = data_path + "annotations_trainval2017/instances_val2017.json", 
        class_labels_map = class_labels,
        augment = False,
        input_size = image_size
    )
    
    
    # visualize some train examples (with augmentation)
    for i in range(0, len(train_dataset), len(train_dataset)//6):
        # extract x, y
        image, target = train_dataset[i]
        
        # Plotting the image with the bounding boxes 
        plot_image(image.permute(1,2,0), target, class_labels)
        
    
    # visualize some valid examples (without augmentation)
    for i in range(0, len(valid_dataset), len(valid_dataset)//6):
        # extract x, y
        image, target = valid_dataset[i]
        
        # Plotting the image with the bounding boxes 
        plot_image(image.permute(1,2,0), target, class_labels)
    
    
    # Create data loaders for training and validation sets
    train_loader = DataLoader(
        train_dataset, batch_size=16, num_workers=4, pin_memory=True,
        persistent_workers=True, shuffle=True, collate_fn=Object_Detection_Dataset.collate_fn
    )

    val_loader = DataLoader(
        valid_dataset, batch_size=32, num_workers=4, pin_memory=True,
        persistent_workers=True, shuffle=False, collate_fn=Object_Detection_Dataset.collate_fn
    )  
    
    # converting the pretrain .pt model to .pth
    #convert_pt_to_pth('../pretrained_models/YOLO_v8/v8_m.pt')
    
    model = YOLO(size='m', num_classes=len(class_labels))
    model = model.to(compute_device())
    
    # model training
    model_training(train_loader, val_loader, model)
    
    # load the best model
    #model.load_state_dict(torch.load(model.name() + '.pth'))
    # load the converted official model
    #model.load_state_dict(torch.load(f'v8_{model.size}.pth')) 
    
    # inference using validation dataset
    for i in range(0, len(valid_dataset), len(valid_dataset)//6):
        # extract x, y
        image, target = valid_dataset[i]
        
        # inferece
        pred = inference(model, image)
        
        # Plotting the image with the bounding boxes 
        plot_images(image.permute(1,2,0), target, pred, class_labels)
    
    
if __name__ == "__main__": 
    main()
