import os
import random
import json
import albumentations as album
from albumentations.pytorch import ToTensorV2 
from PIL import Image 
import cv2
import numpy as np
import torch
from torch.utils import data


# custom object detection dataset 
class Object_Detection_Dataset(data.Dataset):
    def __init__(self, image_dir, label_dir, class_labels_map, augment=False, input_size=640):
        
        # augment probability
        self.prob = dict()
        # 50% prob mosaic, 10% prob mixup, 40% prob neither
        self.prob['mosaic'] = 0.5 if augment else 0
        self.prob['mixup'] = 0.2 if augment else 0
        
        self.prob['hfilp'] = 0.5 if augment else 0
        self.prob['blur'] = 0.05 if augment else 0
        self.prob['clahe'] = 0.05 if augment else 0
        self.prob['gray'] = 0.05 if augment else 0
        self.prob['perspective'] = 0.9 if augment else 0
        self.prob['hsv'] = 0.9 if augment else 0
        
        # input image size
        self.input_size = input_size 

        # Read labels
        self.data = self.load_label(label_dir, class_labels_map)
        self.image_dir = image_dir
    
    
    # overwriting the get item method
    def __getitem__(self, index):
        item = self.data[index]
        # mosaic augmentation
        if random.random() < self.prob['mosaic']:
            img, target = self.load_mosaic(item)
        # mixup augmentation
        elif random.random() < self.prob['mixup']:
            img, target = self.load_mixup(item)
        # neither mosaic nor mixup
        else:
            img, target = self.load_image(item)
        
        # target shape: (num_bbox, 6) -> (img_num, class_id, x_min, y_min, width, height)
        # rescale target to [0, 1]
        target[:, 2:] /= self.input_size
        return img, target
    
    
    # overwriting the len method
    def __len__(self):
        return len(self.data)
    
    
    # overwriting the collate_fn method
    @staticmethod
    def collate_fn(batch):
        samples, targets = zip(*batch)
        # add target image index, the 1st item in dim 6 = the image num
        for i, item in enumerate(targets):
            item[:, 0] = i  
        samples = torch.stack(samples, 0) # (batch, channel, width, height)
        targets = torch.cat(targets, 0) # (batch*boxes, 6)
        return samples, targets

    
    @staticmethod
    def load_label(path, class_labels):
        # Load the JSON file
        with open(path, 'r') as file:
            coco_data = json.load(file)

        # Extract image information and category information
        image_info = {image['id']: image['file_name'] for image in coco_data['images']}
        category_info = {category['id']: category['name'] for category in coco_data['categories']}
        
        # Create a dictionary to store image information with bounding boxes and their classes
        image_bboxes_dict = {}
        for annotation in coco_data['annotations']:
            bbox = annotation['bbox']
            class_id = class_labels.get_key(category_info[annotation['category_id']])
            file_name = image_info[annotation['image_id']]
            
            # Check if the image is already in the dictionary by file name
            if file_name in image_bboxes_dict:
                # Add the bounding box and class to the existing image entry
                image_bboxes_dict[file_name]['bboxes'].append([class_id] + bbox)
            else:
                # Create a new entry for the image
                image_bboxes_dict[file_name] = {
                    'file_name': file_name,
                    'bboxes': [[class_id] + bbox]
                }

        # Convert the dictionary to a list
        image_bboxes = list(image_bboxes_dict.values())
        
        return image_bboxes
    
    
    
    # load image (neither mosaic nor mixup)
    def load_image(self, item):
        image = np.array(Image.open(os.path.join(self.image_dir, item['file_name'])).convert("RGB"))

        # Albumentations
        image, bboxes = Albumentations(self.input_size, self.prob)(image, np.vstack(item['bboxes']))
        
        target = torch.zeros((bboxes.shape[0], 6))
        target[:, 1:] = torch.from_numpy(bboxes)
    
        return image, target
    
    
    # load with mosaic (concatenate 4 images to form 1 image)
    def load_mosaic(self, item):
        label4 = []
        image4 = torch.zeros((3, self.input_size, self.input_size))
        
        # Select 4 random images (including the original)
        indices = [item] + random.choices(self.data, k=3)

        for i, index in enumerate(indices):
            # Load image
            image = np.array(Image.open(os.path.join(self.image_dir, index['file_name'])).convert("RGB"))
            
            # Albumentations (assuming this returns transformed image and bboxes)
            image, bboxes = Albumentations(self.input_size//2, self.prob)(image, np.vstack(index['bboxes']))
    
            # Convert bounding boxes to numpy array
            bboxes = np.array(bboxes)
    
            # Determine placement coordinates in the mosaic
            if i == 0:  # top left
                x1, x2 = 0, self.input_size//2
                y1, y2 = 0, self.input_size//2
            elif i == 1:  # top right
                x1, x2 = self.input_size//2, self.input_size
                y1, y2 = 0, self.input_size//2
            elif i == 2:  # bottom left
                x1, x2 = 0, self.input_size//2
                y1, y2 = self.input_size//2, self.input_size
            elif i == 3:  # bottom right
                x1, x2 = self.input_size//2, self.input_size
                y1, y2 = self.input_size//2, self.input_size
    
            # Place the image in the mosaic
            image4[:, x1:x2, y1:y2] = image
    
            # Adjust bounding boxes for the new position
            if len(bboxes) > 0:
                bboxes[:, 1] += y1
                bboxes[:, 2] += x1
                
                # Add the image's labels to the list
                label4.append(torch.from_numpy(bboxes))
        
        # Stack all labels for the mosaic image
        label4 = torch.cat(label4, dim=0)
        
        # add 1 more dimension on labels
        target = torch.zeros((label4.shape[0], 6))
        target[:, 1:] = label4
        
        return image4, target
        
    
    # load with MixUp (merge 2 images)
    def load_mixup(self, item1):
        item2 = random.choices(self.data, k=1)[0]

        image1 = np.array(Image.open(os.path.join(self.image_dir, item1['file_name'])).convert("RGB"))
        image2 = np.array(Image.open(os.path.join(self.image_dir, item2['file_name'])).convert("RGB"))
        
        # Albumentations
        image1, bboxes1 = Albumentations(self.input_size, self.prob)(image1, np.vstack(item1['bboxes']))
        image2, bboxes2 = Albumentations(self.input_size, self.prob)(image2, np.vstack(item2['bboxes']))
        
        # mixup
        alpha = np.random.beta(32.0, 32.0)  # mix-up ratio, alpha=beta=32.0
        image = image1 * alpha + image2 * (1 - alpha)
        
         
        label = np.concatenate((bboxes1, bboxes2), 0)
        
        # add 1 more dimension on labels
        target = torch.zeros((label.shape[0], 6))
        target[:, 1:] = torch.tensor(label)
        
        return image, target
    
    
       
class Albumentations:
    def __init__(self, image_size, prob):        
        self.transform = album.Compose([
            # Rescale an image so that maximum side is equal to image_size 
            album.LongestMaxSize(max_size=image_size), 
            # Pad remaining areas with zeros 
            album.PadIfNeeded( 
                min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT 
            ), 
            album.HorizontalFlip(p=prob['hfilp']),
            album.Blur(p=prob['blur']),
            album.CLAHE(p=prob['clahe']),
            album.ToGray(p=prob['gray']),
            album.Perspective(scale=(0.05, 0.05), p=prob['perspective']),
            album.HueSaturationValue(hue_shift_limit=0.3, sat_shift_limit=70, val_shift_limit=40, p=prob['hsv']),
            
            # Normalize the image 
            album.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255), 
            # Convert the image to PyTorch tensor 
            ToTensorV2() 
        ], album.BboxParams(format="coco", label_fields=[]))
        # coco format = [x_min, y_min, width, height]
            
            
    def __call__(self, image, label):
        if self.transform:
            # filter out bounding boxes to ensure they have non-zero width and height
            label = self.filter_bboxes(label)

            x = self.transform(image=image, bboxes=label[:, 1:], class_labels=label[:, 0])
            
            image = x['image']
            label = np.array([[c, *b] for c, b in zip(x['class_labels'], x['bboxes'])])
            
            # perspective could make label size = (0, )
            # create an empty array for concatenation
            if label.size == 0:
                label = np.zeros((0, 5))
            
        return image, label
    
    
    @staticmethod
    def filter_bboxes(label):
        # Filter out bounding boxes with zero width or height
        adjusted_label = []
        for bbox in label:
            x_min, y_min, w, h = bbox[1:]
            if w > 0 and h > 0:
                adjusted_label.append(bbox)
        return np.array(adjusted_label)
    
    