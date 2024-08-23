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


# Pose Estimation dataset 
class Pose_Estimation_Dataset(data.Dataset):
    def __init__(self, image_dir, label_dir, augment=False, input_size=640):
        
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
        self.data, self.categories = self.load_label(label_dir)
        self.image_dir = image_dir
    
    
    # overwriting the len method
    def __len__(self):
        return len(self.data)
    
    
    # get the keypoint and skeleton color
    @staticmethod
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
    
    
    # overwriting the get item method
    # bbox shape: (x, y, w, h), range: [0,1]
    def __getitem__(self, index):
        item = self.data[index]
        # mosaic augmentation
        if random.random() < self.prob['mosaic']:
            img, cls, bbox, keypoint = self.load_mosaic(item)
        # mixup augmentation
        elif random.random() < self.prob['mixup']:
            img, cls, bbox, keypoint = self.load_mixup(item)
        # neither mosaic nor mixup
        else:
            img, cls, bbox, keypoint = self.load_image(item)
        
        # rescale bbox and keypoint to [0, 1]
        bbox /= self.input_size
        keypoint[..., :2] = keypoint[..., :2]/self.input_size
        
        return img, cls, bbox, keypoint, self.categories, torch.zeros(bbox.shape[0])
    
    
    # overwriting the collate_fn method
    @staticmethod
    def collate_fn(batch):
        samples, cls, box, kpt, _, indices = zip(*batch)

        cls = torch.cat(cls, 0)
        box = torch.cat(box, 0)
        kpt = torch.cat(kpt, 0)
        
        # index number in batch
        new_indices = list(indices)
        for i in range(len(indices)):
            new_indices[i] += i
        indices = torch.cat(new_indices, 0)

        targets = {'cls': cls, 'box': box, 'kpt': kpt, 'idx': indices}
        return torch.stack(samples, 0), targets

    
    @staticmethod
    def load_label(path):
        # Load the JSON file
        with open(path, 'r') as file:
            coco_data = json.load(file)

        # Extract image information and category information
        image_info = {image['id']: image['file_name'] for image in coco_data['images']}
        categories = {category['name']: category['keypoints'] for category in coco_data['categories']}
        
        # Create a dictionary to store image information with bounding boxes and their classes
        image_bboxes_dict = {}
        for annotation in coco_data['annotations']:
            # no visible keypoints (delete the bounding box and keypoints)
            if annotation['num_keypoints'] == 0:
                continue
            
            # extract bbox
            bbox = annotation['bbox']
            bbox = [0] + bbox # add class label (person)
            
            # width and height has to be greater than 0
            if not (bbox[-2] > 0 and bbox[-1] > 0):
                continue
            
            # extract keypoints
            keypoints = annotation['keypoints']
            keypoints = np.array(keypoints).reshape(-1, 3)
            file_name = image_info[annotation['image_id']]
            
            # Check if the image is already in the dictionary by file name
            if file_name in image_bboxes_dict:
                # Add the bounding box and keypoints to the existing image entry
                image_bboxes_dict[file_name]['bboxes'].append(bbox)
                image_bboxes_dict[file_name]['keypoints'].append(keypoints)
            else:
                # Create a new entry for the image
                image_bboxes_dict[file_name] = {
                    'file_name': file_name,
                    'keypoints': [keypoints],
                    'bboxes': [bbox]
                }
        
        # Convert the dictionary to a list
        image_bboxes = list(image_bboxes_dict.values())
        return image_bboxes, categories
    
    
    
    # load image (neither mosaic nor mixup)
    def load_image(self, item):
        image = np.array(Image.open(os.path.join(self.image_dir, item['file_name'])).convert("RGB"))

        # Albumentations
        aug = Albumentations(self.input_size, self.prob)
        image, bboxes, keypoints = aug(image, np.vstack(item['bboxes']), np.vstack(item['keypoints']))
        
        bboxes = torch.from_numpy(bboxes)
        cls = bboxes[:, 0].unsqueeze(-1)
        bboxes = bboxes[:, 1:]
        
        keypoints = torch.from_numpy(keypoints).float()
        return image, cls, bboxes, keypoints
    
    
    # load with mosaic (concatenate 4 images to form 1 image)
    def load_mosaic(self, item):
        boxes4 = []
        keypoints4 = []
        cls4 = []
        image4 = torch.zeros((3, self.input_size, self.input_size))
        
        # Select 4 random images (including the original)
        indices = [item] + random.choices(self.data, k=3)

        for i, index in enumerate(indices):
            # Load image
            image = np.array(Image.open(os.path.join(self.image_dir, index['file_name'])).convert("RGB"))
            
            # Albumentations
            aug = Albumentations(self.input_size//2, self.prob)
            image, boxes, kps = aug(image, np.vstack(index['bboxes']), np.vstack(index['keypoints']))
            
            # Convert bounding boxes to numpy array
            boxes = np.array(boxes)
            cls = boxes[:, 0:1]
            boxes = boxes[:, 1:]
            kps = np.array(kps)
            
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
            if len(boxes) > 0:
                # change the box position coordinate
                boxes[:, 0] += y1
                boxes[:, 1] += x1
                
                # change the keypoints position coordinates
                kps[..., 0] += y1
                kps[..., 1] += x1

                # Add the image's labels to the list
                boxes4.append(torch.from_numpy(boxes))
                keypoints4.append(torch.from_numpy(kps))
                cls4.append(torch.tensor(cls))
                
        # Stack all labels for the mosaic image
        boxes4 = torch.cat(boxes4, dim=0)
        keypoints4 = torch.cat(keypoints4, dim=0).float()
        cls4 = torch.cat(cls4, dim=0)
        
        return image4, cls4, boxes4, keypoints4
        
    
    # load with MixUp (merge 2 images)
    def load_mixup(self, item1):
        item2 = random.choices(self.data, k=1)[0]

        image1 = np.array(Image.open(os.path.join(self.image_dir, item1['file_name'])).convert("RGB"))
        image2 = np.array(Image.open(os.path.join(self.image_dir, item2['file_name'])).convert("RGB"))
        
        # Albumentations
        aug = Albumentations(self.input_size, self.prob)
        image1, bboxes1, keypoints1 = aug(image1, np.vstack(item1['bboxes']), np.vstack(item1['keypoints']))
        
        aug = Albumentations(self.input_size, self.prob)
        image2, bboxes2, keypoints2 = aug(image2, np.vstack(item2['bboxes']), np.vstack(item2['keypoints']))
        
        # mixup
        alpha = np.random.beta(32.0, 32.0)  # mix-up ratio, alpha=beta=32.0
        image = image1 * alpha + image2 * (1 - alpha)
        
        bboxes = np.concatenate((bboxes1, bboxes2), 0)
        keypoints = np.concatenate((keypoints1, keypoints2), 0)
        
        bboxes = torch.from_numpy(bboxes)
        cls = bboxes[:, 0:1]
        bboxes = bboxes[:, 1:]
        
        keypoints = torch.from_numpy(keypoints).float()
        
        return image, cls, bboxes, keypoints
    
    
       
class Albumentations:
    def __init__(self, image_size, prob):        
        self.transform = album.Compose([
            album.LongestMaxSize(max_size=image_size), 
            album.PadIfNeeded(
                min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT 
            ), 
            album.HorizontalFlip(p=prob['hfilp']),
            album.Blur(p=prob['blur']),
            album.CLAHE(p=prob['clahe']),
            album.ToGray(p=prob['gray']),
            album.Perspective(scale=(0.05, 0.05), p=prob['perspective']),
            album.HueSaturationValue(hue_shift_limit=0.3, sat_shift_limit=70, val_shift_limit=40, p=prob['hsv']),
            album.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
            ToTensorV2()],
            bbox_params=album.BboxParams(format="coco", label_fields=['class_labels']), # COCO format = [x_min, y_min, width, height]
            keypoint_params=album.KeypointParams(format='xy', remove_invisible=False) 
        )
    

    def __call__(self, image, bboxes, keypoints):
        # Apply the transformations
        x = self.transform(image=image, bboxes=bboxes[:, 1:], class_labels=bboxes[:, 0], keypoints=keypoints[:, :2])

        # Filter keypoints invisible
        kps, visible = x['keypoints'], keypoints[:, -1]
        
        # Remove keypoints that correspond to removed bounding boxes
        filtered_bboxes = []
        filtered_keypoints = []
        filtered_visible = []
        for i, bbox in enumerate(x['bboxes']):
            # Check if the bbox is removed
            if len(bbox) == 0:  
                continue
 
            filtered_bboxes.append([x['class_labels'][i], *bbox])
            filtered_keypoints.append(kps[i * 17:(i + 1) * 17])
            filtered_visible.append(visible[i * 17:(i + 1) * 17])

        image = x['image']
        bboxes = np.array(filtered_bboxes)
        keypoints = np.array([[*k, v] for kp, vis in zip(filtered_keypoints, filtered_visible) for k, v in zip(kp, vis)])

        # Reshape the keypoints from (num * 17, 3) to (num, 17, 3)
        keypoints = keypoints.reshape(-1, 17, 3)
        
        # convert bbox from xy to centerXY
        if bboxes.size != 0:
            bboxes[:, 1:] = self.xy2center(bboxes[:, 1:])
        
        # Handle cases where perspective transformation could make label size = (0, )
        if bboxes.size == 0:
            bboxes = np.zeros((0, 5))
        if keypoints.size == 0:
            keypoints = np.zeros((0, 17, 3))
        
        return image, bboxes, keypoints
    
    
    # convert from (x_min, y_min, w, h) to (centerX, centerY, w, h)
    @staticmethod 
    def xy2center(x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] + x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] + x[..., 3] / 2  # top left y
        return y
    
