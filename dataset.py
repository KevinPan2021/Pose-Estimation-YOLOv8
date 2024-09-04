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
import pickle
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="albumentations.core.transforms_interface")


from nets.nn import xy2wh

# Pose Estimation dataset 
class Pose_Dataset(data.Dataset):
    def __init__(self, image_dir, label_dir, augment=False, input_size=640):
        
        # augment probability
        self.prob = dict()
        
        # 1% prob mixup, 90% prob mosaic, 9% neither
        self.prob['mixup'] = 0.01 if augment else 0
        self.prob['mosaic'] = 0.91 if augment else 0
        
        self.prob['hfilp'] = 0.5 if augment else 0
        self.prob['blur'] = 0.01 if augment else 0
        self.prob['clahe'] = 0.01 if augment else 0
        self.prob['gray'] = 0.01 if augment else 0
        self.prob['medianblur'] = 0.01 if augment else 0
        self.prob['hsv'] = 1 if augment else 0
        
        # input image size
        self.input_size = input_size 
        
        # Read labels
        self.data = self.load_label(label_dir)
        self.image_dir = image_dir
    
    
    # overwriting the len method
    def __len__(self):
        return len(self.data)
    
    # overwriting the get item method
    # bbox shape: (x, y, w, h), range: [0,1]
    def __getitem__(self, index):
        item = self.data[index]
        
        # mixup augmentation
        if random.random() < self.prob['mixup']:
            img, cls_bbox, keypoint = self.load_mixup(item)

        # mosaic augmentation
        elif random.random() < self.prob['mosaic']:
            img, cls_bbox, keypoint = self.load_mosaic(item)
        # neither mosaic nor mixup
        else:
            img, cls_bbox, keypoint = self.load_image(item)
        
        # rescale bbox and keypoint to [0, 1]
        cls = cls_bbox[..., 0:1]
        bbox = cls_bbox[..., 1:] / self.input_size
        keypoint[..., :2] /= self.input_size
        
        return img, cls, bbox, keypoint, torch.zeros(bbox.shape[0])
    
    
    # overwriting the collate_fn method
    @staticmethod
    def collate_fn(batch):
        samples, cls, box, kpt, indices = zip(*batch)

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
        
        # save the categories
        if not os.path.exists('categories.pkl'):
            with open('categories.pkl', 'wb') as f:
                pickle.dump(categories, f)
            
        # Create a dictionary to store image information with bounding boxes and their classes
        image_bboxes_dict = {}
        for annotation in coco_data['annotations']:
            # no visible keypoints (delete the bounding box and keypoints)
            if annotation['num_keypoints'] == 0:
                continue
            
            # extract bbox
            bbox = [0] + annotation['bbox'] # add class label (person)
            
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
                    'keypoints': [keypoints], #  format: [x, y, visibility]
                    'bboxes': [bbox] # format: [x_min, y_min, w, h]
                }
        
        # Convert the dictionary to a list
        image_bboxes = list(image_bboxes_dict.values())
        return image_bboxes
    
    
    # turning off mosaic augmentation
    def close_mosaic(self):
        self.prob['mosaic'] = 0
        
    
    # load image (neither mosaic nor mixup)
    def load_image(self, item):
        image = np.array(Image.open(os.path.join(self.image_dir, item['file_name'])).convert("RGB"))
        
        # convert from [cls, x_min, y_min, w, h] to [cls, x_min, y_min, x_max, y_max]
        bboxes = np.vstack(item['bboxes'])
        bboxes[..., 3:] += bboxes[..., 1:3]
        
        # Albumentations
        aug = Albumentations(self.input_size, self.prob, mosaic=False)
        image, bboxes, keypoints = aug(image, bboxes, np.vstack(item['keypoints']))
        
        bboxes = torch.from_numpy(bboxes)
        keypoints = torch.from_numpy(keypoints).float()

        return image, bboxes, keypoints
    
    
    # load with mosaic (concatenate 4 images to form 1 image)
    def load_mosaic(self, item):
        box4, kpt4 = [], []
        border = [-self.input_size // 2, -self.input_size // 2]
        image4 = np.full((self.input_size * 2, self.input_size * 2, 3), 0, dtype=np.uint8)
        y1a, y2a, x1a, x2a, y1b, y2b, x1b, x2b = None, None, None, None, None, None, None, None

        xc = int(random.uniform(-border[0], 2 * self.input_size + border[1]))
        yc = int(random.uniform(-border[0], 2 * self.input_size + border[1]))

        indices = [item] + random.choices(self.data, k=3)
        random.shuffle(indices)

        for i, index in enumerate(indices):
            # Load image
            image = np.array(Image.open(os.path.join(self.image_dir, index['file_name'])).convert("RGB"))
            
            shape = image.shape
            if i == 0:  # top left
                x1a = max(xc - shape[1], 0)
                y1a = max(yc - shape[0], 0)
                x2a = xc
                y2a = yc
                x1b = shape[1] - (x2a - x1a)
                y1b = shape[0] - (y2a - y1a)
                x2b = shape[1]
                y2b = shape[0]
            elif i == 1:  # top right
                x1a = xc
                y1a = max(yc - shape[0], 0)
                x2a = min(xc + shape[1], self.input_size * 2)
                y2a = yc
                x1b = 0
                y1b = shape[0] - (y2a - y1a)
                x2b = min(shape[1], x2a - x1a)
                y2b = shape[0]
            elif i == 2:  # bottom left
                x1a = max(xc - shape[1], 0)
                y1a = yc
                x2a = xc
                y2a = min(self.input_size * 2, yc + shape[0])
                x1b = shape[1] - (x2a - x1a)
                y1b = 0
                x2b = shape[1]
                y2b = min(y2a - y1a, shape[0])
            elif i == 3:  # bottom right
                x1a = xc
                y1a = yc
                x2a = min(xc + shape[1], self.input_size * 2)
                y2a = min(self.input_size * 2, yc + shape[0])
                x1b = 0
                y1b = 0
                x2b = min(shape[1], x2a - x1a)
                y2b = min(y2a - y1a, shape[0])

            image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            pad_w = x1a - x1b
            pad_h = y1a - y1b
            
            # Labels
            box = np.vstack(index['bboxes'])
            kpt = np.vstack(index['keypoints'])
            if len(box):
                box[..., 1] += pad_w
                box[..., 2] += pad_h
                kpt[..., 0] += pad_w
                kpt[..., 1] += pad_h
            
            box4.append(box)
            kpt4.append(kpt)

        # Concat/clip labels
        box4 = np.concatenate(box4, 0)
        kpt4 = np.concatenate(kpt4, 0)
        
        # convert from [cls, x_min, y_min, w, h] to [cls, x_min, y_min, x_max, y_max]
        box4[..., 3:] += box4[..., 1:3]

        # clip (x_min, y_min) to [0, size - epsilon]
        # clip (x_max, y_max) to [epsilon, size]
        box4[..., 1:3] = np.clip(box4[..., 1:3], 0, self.input_size * 2 - 1e-4)
        box4[..., 3:] = np.clip(box4[..., 3:], 1e-4, self.input_size * 2)

        # Augment
        aug = Albumentations(self.input_size, self.prob, mosaic=True)
        image4, box4, kpt4 = aug(image4, box4, kpt4.reshape(-1, 3))
        
        return image4, torch.tensor(box4), torch.tensor(kpt4)

        
    
    # load with MixUp (merge 2 images)
    def load_mixup(self, item1):
        item2 = random.choices(self.data, k=1)[0]

        image1 = np.array(Image.open(os.path.join(self.image_dir, item1['file_name'])).convert("RGB"))
        image2 = np.array(Image.open(os.path.join(self.image_dir, item2['file_name'])).convert("RGB"))
        
        # Albumentations
        aug = Albumentations(self.input_size, self.prob, mosaic=False)
        bboxes1 = np.vstack(item1['bboxes'])
        bboxes1[..., 3:] += bboxes1[..., 1:3]
        image1, bboxes1, kpt1 = aug(image1, bboxes1, np.vstack(item1['keypoints']))
        
        bboxes2 = np.vstack(item2['bboxes'])
        bboxes2[..., 3:] += bboxes2[..., 1:3]
        image2, bboxes2, kpt2 = aug(image2, bboxes2, np.vstack(item2['keypoints']))
        
        # image mixup
        alpha = np.random.beta(32.0, 32.0)  # mix-up ratio, alpha=beta=32.0
        image = image1 * alpha + image2 * (1 - alpha)
        
        bboxes = torch.from_numpy(np.concatenate((bboxes1, bboxes2), 0))
        keypoints = torch.from_numpy(np.concatenate((kpt1, kpt2), 0))
        
        return image, bboxes, keypoints
    
       
        
class Albumentations:
    def __init__(self, image_size, prob, mosaic=False):  
         
        self.flip_index = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
        
        # use replay compose to later check if hflip is applied
        self.transform = album.ReplayCompose([
            
            # if mosaic, use random sized crop
            album.CenterCrop(
                int(image_size * 1.5), int(image_size * 1.5),
                p = 1 if mosaic else 0
            ),
            album.RandomSizedCrop(
                min_max_height=(int(image_size * 1), int(image_size * 1)),  # crop range
                height=image_size, width=image_size, # final image size
                w2h_ratio=1.0, # square crop
                interpolation=cv2.INTER_LINEAR,
                p = 1 if mosaic else 0
            ),
            
            # not not mosaic, pad to square
            # pad the image to (image_size, image_size) square preserving aspect ratio 
            album.LongestMaxSize(max_size=image_size, p = 0 if mosaic else 1), 
            album.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT, value=0, p = 0 if mosaic else 1),

            album.HorizontalFlip(p=prob['hfilp']),
            album.Blur(p=prob['blur']),
            album.CLAHE(p=prob['clahe']),
            album.ToGray(p=prob['gray']),
            album.MedianBlur(p=prob['medianblur']),
            
            # opencv hsv [0.015*180, 0.7*100, 0.4*100]
            album.HueSaturationValue(
                hue_shift_limit=2.7, sat_shift_limit=70, val_shift_limit=40, p=prob['hsv']
            ),
            
            album.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
            ToTensorV2()],
            bbox_params=album.BboxParams(
                format="pascal_voc", label_fields=[], min_width=2, min_height=2
            ), # PASCAL format = [x_min, y_min, x_max, y_max]
            keypoint_params=album.KeypointParams(format='xy', remove_invisible=False) 
        )

    def __call__(self, image, label_bboxes, keypoints):
        #print('shape', label_bboxes.shape, keypoints.shape)
        #exit()
        # add idx to keep track of which bounding is removed
        bboxes = np.zeros((label_bboxes.shape[0], 6))
        bboxes[:, :4] = label_bboxes[:, 1:]
        bboxes[:, 4] = label_bboxes[:, 0]
        bboxes[:, -1] += np.arange(bboxes.shape[0])
        
        # Apply the transformations
        x = self.transform(image=image, bboxes=bboxes, class_labels=bboxes[:, 0], keypoints=keypoints[..., :2])
        
        # Filter keypoints invisible
        kps, visible = x['keypoints'], keypoints[:, -1]
        
        # Remove keypoints that correspond to removed bounding boxes
        filtered_keypoints, filtered_visible = [], []
        for bbox in x['bboxes']:
            i = int(bbox[-1])
            filtered_keypoints.append(kps[i * 17:(i + 1) * 17])
            filtered_visible.append(visible[i * 17:(i + 1) * 17])

        image = x['image']
        bboxes = np.array(x['bboxes'])
        
        if len(bboxes):
            bboxes = bboxes[:, :-1][:, [4, 0, 1, 2, 3]]
        keypoints = np.array([[*k, v] for kp, vis in zip(filtered_keypoints, filtered_visible) for k, v in zip(kp, vis)])

        # Reshape the keypoints from (num * 17, 3) to (num, 17, 3)
        keypoints = keypoints.reshape(-1, 17, 3)
        
        # applied flip index on HorizontalFlip keypoints
        if self.was_image_flipped(x['replay']):
            keypoints = keypoints[:, self.flip_index, :]

        # turn the keypoint outside bound to invisible
        keypoints = self.filter_keypoints(keypoints, bboxes)

        # convert bbox from xy to centerXY
        if bboxes.size != 0:
            bboxes[:, 1:] = xy2wh(bboxes[:, 1:])
        
        # Handle cases where perspective transformation could make label size = (0, )
        if bboxes.size == 0:
            bboxes = np.zeros((0, 5))
        if keypoints.size == 0:
            keypoints = np.zeros((0, 17, 3))
        
        return image, bboxes, keypoints
    
    
    def was_image_flipped(self, replay):
        for transform in replay['transforms']:
            if transform['__class_fullname__'] == 'HorizontalFlip':
                return transform['applied']
        return False
    
    
    # use bounding box to filter the keypoints that is outside of the bounding box
    def filter_keypoints(self, keypoints, bboxes):
        for i in range(bboxes.shape[0]):
            _, x_min, y_min, x_max, y_max = bboxes[i, :]
            mask = (keypoints[i,:,0]<x_min) | (keypoints[i,:,1]<y_min) | (keypoints[i,:,0]>x_max) | (keypoints[i,:,1]>y_max)
            keypoints[i, mask, 2] = 0
        return keypoints
        