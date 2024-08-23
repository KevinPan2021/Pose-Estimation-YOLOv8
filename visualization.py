import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches 



# Function to draw lines connecting the keypoints based on a skeleton.
def plot_skeletons(ax, keypoints, skeleton, limb_color):
    for (start_idx, end_idx), color in zip(skeleton, limb_color):
        x1, y1, v1 = keypoints[start_idx-1, :]
        x2, y2, v2 = keypoints[end_idx-1, :]
        # both points are visiable
        if v1 > 0 and v2 > 0:
            ax.plot([x1, x2], [y1, y2], color=color/255.0, linewidth=2)


# Function to draw the keypoints.
def plot_keypoints(ax, keypoints, categories, kpt_color):
    for i in range(keypoints.shape[0]):
        keypoint = keypoints[i, :]
        x, y, v = keypoint
        # Only plot visible keypoints
        if v > 0:
        #if True:
            # Plot scatter point with corresponding color
            ax.scatter(x, y, color=kpt_color[i]/255.0, s=50)
            
            # Add class name to the patch 
            ax.text(
                x, y, 
                s=categories['person'][i], 
                color="white", 
                verticalalignment="top", 
                bbox={"color": kpt_color[i]/255.0, "pad": 0}, 
                clip_on=True
            )

# plot bbox (centerX, centerY, w, h)
def plot_bboxes(ax, bbox):
    # Plot bounding box
    centerX, centerY, w, h = bbox
    
    x = centerX - w/2
    y = centerY - h/2
    
    # Create a Rectangle patch with the bounding box 
    rect = patches.Rectangle(
        (x, y), w, h,
        linewidth=2, 
        edgecolor='r', 
        facecolor="none",
    ) 
    
    # Add the patch to the Axes 
    ax.add_patch(rect) 
    
    
    

# Function to plot images with bounding boxes and class labels 
def plot_image(image, bboxes, keypoints, categories, colors):
    palette, kpt_color, limb_color, COCO_SKELETON = colors
    
    img = np.array(image) 

    # Create figure and axes 
    fig, ax = plt.subplots(1, figsize=(20,20)) 
    
    # Set axis limits to match the image size to crop any overflow
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)  # Invert y-axis to match image coordinates
    
    # Add image to plot 
    ax.imshow(img) 
    
    # rescale to image size
    bboxes *= img.shape[0]
    keypoints *= img.shape[0]
    
    # Plotting the bounding boxes and labels over the image 
    for i in range(bboxes.shape[0]):
        keypoint = keypoints[i, :]
        bbox = bboxes[i, :] 
        
        plot_keypoints(ax, keypoint, categories, kpt_color)
        plot_skeletons(ax, keypoint, COCO_SKELETON, limb_color)
        plot_bboxes(ax, bbox)
    
    plt.axis('off')
    
    # Display the plot 
    plt.show()
    


# Function to plot images with target and pred boxes, class labels 
def plot_images(
        image, target_bboxes, target_keypoints, pred_bboxes, pred_keypoints,
        categories, colors
    ): 
    img = np.array(image) 
    palette, kpt_color, limb_color, COCO_SKELETON = colors
    
    # Create figure and axes 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,20)) 
        
    
    def plot(ax, bboxes, keypoints, title):
        # Set axis limits to match the image size to crop any overflow
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)  # Invert y-axis to match image coordinates
        
        # Add image to plot 
        ax.imshow(img) 
        
        # rescale to image size
        bboxes *= img.shape[0]
        keypoints *= img.shape[0]
        
        # Plotting the bounding boxes and labels over the image 
        for i in range(bboxes.shape[0]):
            
            keypoint = keypoints[i, :]
            bbox = bboxes[i, :] 
            
            plot_keypoints(ax, keypoint, categories, kpt_color)
            plot_skeletons(ax, keypoint, COCO_SKELETON, limb_color)
            plot_bboxes(ax, bbox)
        
        plt.axis('off')
        ax.set_title(title)
    
    plot(ax1, target_bboxes, target_keypoints, 'target')
    plot(ax2, pred_bboxes, pred_keypoints, 'pred')
    
    plt.tight_layout()
    plt.show()# Display the plot 
    
    

# plot the loss and acc curves
def plot_training_curves(train_mAP, train_loss, valid_mAP, valid_loss):
    plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.title('loss curves')
    plt.xlabel('epochs')
    plt.ylabel('cross entropy loss')
    plt.legend()
    
    plt.figure()
    plt.plot(train_mAP, label='train')
    plt.plot(valid_mAP, label='valid')
    plt.title('mAP curves')
    plt.xlabel('epochs')
    plt.ylabel('mAP score')
    plt.legend()