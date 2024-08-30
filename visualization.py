import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches 



# Function to plot images with bounding boxes and class labels 
def plot_image(image, boxes, class_labels): 
    # Getting the color map from matplotlib 
    colour_map = plt.get_cmap("tab20b") 
    # Getting 20 different colors from the color map for 20 different classes 
    colors = [colour_map(i) for i in np.linspace(0, 1, len(class_labels))] 
  
    img = np.array(image) 

    # Create figure and axes 
    fig, ax = plt.subplots(1, figsize=(10,10)) 
  
    # Add image to plot 
    ax.imshow(img) 
  
    # Plotting the bounding boxes and labels over the image 
    for i in range(boxes.shape[0]):
        box = boxes[i,:]
        
        # Get the class from the box 
        class_pred = box[1] 
        
        # Get the center x and y coordinates and rescale to img shape
        x, y, w, h = box[2:] * img.shape[0]

        # Create a Rectangle patch with the bounding box 
        rect = patches.Rectangle( 
            (x, y), w, h,
            linewidth=2, 
            edgecolor=colors[int(class_pred)], 
            facecolor="none", 
        ) 
          
        # Add the patch to the Axes 
        ax.add_patch(rect) 
          
        # Add class name to the patch 
        plt.text( 
            x, y, 
            s = class_labels.get_value(int(class_pred)), 
            color="white", 
            verticalalignment="top", 
            bbox={"color": colors[int(class_pred)], "pad": 0}, 
        ) 
    
    plt.axis('off')
    
    # Display the plot 
    plt.show()
    


# Function to plot images with target and pred boxes, class labels 
def plot_images(image, target_boxes, pred_boxes, class_labels): 
    # Getting the color map from matplotlib 
    colour_map = plt.get_cmap("tab20b") 
    # Getting 20 different colors from the color map for 20 different classes 
    colors = [colour_map(i) for i in np.linspace(0, 1, len(class_labels))] 
  
    img = np.array(image) 

    # Create figure and axes 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,10)) 
        
    def plot(ax, boxes, title):
        # plot target
        ax.imshow(img) 
      
        # Plotting the bounding boxes and labels over the image 
        for i in range(boxes.shape[0]):
            box = boxes[i,:]
            
            # Get the class from the box 
            class_pred = box[1] 
            
            # Get the center x and y coordinates and rescale to img shape
            x, y, w, h = box[2:] * img.shape[0]
    
            # Create a Rectangle patch with the bounding box 
            rect = patches.Rectangle( 
                (x, y), w, h,
                linewidth=2, 
                edgecolor=colors[int(class_pred)], 
                facecolor="none", 
            ) 
              
            # Add the patch to the Axes 
            ax.add_patch(rect) 
              
            # Add class name to the patch 
            ax.text( 
                x, y, 
                s = class_labels.get_value(int(class_pred)), 
                color="white", 
                verticalalignment="top", 
                bbox={"color": colors[int(class_pred)], "pad": 0}, 
            ) 
        ax.axis('off')
        ax.set_title(title)
    
    plot(ax1, target_boxes, 'target')
    plot(ax2, pred_boxes, 'pred')
    
    # Display the plot 
    plt.show()
    
    

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