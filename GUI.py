application_name = 'Object Detection (YOLO_v8)'

# pyqt packages
from PyQt5 import uic
from PyQt5.QtGui import QPainter, QPixmap, QImage, QColor, QFont
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog

import albumentations as album
from albumentations.pytorch import ToTensorV2 
import matplotlib.pyplot as plt
import sys
import numpy as np
import torch
from PIL import Image
import cv2
import pickle

from nets.nn import YOLO
from main import BidirectionalMap, compute_device, inference



def show_message(parent, title, message, icon=QMessageBox.Warning):
        msg_box = QMessageBox(icon=icon, text=message)
        msg_box.setWindowIcon(parent.windowIcon())
        msg_box.setWindowTitle(title)
        msg_box.setStyleSheet(parent.styleSheet() + 'color:white} QPushButton{min-width: 80px; min-height: 20px; color:white; \
                              background-color: rgb(91, 99, 120); border: 2px solid black; border-radius: 6px;}')
        msg_box.exec()
        
        
        
class QT_Action(QMainWindow):
    def __init__(self):
        # system variable
        super(QT_Action, self).__init__()
        uic.loadUi('qt_main.ui', self)
        
        self.setWindowTitle(application_name) # set the title
        
        # runtime variable
        self.image_size = 640
        with open('class_ind_pair.pkl', 'rb') as f:
            self.class_labels = pickle.load(f)
        self.image = None
        self.model = None
        
        self.transform = album.Compose([
            # Rescale an image so that maximum side is equal to image_size 
            album.LongestMaxSize(max_size=self.image_size), 
            # Pad remaining areas with zeros 
            album.PadIfNeeded( 
                min_height=self.image_size, min_width=self.image_size, border_mode=cv2.BORDER_CONSTANT 
            ),
            # Normalize the image 
            album.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255), 
            # Convert the image to PyTorch tensor 
            ToTensorV2() 
        ])
        
        # load the model
        self.load_model_action()
        
        
    # linking all button/textbox with actions    
    def link_commands(self,):
        self.toolButton_import.clicked.connect(self.import_action)
        self.toolButton_process.clicked.connect(self.process_action)
        
    
    # choosing between models
    def load_model_action(self,):
        # load the model architechture
        self.model = YOLO(size='m', num_classes=len(self.class_labels))
        
        # loading the best model weights
        #self.model.load_state_dict(torch.load(self.model.name() + '.pth'))
        
        # loading the official weights
        self.model.load_state_dict(torch.load(f'v8_{self.model.size}.pth')) 
        
        # move model to GPU
        self.model = self.model.to(compute_device())
        
        self.model.eval() # Set model to evaluation mode
        
        
    # clicking the import button action
    def import_action(self,):
        self.label_image.setPixmap(QPixmap())
        self.label_detection.setPixmap(QPixmap())
        
        # show an "Open" dialog box and return the path to the selected file
        filename, _ = QFileDialog.getOpenFileName(None, "Select file", options=QFileDialog.Options())
        self.lineEdit_import.setText(filename)
        
        # didn't select any files
        if filename is None or filename == '': 
            return
    
        # selected .jpg images
        if filename.endswith('.jpg'):
            self.image = Image.open(filename) 
            self.lineEdit_import.setText(filename)
            #X = [transform(img)]
            self.update_display()
        
        # selected the wrong file format
        else:
            show_message(self, title='Load Error', message='Available file format: .jpg')
            self.import_action()
        
        
    def update_display(self):
        if not self.image is None:
            augs = self.transform(image=np.array(self.image))
            data = augs["image"] 
            data = data.permute(1,2,0).numpy()
            data = (data*255).astype(np.uint8)
            height, width, channels = data.shape
            q_image = QImage(data.tobytes(), width, height, width*channels, QImage.Format_RGB888)  # Create QImage
            qpixmap = QPixmap.fromImage(q_image)  # Convert QImage to QPixmap
            self.label_image.setPixmap(qpixmap)
            
            
    def process_action(self):
        if self.image is None:
            show_message(self, title='Process Error', message='Please load an image first')
            return
        
        augs = self.transform(image=np.array(self.image))
        data = augs["image"] 
        
        # move data to GPU
        data = data.to(compute_device())
        
        # model inference
        with torch.no_grad():  # Disable gradient calculation
            boxes = inference(self.model, data)
        
        # convert x to gray image
        img = data.permute(1,2,0).detach().cpu().numpy()
        gray_img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]) # convert to gray image
        gray_img = gray_img*255
        height, width = gray_img.shape
        q_image = QImage(gray_img.astype(np.uint8).tobytes(), width, height, width, QImage.Format_Grayscale8)  # Create QImage
        qpixmap = QPixmap.fromImage(q_image)  # Convert QImage to QPixmap
        
        # draw bounding boxes
        qp = QPainter(qpixmap)
        
        # Getting the color map from matplotlib 
        colour_map = plt.get_cmap("tab20b") 
        # Getting 20 different colors from the color map for 20 different classes 
        colors = [colour_map(i) for i in np.linspace(0, 1, len(self.class_labels))] 
        # Getting the height and width of the image 
        h, w, _ = img.shape 
        
        for box in boxes:
            # Get the class from the box 
            score, class_pred, a, b, c, d = box
            red, green, blue, _ = colors[int(class_pred)]
            qp.setPen(QColor.fromRgbF(red, green, blue))
            
            # draw bounding box
            qp.drawRect(int(a*w), int(b*h), int(c*w), int(d*h))
            
            # Write class label below the upper left corner of the bounding box
            class_label = self.class_labels.get_value(int(class_pred))
            qp.setFont(QFont("Arial", 10))  # Set font size and type
            qp.drawText(int(a*w)+5, int(b*h)+10, class_label)
    
        qp.end()
        
        # Display the result in label_detection
        self.label_detection.setPixmap(qpixmap)
    
    
def main():
    app = QApplication(sys.argv)
    action = QT_Action()
    action.link_commands()
    action.show()
    sys.exit(app.exec_())
    
    
if __name__ == '__main__':
    main()