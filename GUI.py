application_name = 'Pose Estimation (YOLO_v8)'

# pyqt packages
from PyQt5 import uic
from PyQt5.QtGui import QPainter, QPixmap, QImage, QColor, QFont, QPen
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog
from PyQt5.QtCore import Qt

import albumentations as album
from albumentations.pytorch import ToTensorV2 
import matplotlib.pyplot as plt
import sys
import numpy as np
import torch
from PIL import Image
import cv2
import pickle

from main import compute_device, inference, get_color
from nets.nn import YOLO



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
        
        self.palette, self.kpt_color, self.limb_color, self.skeleton = get_color()
        with open('categories.pkl', 'rb') as f:
            self.categories = pickle.load(f)
            
        # load the model
        self.load_model_action()
        
        
    # linking all button/textbox with actions    
    def link_commands(self,):
        self.toolButton_import.clicked.connect(self.import_action)
        #self.toolButton_process.clicked.connect(self.process_action)
        
    
    # choosing between models
    def load_model_action(self,):
        # load the model architechture
        self.model = YOLO(size='n', num_classes=1)
        
        # loading the training model weights
        self.model.load_state_dict(torch.load(self.model.name() + '_pose.pth'))
            
        # move model to GPU
        self.model = self.model.to(compute_device())
        
        self.model.eval() # Set model to evaluation mode
        
        
    # clicking the import button action
    def import_action(self,):
        self.label_image.setPixmap(QPixmap())
        
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
        
        # selected the wrong file format
        else:
            show_message(self, title='Load Error', message='Available file format: .jpg')
            self.import_action()
        
        self.process_action()
        
            
    def process_action(self):
        
        augs = self.transform(image=np.array(self.image))
        data = augs["image"] 
        data_display = data.permute(1,2,0).numpy()
        data_display = (data_display*255).astype(np.uint8)
        height, width, channels = data_display.shape
        q_image = QImage(data_display.tobytes(), width, height, width*channels, QImage.Format_RGB888)  # Create QImage
        q_image = q_image.scaled(self.label_image.size(), Qt.KeepAspectRatio)
        
        
        # move data to GPU
        data = data.to(compute_device())
        
        # model inference
        with torch.no_grad():  # Disable gradient calculation
            bboxes, keypoints = inference(self.model, data)
        
        # normalizing from [0, 1] to display size (only works for square image)
        scale = min(self.label_image.height(), self.label_image.width())
        
        keypoints[..., :2] *= scale
        bboxes *= scale
        
        
        # draw bounding boxes
        qp = QPainter(q_image)
        self.draw_bboxes(qp, bboxes)
        self.draw_keypoints(qp, keypoints)  
        self.draw_skeletons(qp, keypoints)
        qp.end()
        
        qpixmap = QPixmap.fromImage(q_image)
        # Display the result in label_detection
        self.label_image.setPixmap(qpixmap)
    
    
    
    # draw the bounding boxes on qp
    def draw_bboxes(self, qp, bboxes):
        pen = QPen(Qt.red, 5)  # Qt.black for color, 5 for line width
        qp.setPen(pen)
        
        # draw the bounding boxes
        for bbox in bboxes:
            # Get the class from the box 
            centerX, centerY, w, h = bbox
            
            x = centerX - w/2
            y = centerY - h/2
            
            # draw bounding box
            qp.drawRect(int(x), int(y), int(w), int(h))
    
    
    # draw the keypoints on qp
    def draw_keypoints(self, qp, keypoints):
        qp.setFont(QFont('Arial', 8))
        
        for i in range(keypoints.shape[0]):
            for j in range(keypoints.shape[1]):
                x, y, v = keypoints[i, j, :]
                # Only plot visible keypoints
                if v > 0:
                    # Set the pen color for the keypoint
                    pen = QPen(QColor(self.kpt_color[j][0], self.kpt_color[j][1], self.kpt_color[j][2]))
                    qp.setPen(pen)
                    
                    # Draw the keypoint as a small circle
                    qp.drawEllipse(int(x) - 3, int(y) - 3, 6, 6)  # Circle with radius 3
    
    
    # Function to draw lines connecting the keypoints based on a skeleton.
    def draw_skeletons(self, qp, keypoints):
        for i in range(keypoints.shape[0]):
            for (start_idx, end_idx), color in zip(self.skeleton, self.limb_color):
                    x1, y1, v1 = keypoints[i, start_idx-1, :]
                    x2, y2, v2 = keypoints[i, end_idx-1, :]
                    # both points are visiable
                    if v1 > 0 and v2 > 0:
                        pen = QPen(QColor(color[0], color[1], color[2]), 2)  # Set pen color and line width
                        qp.setPen(pen)
                        qp.drawLine(int(x1), int(y1), int(x2), int(y2))  # Draw the line connecting keypoints

                
                
    
def main():
    app = QApplication(sys.argv)
    action = QT_Action()
    action.link_commands()
    action.show()
    sys.exit(app.exec_())
    
    
if __name__ == '__main__':
    main()