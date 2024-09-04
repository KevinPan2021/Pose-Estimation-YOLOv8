application_name = 'Pose Estimation (YOLO_v8)'

# pyqt packages
from PyQt5 import uic
from PyQt5.QtGui import QPainter, QPixmap, QImage, QColor, QFont, QPen
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

import albumentations as album
from albumentations.pytorch import ToTensorV2 
import sys
import numpy as np
import torch
from PIL import Image
import cv2
import pickle
import time

from main import compute_device, inference, get_color
from nets.nn import YOLO



def show_message(parent, title, message, icon=QMessageBox.Warning):
        msg_box = QMessageBox(icon=icon, text=message)
        msg_box.setWindowIcon(parent.windowIcon())
        msg_box.setWindowTitle(title)
        msg_box.setStyleSheet(parent.styleSheet() + 'color:white} QPushButton{min-width: 80px; min-height: 20px; color:white; \
                              background-color: rgb(91, 99, 120); border: 2px solid black; border-radius: 6px;}')
        msg_box.exec()
        
        
        
class ImageProcessingThread(QThread):
    finished = pyqtSignal()  # Signal emitted when processing is finished
    parent_class = None
    
    def __init__(self, image, parent_class=None):
        super().__init__(parent_class)
        self.image = image
        self.parent_class = parent_class
        self.last_time = time.time()
        
    def run(self):
        # Your image processing logic here
        self.process_image_action()


    def process_image_action(self):
        image = np.array(self.parent_class.image)
        h, w, _ = image.shape
        
        # Apply transformations to the image
        data = self.parent_class.transform(image=image)["image"]
        
        # Convert tensor to a displayable format
        data_display = data.permute(1, 2, 0).numpy()
        data_display = (data_display * 255).astype(np.uint8)
        
        # Calculate padding and crop the image
        if w < h:  # Vertical padding (image originally taller)
            pad = int((1 - w / h) * self.parent_class.image_size / 2)
            data_display = data_display[:, pad:-pad, :]
        elif w > h:  # Horizontal padding (image originally wider)
            pad = int((1 - h / w) * self.parent_class.image_size / 2)
            data_display = data_display[pad:-pad, :, :]

        height, width, channels = data_display.shape
        q_image = QImage(data_display.tobytes(), width, height, width * channels, QImage.Format_RGB888)
        q_image = q_image.scaled(self.parent_class.label_image.size(), Qt.KeepAspectRatio)
        
        # Model inference
        with torch.no_grad():
            bboxes, keypoints = inference(self.parent_class.model, data)
            #bboxes, keypoints = np.zeros((0, 5)), np.zeros((0, 17, 3))
        
        # Subtract the padding for keypoints and bboxes
        if w < h:
            pad = (1 - w / h) / 2
            keypoints[..., 0] -= pad  # unpad keypoint x
            bboxes[..., 0] -= pad  # unpad bbox centerX

        elif w > h:
            pad = (1 - h / w) / 2
            scale = min(w / h, self.parent_class.label_image.width() / self.parent_class.label_image.height())

            keypoints[..., 1] -= pad  # unpad keypoint y
            bboxes[..., 1] -= pad  # unpad bbox centerY

            keypoints[..., :2] *= scale  # scale keypoints
            bboxes *= scale  # scale bboxes

        # Normalize bounding boxes and keypoints from [0, 1] to the display size
        scale = min(self.parent_class.label_image.height(), self.parent_class.label_image.width())
        keypoints[..., :2] *= scale
        bboxes *= scale
        
        # Calculate and draw FPS
        end_time = time.time()
        fps = 1.0 / (end_time - self.last_time)
        self.last_time = end_time
        
        
        # Draw bounding boxes, keypoints, and skeletons
        qp = QPainter(q_image)
        self.draw_bboxes(qp, bboxes)
        self.draw_keypoints(qp, keypoints)
        self.draw_skeletons(qp, keypoints)
        self.draw_fps(qp, fps)
        qp.end()

        qpixmap = QPixmap.fromImage(q_image)
        # Display the result in label_detection
        self.parent_class.label_image.setPixmap(qpixmap)

        self.finished.emit()  # Emit the finished signal when processing is done


     # Function to draw FPS on the screen
    def draw_fps(self, qp, fps):
        qp.setFont(QFont('Arial', 12))
        pen = QPen(Qt.green)
        qp.setPen(pen)
        qp.drawText(10, 30, f"FPS: {fps:.2f}")
        
        
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
                    pen = QPen(QColor(self.parent_class.kpt_color[j][0], self.parent_class.kpt_color[j][1], self.parent_class.kpt_color[j][2]))
                    qp.setPen(pen)
                    
                    # Draw the keypoint as a small circle
                    qp.drawEllipse(int(x) - 3, int(y) - 3, 6, 6)  # Circle with radius 3
    
    
    # Function to draw lines connecting the keypoints based on a skeleton.
    def draw_skeletons(self, qp, keypoints):
        for i in range(keypoints.shape[0]):
            for (start_idx, end_idx), color in zip(self.parent_class.skeleton, self.parent_class.limb_color):
                    x1, y1, v1 = keypoints[i, start_idx-1, :]
                    x2, y2, v2 = keypoints[i, end_idx-1, :]
                    # both points are visiable
                    if v1 > 0 and v2 > 0:
                        pen = QPen(QColor(color[0], color[1], color[2]), 2)  # Set pen color and line width
                        qp.setPen(pen)
                        qp.drawLine(int(x1), int(y1), int(x2), int(y2))  # Draw the line connecting keypoints




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
        
        # process thread
        self.processing_thread = ImageProcessingThread(self.image, self)
        
        
        
    # linking all button/textbox with actions    
    def link_commands(self,):
        self.toolButton_import.clicked.connect(self.import_action)
        self.toolButton_camera.clicked.connect(self.camera_action)
        
    
    # choosing between models
    def load_model_action(self,):
        # load the model architechture
        self.model = YOLO(size='n', num_classes=1)
        
        # loading the training model weights
        self.model.load_state_dict(torch.load(self.model.name() + '_pose.pth'))
        #self.model.load_state_dict(torch.load("v8_m_pose.pth"))
            
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
        
        self.processing_thread.start()
        
    
    # open the camera and stream videos
    def camera_action(self):
        if self.toolButton_camera.isChecked():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                return
    
            # start the camera feed
            self.processing_thread.finished.connect(self.start_camera_action)
            self.start_camera_action()
            
        # stop the stream
        else:
            self.cap.release()  # Release the camera
            self.processing_thread.finished.disconnect(self.start_camera_action)  # Disconnect the signal
            self.label_image.clear()  # Clear the label displaying the image


    @pyqtSlot()
    def start_camera_action(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        self.image = frame[:, ::-1, [2, 1, 0]]  # Convert from BGR to RGB
        # Create and start the processing thread
        self.processing_thread.start()
            
                
    
def main():
    app = QApplication(sys.argv)
    action = QT_Action()
    action.link_commands()
    action.show()
    sys.exit(app.exec_())
    
    
if __name__ == '__main__':
    main()