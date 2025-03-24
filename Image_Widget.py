from PyQt5.QtGui import QPainter, QPixmap, QImage, QPainterPath  # Import QPainterPath here
from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtWidgets import QWidget, QFileDialog
from PIL import Image, ImageQt, ImageEnhance
from numpy.fft import ifft2, ifftshift
import numpy as np
from scipy.fft import fft2, fftshift

import cv2
from PyQt5.QtCore import pyqtSignal
class ImageWidget(QWidget):
    image_uploaded = pyqtSignal(bool)
    def __init__(self, image_path=None, parent=None):
        super(ImageWidget, self).__init__(parent)
        self.pixmap = None  
        self.image=None
        self.gray_img=None
        self.edges_Canny=None

        self.setMouseTracking(True)

        if image_path:
            self.image_path = image_path
            print(f"Image Path: {self.image_path}")
            self.load_image(image_path)
    
    # automatically run to paint the img
    def paintEvent(self, event):
        if self.pixmap:
            painter = QPainter(self)
            # make img border raduis 
            rect = QRectF(self.rect())
            path = QPainterPath() 
            path.addRoundedRect(rect, 20, 20) 
            painter.setClipPath(path)
           
            # resize the img 
            scaled_pixmap = self.pixmap.scaled(
                # set its geomerty from the main
                self.size(),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation
            )
            
            painter.drawPixmap(self.rect(), scaled_pixmap)

    def load_image(self, image_path):  
            # read image for processing and convert to grayscale
            self.image_path = image_path
            self.image = cv2.imread(image_path)
            self.image = cv2.cvtColor(self.image, code=cv2.COLOR_BGR2RGB)
            # sure the resized_img is the size of widget 
            self.image = cv2.resize(self.image, (self.width(), self.height()), interpolation=cv2.INTER_AREA)
            
            self.gray_img=cv2.cvtColor(self.image , cv2.COLOR_RGB2GRAY)
            # filtered=cv2.GaussianBlur(self.gray_img, (7, 7), 2)
            self.edges_Canny=cv2.Canny(self.gray_img, 50, 128)
            self.pixmap=self.convert_np_pixmap(self.image)

            self.center_x, self.center_y = self.image.shape[1] // 2, self.image.shape[0] // 2
            self.radius = min(self.image.shape) // 2
            
            # repaint
            self.update() 

    def get_curr_GrayImg(self):
        return self.gray_img

    def get_curr_RGBImg(self):
        return self.image
    

    def Set_Image(self , image):
        self.image=image
        # sure the resized_img is the size of widget 
        self.image = cv2.resize(self.image, (self.width(), self.height()), interpolation=cv2.INTER_AREA)
        if (image.shape ==3 ) :
            self.image= cv2.cvtColor(self.image , code=cv2.COLOR_BGR2RGB)
            self.gray_img=cv2.cvtColor(self.image , cv2.COLOR_RGB2GRAY)
        else : self.gray_img = image

        
        self.pixmap=self.convert_np_pixmap(self.gray_img)

        # repaint
        self.update()

    def display_RGBImg(self):
        self.pixmap=self.convert_np_pixmap(self.image)
        self.update()
    
    def display_GrayImg(self):
        self.pixmap=self.convert_np_pixmap(self.gray_img)
        self.update()



    def mouseDoubleClickEvent(self, event):
        """This method is automatically called on double-click."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.bmp *.jpeg *.gif);;All Files (*)")
        if file_path:
            self.load_image(file_path) 
            self.image_uploaded.emit(True)
    



    def convert_np_pixmap(self, np_arr):
        """
        Converts a NumPy array (H, W, 3) representing an RGB image to QPixmap.
        """

        
        # Convert the NumPy array to bytes
        byte_data = np_arr.tobytes()
        
        
        # Ensure input is an RGB image
        if len(np_arr.shape) == 3:
            height, width, channels = np_arr.shape
            # Create QImage from the byte data
            qimage = QImage(byte_data, width, height, width * channels, QImage.Format_RGB888)

        elif len(np_arr.shape) == 2:
            height, width = np_arr.shape
            # Create QImage from the byte data (using width, height, bytesPerLine, and format)
            qimage = QImage(byte_data, width, height, width, QImage.Format_Grayscale8)

        
        # Convert QImage to QPixmap for displaying in PyQt
        pixmap = QPixmap.fromImage(qimage)
        return pixmap
    

    def convert_pixmap_np(self, pixmap): 
        """
        Converts a QPixmap to a NumPy array (H, W, 3) representing an RGB image.
        """
        # Convert QPixmap to QImage
        image = pixmap.toImage()
        if (image.shape ==3):
            # Ensure the QImage is in the correct RGB format
            image = image.convertToFormat(QImage.Format_RGB888)
            # Access raw pixel data from QImage
            width = image.width()
            height = image.height()
            # Extract pixel data from QImage as a NumPy array
            ptr = image.bits()
            ptr.setsize(image.byteCount())

            # Convert the byte data into a NumPy array with the appropriate shape (height, width, 3) for RGB
            np_array = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 3))
        else :
                    # Ensure the QImage is in the correct format for manipulation
            image = image.convertToFormat(QImage.Format_Grayscale8)  # Convert to grayscale format
            
            # Access raw pixel data from the QImage
            width = image.width()
            height = image.height()
            
            # Create a NumPy array from the QImage pixel data
            ptr = image.bits()
            ptr.setsize(image.byteCount())
            
            # Convert the byte data into a NumPy array with the appropriate shape (height, width) for grayscale
            np_array = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width))

        return np_array