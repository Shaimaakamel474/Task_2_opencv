import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget , QRadioButton, QButtonGroup
from PyQt5 import uic 
from Imag_Widget import ImageWidget
from PyQt5.QtWidgets import QWidget, QFileDialog
from PIL import Image, ImageQt, ImageEnhance
import numpy as np
from PyQt5.QtWidgets import QVBoxLayout
from Shape_Detector import *
# Load the UI file
Ui_MainWindow, QtBaseClass = uic.loadUiType("MainWindow.ui")

# Main Window
class MainWindow(QMainWindow , Ui_MainWindow ):
    def __init__(self):

        super(MainWindow, self).__init__()
        self.setupUi(self)

        
       # make groups for  radio buttons 
        self.Mode_group = QButtonGroup(self)
        self.Mode_group.addButton(self.RadioButton_CannyEdge)
        self.Mode_group.addButton(self.RadioButton_ActiveContour)
        self.Mode_group.addButton(self.RadioButton_ShapeDetector)


        # make groups for  radio buttons 
        self.Mode_group_2 = QButtonGroup(self)
        self.Mode_group_2.addButton(self.RadioButton_Line)
        self.Mode_group_2.addButton(self.RadioButton_Circle)
        self.Mode_group_2.addButton(self.RadioButton_Ellipse)

        self.Remove_checked_Radios()


        layout = QVBoxLayout(self.Widget_Org_Image)
        self.org_ImgWidget = ImageWidget(None, self.Widget_Org_Image)
        layout.addWidget(self.org_ImgWidget)
        self.Widget_Org_Image.setLayout(layout)

        self.org_ImgWidget.image_uploaded.connect(self.on_new_image_uploaded)
       
       
        layout_2 = QVBoxLayout(self.Widget_Output_1)
        self.Output_Widget_1 = ImageWidget(None, self.Widget_Output_1)
        layout_2.addWidget(self.Output_Widget_1)
        self.Widget_Output_1.setLayout(layout_2)
      
        # self.Output_Widget_1.image_uploaded.connect(self.on_new_image_uploaded)

    
        self.RadioButton_ShapeDetector.clicked.connect(self.Shape_detector_Modes)
        self.RadioButton_Line.clicked.connect(self.Shape_detector_Modes)
        self.RadioButton_Circle.clicked.connect(self.Shape_detector_Modes)
        self.RadioButton_Ellipse.clicked.connect(self.Shape_detector_Modes)
        self.slider_Threshold.sliderReleased.connect(self.Shape_detector_Modes)



    def Shape_detector_Modes(self):
        selected_button = self.Mode_group_2.checkedButton()
        threshold_value=self.slider_Threshold.value()
        if selected_button == self.RadioButton_Line:
            lines_detected=Line_Detector(self.org_ImgWidget.image, self.org_ImgWidget.edges_Canny ,threshold_value/10 )
            self.Output_Widget_1.Set_Image(lines_detected)
            self.Output_Widget_1.display_RGBImg()
            print(f"slider value : {threshold_value} New Lines drawed")
        elif selected_button == self.RadioButton_Circle:
            circles_detected=detect_hough_circles(self.org_ImgWidget.image ,bin_threshold=threshold_value/10 )
            self.Output_Widget_1.Set_Image(circles_detected)
            self.Output_Widget_1.display_RGBImg()
            print(f"slider value : {threshold_value/10}New circles drawed")
        elif selected_button == self.RadioButton_Ellipse:
            pass
            
    def on_new_image_uploaded(self):
        selected_button = self.Mode_group.checkedButton()
        if selected_button == self.RadioButton_ShapeDetector:
            self.Shape_detector_Modes()
        elif selected_button == self.RadioButton_ActiveContour:
            pass
        elif selected_button == self.RadioButton_CannyEdge:
            pass



    def Remove_checked_Radios(self):
        self.RadioButton_CannyEdge.setChecked(False) 
        self.RadioButton_ActiveContour.setChecked(False) 
        self.RadioButton_ShapeDetector.setChecked(False) 
        self.RadioButton_Line.setChecked(False) 
        self.RadioButton_Circle.setChecked(False) 
        self.RadioButton_Ellipse.setChecked(False) 


    

if __name__=="__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    

    mainWindow.show()
    sys.exit(app.exec_())
