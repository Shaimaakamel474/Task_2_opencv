import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QButtonGroup, QDoubleSpinBox, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5 import uic
from Image_Widget import ImageWidget
from PIL import Image
from PyQt5.QtWidgets import QVBoxLayout
from Shape_Detector import *
from ActiveContour import run_active_contour_demo

# Load the UI file
Ui_MainWindow, QtBaseClass = uic.loadUiType("MainWindow.ui")


# Main Window
class MainWindow(QMainWindow, Ui_MainWindow):
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

        # ACTIVE CONTOUR CONTROLS #
        self.spin_boxes = []
        self.x_input = self.findChild(QDoubleSpinBox, "rg_threshold_spinbox_3")
        self.y_input = self.findChild(QDoubleSpinBox, "rg_threshold_spinbox_4")
        self.radius_input = self.findChild(QDoubleSpinBox, "rg_threshold_spinbox_13")
        self.iterations_input = self.findChild(QDoubleSpinBox, "rg_threshold_spinbox_14")
        self.alpha_input = self.findChild(QDoubleSpinBox, "rg_threshold_spinbox_5")
        self.beta_input = self.findChild(QDoubleSpinBox, "rg_threshold_spinbox_6")
        self.gamma_input = self.findChild(QDoubleSpinBox, "rg_threshold_spinbox_9")
        self.spin_boxes.extend([
            self.x_input, self.y_input, self.radius_input,
            self.iterations_input, self.alpha_input, self.beta_input, self.gamma_input
        ])

        self.RadioButton_ActiveContour.clicked.connect(self.update_active_contour)

        self.area_label = self.findChild(QLabel, "label")
        self.perimeter_label = self.findChild(QLabel, "label_2")

        # self.output_widget = self.findChild(QLabel, "Widget_Output_1")

        for spin_box in self.spin_boxes:
            spin_box.valueChanged.connect(self.update_active_contour)

    def Shape_detector_Modes(self):
        selected_button = self.Mode_group_2.checkedButton()
        threshold_value = self.slider_Threshold.value()
        if selected_button == self.RadioButton_Line:
            lines_detected = Line_Detector(self.org_ImgWidget.image, self.org_ImgWidget.edges_Canny,
                                           threshold_value / 10)
            self.Output_Widget_1.Set_Image(lines_detected)
            self.Output_Widget_1.display_RGBImg()
            print(f"slider value : {threshold_value} New Lines drawed")
        elif selected_button == self.RadioButton_Circle:
            circles_detected = detect_hough_circles(self.org_ImgWidget.image, bin_threshold=threshold_value / 10)
            self.Output_Widget_1.Set_Image(circles_detected)
            self.Output_Widget_1.display_RGBImg()
            print(f"slider value : {threshold_value / 10}New circles drawed")
        elif selected_button == self.RadioButton_Ellipse:
            pass

    def on_new_image_uploaded(self):
        selected_button = self.Mode_group.checkedButton()
        if selected_button == self.RadioButton_ShapeDetector:
            self.Shape_detector_Modes()
        elif selected_button == self.RadioButton_ActiveContour:
            self.update_active_contour()
        elif selected_button == self.RadioButton_CannyEdge:
            pass

    def Remove_checked_Radios(self):
        self.RadioButton_CannyEdge.setChecked(False)
        self.RadioButton_ActiveContour.setChecked(False)
        self.RadioButton_ShapeDetector.setChecked(False)
        self.RadioButton_Line.setChecked(False)
        self.RadioButton_Circle.setChecked(False)
        self.RadioButton_Ellipse.setChecked(False)

    # def update_active_contour(self):
    #     print("UPDATING CONTOUR")
    #     # Check if an image is loaded

    #     if not hasattr(self.org_ImgWidget, 'image_path') or not self.org_ImgWidget.image_path:
    #         print("No image loaded!")
    #         return

    #     # Use the full path directly; avoid splitting unless necessary
    #     img_path = self.org_ImgWidget.image_path
    #     print(f"Image Path: {img_path}")

    #     try:
    #         # Call run_active_contour_demo with visualize=False to avoid plt.show()
    #         results = run_active_contour_demo(
    #             img_path,
    #             initial_contour=None,
    #             alpha=self.alpha_input.value(),
    #             beta=self.beta_input.value(),
    #             gamma=self.gamma_input.value(),
    #             max_iterations=int(self.iterations_input.value()),
    #             visualize=False,  # Prevent plt.show() from blocking the GUI
    #             output_path="contour_result.png"
    #         )

    #         if results:
    #             print(f"Results received: {results.keys()}")
    #             # Load the saved image (contour_result.png) into Output_Widget_1
    #             result_image = Image.open(results['image_path'])
    #             # Convert PIL Image to NumPy array for ImageWidget
    #             result_image_np = np.array(result_image)
    #             self.Output_Widget_1.Set_Image(result_image_np)
    #             self.Output_Widget_1.display_RGBImg()

    #             # Optionally update labels with perimeter and area
    #             if hasattr(self, 'area_label') and self.area_label:
    #                 self.area_label.setText(f"Area: {results['area']:.2f} px²")
    #             else:
    #                 print("Area label not found or not initialized properly")
    #         else:
    #             print("Active contour processing failed: No results returned")

    #     except Exception as e:
    #         print(f"Error in update_active_contour: {e}")

    def update_active_contour(self):
        image_path = self.org_ImgWidget.image_path

        alpha = self.alpha_input.value()
        beta = self.beta_input.value()
        gamma = self.gamma_input.value()
        iterations = self.iterations_input.value()
        try:
            results = run_active_contour_demo(
            image_path,
            initial_contour=None,
            alpha=int(alpha),
            beta=int(beta),
            gamma=int(gamma),
            max_iterations=int(iterations),
            visualize=True
            )
            # self.output_widget.setPixmap(QPixmap("contoured.png"))
            img = cv2.imread("contoured.png")
            self.Output_Widget_1.Set_Image(img)

            self.area_label.setText(str(results['area']))
            self.perimeter_label.setText(str(results['perimeter']))
        except Exception as e:
            print(f"EXCEPTION: {e}")
    


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()

    mainWindow.show()
    sys.exit(app.exec_())
