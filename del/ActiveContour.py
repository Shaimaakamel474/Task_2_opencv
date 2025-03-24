import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.feature import canny
from contour import ActiveContourModel  # Import your existing class

class ContourApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Active Contour Model - PyQt5")
        self.setGeometry(100, 100, 900, 600)

        # Main widget and layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Button to load an image
        self.btn_load = QPushButton("Load Image", self)
        self.btn_load.clicked.connect(self.load_image)
        self.layout.addWidget(self.btn_load)

        # Button to apply active contour
        self.btn_process = QPushButton("Run Active Contour", self)
        self.btn_process.setEnabled(False)
        self.btn_process.clicked.connect(self.run_active_contour)
        self.layout.addWidget(self.btn_process)

        # Matplotlib figure for visualization
        self.figure, self.ax = plt.subplots(figsize=(7, 5))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.image_path = None
        self.original_image = None

    def load_image(self):
        """Load an image and display it"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.btn_process.setEnabled(True)  # Enable the process button
            self.show_image()  # Display image in Matplotlib

    def show_image(self):
        """Display the loaded image in Matplotlib"""
        self.ax.clear()
        self.ax.imshow(self.original_image, cmap='gray')
        self.ax.set_title("Loaded Image")
        self.canvas.draw()

    def run_active_contour(self):
        """Run the active contour model and visualize evolution"""
        try:
            if self.image_path is None:
                return

            # Preprocess the image
            img = np.array(Image.open(self.image_path).convert('L'))
            img = gaussian_filter(img, sigma=2)
            edges = canny(img, sigma=1.5, low_threshold=0.1, high_threshold=0.2)
            img = edges.astype(float)

            # Create initial contour (ellipse)
            height, width = img.shape
            center_x, center_y = width // 2, height // 2
            a, b = width // 3, height // 3
            theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
            x = center_x + a * np.cos(theta)
            y = center_y + b * np.sin(theta)
            initial_contour = np.column_stack([x, y]).astype(int)

            # Run Active Contour Model
            snake = ActiveContourModel(img, initial_contour, alpha=0.1, beta=0.2, gamma=0.9, w_line=0.5, w_edge=3.0, max_iterations=200)
            contours = snake.evolve(return_intermediate=True)  # Get intermediate contours

            # Plot the contour evolution
            self.ax.clear()
            self.ax.imshow(self.original_image, cmap='gray')
            self.ax.set_title("Active Contour Model Evolution")

            # Colors for iterations
            colors = ['red', 'green', 'blue', 'purple', 'magenta']
            iteration_labels = [0, 50, 100, 150, 200]

            for i, contour in enumerate(contours):
                contour = np.round(contour).astype(int)
                color = colors[i % len(colors)]
                self.ax.plot(contour[:, 0], contour[:, 1], color=color, marker='o', label=f"Iteration {iteration_labels[i]}")

            # Final contour in black
            final_contour = np.round(contours[-1]).astype(int)
            self.ax.plot(final_contour[:, 0], final_contour[:, 1], 'k-', marker='o', label="Final Contour")

            self.ax.legend()
            self.canvas.draw()
        except Exception as e:
            print(f"Exception: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ContourApp()
    window.show()
    sys.exit(app.exec_())
