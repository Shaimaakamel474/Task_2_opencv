import numpy as np
import cv2
from collections import defaultdict

class HoughEllipse:
    @staticmethod
    def detect_and_draw_hough_ellipses(image, a_min=30, a_max=100, b_min=30, b_max=100, delta_a=2, delta_b=2, num_thetas=100, bin_threshold=0.4, min_edge_threshold=100, max_edge_threshold=150):
        """
        Detect ellipses using Hough Transform.
        Args:
            image (np.array): Input image.
            a_min (int): Minimum semi-major axis length of ellipses to detect.
            a_max (int): Maximum semi-major axis length of ellipses to detect.
            b_min (int): Minimum semi-minor axis length of ellipses to detect.
            b_max (int): Maximum semi-minor axis length of ellipses to detect.
            delta_a (int): Step size for semi-major axis length.
            delta_b (int): Step size for semi-minor axis length.
            num_thetas (int): Number of steps for theta from 0 to 2PI.
            bin_threshold (float): Thresholding value in percentage to shortlist candidate ellipses.
            min_edge_threshold (int): Minimum threshold value for edge detection.
            max_edge_threshold (int): Maximum threshold value for edge detection.
        Returns:
            tuple: A tuple containing the output image with detected ellipses drawn and a list of detected ellipses.
        """

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge detection using Canny
        edge_image = cv2.Canny(gray_image, min_edge_threshold, max_edge_threshold)

        # Get image dimensions
        img_height, img_width = edge_image.shape[:2]

        # Parameters for ellipse detection
        thetas = np.linspace(0, 2 * np.pi, num_thetas)
        as_ = np.arange(a_min, a_max, delta_a)
        bs = np.arange(b_min, b_max, delta_b)
        cos_thetas = np.cos(thetas)
        sin_thetas = np.sin(thetas)

        # List of ellipse candidates with parametric (a, b, cos(theta), sin(theta))
        ellipse_candidates = [(a, b, np.array([a * cos_t for cos_t in cos_thetas]), np.array([b * sin_t for sin_t in sin_thetas]))
                              for a in as_ for b in bs]

        # Initialize accumulator for voting
        accumulator = defaultdict(int)

        # Iterate through each edge pixel and perform Hough Voting
        edge_points = np.column_stack(np.where(edge_image > 0))

        for (x, y) in edge_points:
            for a, b, cos_thetas, sin_thetas in ellipse_candidates:
                # Compute ellipse centers for each candidate ellipse
                x_center = x - cos_thetas
                y_center = y - sin_thetas

                for xc, yc in zip(x_center, y_center):
                    accumulator[(xc, yc, a, b)] += 1

        # Post-process accumulator to shortlist ellipses
        total_votes = len(edge_points)
        out_ellipses = [(x, y, a, b, votes / total_votes)
                        for (x, y, a, b), votes in accumulator.items() if votes / total_votes >= bin_threshold]

        # Post-processing to remove duplicate ellipses based on proximity
        postprocess_ellipses = []
        pixel_threshold = 5
        for x, y, a, b, v in out_ellipses:
            if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(a - ac) > pixel_threshold or abs(b - bc) > pixel_threshold for xc, yc, ac, bc, v in postprocess_ellipses):
                postprocess_ellipses.append((x, y, a, b, v))
        
        # Draw the detected ellipses on the image
        output_img = image.copy()
        for x, y, a, b, _ in postprocess_ellipses:
            output_img = cv2.ellipse(output_img, (int(x), int(y)), (int(a), int(b)), 0, 0, 360, (0, 255, 0), 2)

        return output_img, postprocess_ellipses

# Example usage:
image = cv2.imread(r"C:\Users\shaim_qkqsx\Downloads\Computer_Vision-Toolbox-main\Computer_Vision-Toolbox-main\Image Processing Edge Detector & Active Contour\dataset\Ellipse-detection-algorithm-over-natural-images-the-image-in-a-shows-the-original-image.png")  # Provide the correct path to your image

# Detect and draw ellipses
output_image, detected_ellipses = HoughEllipse.detect_and_draw_hough_ellipses(image)

# Display the result
cv2.imshow('Detected Ellipses', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the result image
cv2.imwrite('result_with_ellipses.jpg', output_image)
