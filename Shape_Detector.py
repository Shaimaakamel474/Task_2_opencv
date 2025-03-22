
import numpy as np
import cv2

def Line_Detector(image, edge_image, threshold=0.5, theta_res=1):
    '''
    - `image`: Original image.
    - `edge_image`: Grayscale edge-detected image (Canny filter).
    - `threshold`: Percentage of max value in the accumulator to consider peaks.
    - `theta_res`: Angular resolution (default is 1 degree).
    
    Returns:
    - Image with detected lines drawn.
    '''

    height, width = edge_image.shape
    rho_max = int(np.sqrt(height**2 + width**2))  # Maximum possible rho
    thetas = np.deg2rad(np.arange(0, 180, theta_res))  # Angle range
    rhos = np.arange(-rho_max, rho_max + 1, 1)  # Rho values

    
    y_idxs, x_idxs = np.nonzero(edge_image)  # Get all edge points
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint32)
    
    for x, y in zip(x_idxs, y_idxs):  # Vectorized calculation
        rho_values = np.round(x * cos_t + y * sin_t).astype(int) + rho_max 
        accumulator[rho_values, np.arange(len(thetas))] += 1
    
    
    max_acc = np.max(accumulator)
    peaks = np.argwhere(accumulator > max_acc * threshold)
    
    
    output_image = image.copy()
    for rho_idx, theta_idx in peaks:
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = int(a * rho)
        y0 = int(b * rho)
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(output_image, (x1, y1), (x2, y2), (255, 0, 0), 1)

    return output_image



import numpy as np
import cv2
from collections import defaultdict


def detect_hough_circles(image, r_min=20, r_max=None, bin_threshold=0.4, pixel_threshold=5, num_thetas=50):
    """
    Optimized Hough Transform for Circle Detection using NumPy.

    Args:
    - image (numpy.ndarray): Input image.
    - r_min (int): Minimum circle radius.
    - r_max (int): Maximum circle radius.
    - bin_threshold (float): Minimum vote ratio for a valid circle.
    - pixel_threshold (int): Minimum distance between detected circles.
    - num_thetas (int): Number of angle samples.

    Returns:
    - numpy.ndarray: Image with detected circles drawn.
    """
    # Make a copy of the image to draw on
    output_image = image.copy()

    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Get image dimensions
    height, width = edges.shape
    if r_max is None:
        r_max = min(height, width) // 2  # Set the maximum radius based on image size

    # Find edge pixel coordinates
    edge_points = np.argwhere(edges > 0)

    # Initialize the accumulator using NumPy
    accumulator = np.zeros((height, width, r_max - r_min), dtype=np.int32)

    # Precompute theta values
    thetas = np.linspace(0, 2 * np.pi, num_thetas, endpoint=False)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # Voting process
    for r_idx, r in enumerate(range(r_min, r_max)):
        a_candidates = (edge_points[:, 0][:, None] - r * cos_t).astype(np.int32)
        b_candidates = (edge_points[:, 1][:, None] - r * sin_t).astype(np.int32)

        # Filter valid points within image boundaries
        valid_mask = (0 <= a_candidates) & (a_candidates < height) & (0 <= b_candidates) & (b_candidates < width)
        valid_a = a_candidates[valid_mask]
        valid_b = b_candidates[valid_mask]

        # Accumulate votes efficiently
        np.add.at(accumulator, (valid_a, valid_b, r_idx), 1)

    # Extract circles from the accumulator
    detected_circles = []
    votes_threshold = bin_threshold * num_thetas

    for r_idx, r in enumerate(range(r_min, r_max)):
        y_idxs, x_idxs = np.where(accumulator[:, :, r_idx] >= votes_threshold)

        for x, y in zip(x_idxs, y_idxs):
            if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(r - rc) > pixel_threshold 
                   for xc, yc, rc in detected_circles):
                detected_circles.append((x, y, r))

    # Draw detected circles on the copy of the image
    for x, y, r in detected_circles:
        cv2.circle(output_image, (y, x), r, (0, 255, 0), 2)

    return output_image  





