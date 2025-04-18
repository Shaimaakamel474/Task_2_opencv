from collections import defaultdict
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
    # modified_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    return output_image







def detect_and_draw_hough_circles(image, threshold=8.1, region=15, radius=[70, 2]):
        """
        Detect circles in the input image using Hough Transform and draw them.

        Args:
        - image (numpy.ndarray): Input image.
        - threshold (float): Threshold for circle detection.
        - region (int): Region size for local maximum search.
        - radius (list): Range of radii to search for circles.

        Returns:
        - numpy.ndarray: Processed image with detected circles drawn.

        The function first applies Gaussian smoothing and Canny edge detection to the input image.
        It then constructs an accumulator array to detect circles of different radii.
        The accumulator array is initialized with zeros, and each edge point contributes to the
        accumulator by drawing circles of different radii centered at that point.
        Local maxima in the accumulator array indicate potential circle centers, and the function
        extracts these centers and draws circles on the input image.

        Note:
        - The input image should be in BGR format.
        - The returned image is in RGB format with detected circles drawn on it.
        """

        # Make a copy of the input image to prevent modifications to the original
        modified_image = np.copy(image)

        # Convert image to grayscale if it's in color
        if len(modified_image.shape) > 2:
            grayscale_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2GRAY)
        else:
            grayscale_image = modified_image

        # Apply Gaussian filter for smoothing
        smoothed_image = cv2.GaussianBlur(grayscale_image, (7, 7), 2)

        # Apply Canny edge detection
        edges = cv2.Canny(smoothed_image, 50, 128)

        # Get image dimensions
        (height, width) = edges.shape

        # Determine maximum and minimum radii
        if radius is None:
            max_radius = max(height, width)
            min_radius = 3
        else:
            [max_radius, min_radius] = radius

        num_radii = max_radius - min_radius

        # Initialize accumulator array
        accumulator = np.zeros(
            (max_radius, height + 2 * max_radius, width + 2 * max_radius))
        detected_circles = np.zeros(
            (max_radius, height + 2 * max_radius, width + 2 * max_radius))

        # Precompute angles
        angles = np.arange(0, 360) * np.pi / 180
        edges_coordinates = np.argwhere(edges)

        # Iterate over radii
        for r_idx in range(num_radii):
            radius = min_radius + r_idx

            # Create circle template
            circle_template = np.zeros((2 * (radius + 1), 2 * (radius + 1)))
            # Center of the circle template
            (center_x, center_y) = (radius + 1, radius + 1)
            for angle in angles:
                x = int(np.round(radius * np.cos(angle)))
                y = int(np.round(radius * np.sin(angle)))
                circle_template[center_x + x, center_y + y] = 1

            template_size = np.argwhere(circle_template).shape[0]

            # Iterate over edge points
            for x, y in edges_coordinates:
                # Center the circle template over the edge point and update the accumulator array
                X = [x - center_x + max_radius, x + center_x + max_radius]
                Y = [y - center_y + max_radius, y + center_y + max_radius]
                accumulator[radius, X[0]:X[1], Y[0]:Y[1]] += circle_template

            accumulator[radius][accumulator[radius] <
                                threshold * template_size / radius] = 0

        # Find local maxima in the accumulator array
        for r, x, y in np.argwhere(accumulator):
            local_maxima = accumulator[r - region:r + region,
                                       x - region:x + region, y - region:y + region]
            try:
                p, a, b = np.unravel_index(
                    np.argmax(local_maxima), local_maxima.shape)
            except:
                continue
            detected_circles[r + (p - region), x +
                             (a - region), y + (b - region)] = 1

        # Extract circle information and draw circles on the image
        circle_coordinates = np.argwhere(detected_circles)
        for r, x, y in circle_coordinates:
            cv2.circle(modified_image, (y - max_radius,
                       x - max_radius), r, (255, 0, 0), 2)

        # Convert BGR image to RGB format
        # modified_image_rgb = cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)

        return modified_image










# def detect_ellipses(image, edge_thresholds=(50, 150), min_axis=30, max_axis=100, delta_a=2, delta_b=2, num_thetas=100, bin_threshold=0.4):
#     # Convert to grayscale and detect edges
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
#     edges = cv2.Canny(gray, edge_thresholds[0], edge_thresholds[1])

#     # Get the coordinates of edge pixels
#     edge_points = np.column_stack(np.where(edges > 0))

#     # Precompute the cosines and sines for all thetas
#     thetas = np.radians(np.arange(0, 360, 360 // num_thetas))
#     cos_thetas = np.cos(thetas)
#     sin_thetas = np.sin(thetas)

#     # Prepare to store ellipse candidate votes in an accumulator
#     accumulator = defaultdict(int)

#     # Generate ellipse candidate parameters once and use them efficiently
#     a_vals = np.arange(min_axis, max_axis, delta_a)
#     b_vals = np.arange(min_axis, max_axis, delta_b)

#     # Create ellipse candidates array with correct shape (a, b, cos, sin)
#     ellipse_candidates = []
#     for a in a_vals:
#         for b in b_vals:
#             for cos_t, sin_t in zip(cos_thetas, sin_thetas):
#                 ellipse_candidates.append([a, b, a * cos_t, b * sin_t])

#     # Convert the list to a NumPy array and reshape it to a 2D array
#     ellipse_candidates = np.array(ellipse_candidates)

#     # Efficient voting process: no nested loops
#     for x, y in edge_points:
#         # Reshape ellipse_candidates into a 2D array where each row is [a, b, cos, sin]
#         offsets = ellipse_candidates[:, 2:4]  # Extract cos and sin values for ellipse parameters
#         centers = np.array([x - offsets[:, 0], y - offsets[:, 1]]).T  # Calculate center offsets
#         for center in centers:
#             accumulator[tuple(center)] += 1  # Increment votes for each candidate center

#     # Filter valid ellipses based on bin threshold
#     detected_ellipses = [(x, y, a, b) for (x, y, a, b), votes in accumulator.items()
#                           if votes / num_thetas >= bin_threshold]

#     # Draw the detected ellipses
#     output_img = image.copy()
#     for x, y, a, b in detected_ellipses:
#         cv2.ellipse(output_img, (x, y), (a, b), 0, 0, 360, (0, 255, 0), 2)

#     return output_img




import cv2
import numpy as np

def fit_ellipse_manual(points):
    """
    Fit an ellipse manually using least squares method.
    :param points: Contour points (Nx2 numpy array)
    :return: (center_x, center_y), (major_axis, minor_axis), angle
    """
    if len(points) < 5:
        return None  # Not enough points to fit an ellipse

    # Compute centroid
    centroid = np.mean(points, axis=0)
    x_c, y_c = centroid

    # Shift points to centroid
    shifted_points = points - centroid

    # Compute covariance matrix
    cov_matrix = np.cov(shifted_points.T)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    eigenvalues = np.abs(eigenvalues)

    # Identify major and minor axes
    major_index = np.argmax(eigenvalues)
    minor_index = 1 - major_index

    # Compute axis lengths
    major_axis_length = 2.5 * np.sqrt(eigenvalues[major_index])
    minor_axis_length = 2.5 * np.sqrt(eigenvalues[minor_index])

    if major_axis_length < minor_axis_length:
        major_axis_length, minor_axis_length = minor_axis_length, major_axis_length

    # Compute ellipse orientation
    major_eigenvector = eigenvectors[:, major_index]
    angle = np.degrees(np.arctan2(major_eigenvector[1], major_eigenvector[0]))

    return (x_c, y_c), (major_axis_length, minor_axis_length), angle

def detect_ellipses_manual(image, min_axis=20, max_axis=300):
    """
    Detect ellipses in an image using manual least squares fitting.
    :param image: Input color image (RGB)
    :param min_axis: Minimum axis length for valid ellipses
    :param max_axis: Maximum axis length for valid ellipses
    :return: Image with detected ellipses drawn in RGB format
    """
    if image is None:
        print("Error: Image is None.")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    
    # Manual edge detection using Sobel filter
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    edges = np.sqrt(grad_x**2 + grad_y**2)
    edges = (edges / edges.max() * 255).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output_image = image.copy()
    
    for contour in contours:
        if len(contour) >= 5:
            contour_points = contour[:, 0, :]
            ellipse = fit_ellipse_manual(contour_points)
            
            if ellipse:
                (x, y), (major_axis, minor_axis), angle = ellipse
                
                # Validate size constraints
                if min_axis <= major_axis <= max_axis and min_axis <= minor_axis <= max_axis:
                    cv2.ellipse(output_image, ((int(x), int(y)), (int(major_axis), int(minor_axis)), angle),
                                (0, 255, 0), 2)
    # modified_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    return output_image
