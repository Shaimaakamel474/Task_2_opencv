import numpy as np
import cv2
from scipy.interpolate import splprep, splev


class ActiveContour:
    def __init__(self, image, x_center, y_center, radius, iterations, alpha=0.01, beta=0.1, gamma=0.1):
        """
        Initialize active contour (snake) parameters

        Args:
            image: Input RGB image
            x_center: x-coordinate of initial contour center
            y_center: y-coordinate of initial contour center
            radius: Initial radius of the contour
            iterations: Number of evolution iterations
            alpha: Elasticity parameter (continuity)
            beta: Rigidity parameter (curvature)
            gamma: Image force parameter
        """
        self.image = image
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.iterations = iterations

        # Convert to grayscale for energy calculations
        self.gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.gradient = self._compute_gradient()

        # Initialize circular contour
        self.points = self._initialize_contour(x_center, y_center, radius)
        self.num_points = len(self.points)

    def _initialize_contour(self, x_center, y_center, radius, num_points=60):
        """Initialize circular contour points"""
        theta = np.linspace(0, 2 * np.pi, num_points)
        x = x_center + radius * np.cos(theta)
        y = y_center + radius * np.sin(theta)
        return np.column_stack((x, y))

    def _compute_gradient(self):
        """Compute image gradient for external energy"""
        gray_blur = cv2.GaussianBlur(self.gray, (5, 5), 0)
        grad_x = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        return cv2.normalize(grad_mag, None, 0, 1, cv2.NORM_MINMAX)

    def _get_internal_energy(self, point_idx, new_pos):
        """Calculate internal energy (continuity + curvature)"""
        prev_idx = (point_idx - 1) % self.num_points
        next_idx = (point_idx + 1) % self.num_points

        # Continuity (distance to neighbors)
        d_prev = np.linalg.norm(new_pos - self.points[prev_idx])
        d_next = np.linalg.norm(self.points[next_idx] - new_pos)
        continuity = (d_prev ** 2 + d_next ** 2)

        # Curvature
        curvature = np.linalg.norm(2 * new_pos -
                                   self.points[prev_idx] -
                                   self.points[next_idx]) ** 2

        return self.alpha * continuity + self.beta * curvature

    def _get_external_energy(self, x, y):
        """Calculate external energy from image gradient"""
        x, y = int(x), int(y)
        if 0 <= x < self.gradient.shape[1] and 0 <= y < self.gradient.shape[0]:
            return -self.gamma * self.gradient[y, x]
        return float('inf')

    def evolve(self):
        """Evolve the contour using greedy algorithm"""
        window_size = 5  # Search window size

        for _ in range(self.iterations):
            new_points = self.points.copy()

            for i in range(self.num_points):
                x, y = self.points[i]
                min_energy = float('inf')
                best_pos = self.points[i].copy()

                # Search in a window around current point
                for dx in range(-window_size, window_size + 1):
                    for dy in range(-window_size, window_size + 1):
                        new_x, new_y = x + dx, y + dy
                        internal = self._get_internal_energy(i, np.array([new_x, new_y]))
                        external = self._get_external_energy(new_x, new_y)
                        total_energy = internal + external

                        if total_energy < min_energy:
                            min_energy = total_energy
                            best_pos = [new_x, new_y]

                new_points[i] = best_pos

            self.points = new_points
        return self.get_output_image()

    def get_chain_code(self):
        """Convert contour to Freeman chain code"""
        directions = [(1, 0), (1, 1), (0, 1), (-1, 1),
                      (-1, 0), (-1, -1), (0, -1), (1, -1)]
        chain = []

        points_int = self.points.astype(int)
        for i in range(self.num_points):
            curr = points_int[i]
            next_p = points_int[(i + 1) % self.num_points]
            diff = next_p - curr

            # Find closest direction
            for code, (dx, dy) in enumerate(directions):
                if diff[0] == dx and diff[1] == dy:
                    chain.append(code)
                    break
                elif np.linalg.norm(diff - np.array([dx, dy])) < 1.5:
                    chain.append(code)
                    break

        return chain

    def get_perimeter(self):
        """Calculate perimeter of the contour"""
        perimeter = 0
        for i in range(self.num_points):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % self.num_points]
            perimeter += np.linalg.norm(p1 - p2)
        return perimeter

    def get_area(self):
        """Calculate area inside the contour using Green's theorem"""
        area = 0
        points = self.points
        for i in range(self.num_points):
            j = (i + 1) % self.num_points
            area += points[i, 0] * points[j, 1]
            area -= points[j, 0] * points[i, 1]
        return abs(area) / 2

    def get_output_image(self):
        """Return image with contour drawn"""
        output = self.image.copy()
        points_int = self.points.astype(int)

        # Draw contour
        for i in range(self.num_points):
            p1 = tuple(points_int[i])
            p2 = tuple(points_int[(i + 1) % self.num_points])
            cv2.line(output, p1, p2, (0, 255, 0), 2)

        return output


# Example usage in your MainWindow class:
def active_contour_processing(self):
    if self.org_ImgWidget.image is not None:
        # Example parameters - you can get these from UI elements
        x_center = self.org_ImgWidget.width() // 2
        y_center = self.org_ImgWidget.height() // 2
        radius = min(self.org_ImgWidget.width(), self.org_ImgWidget.height()) // 4
        iterations = 50

        # Get parameters from UI sliders or inputs if available
        alpha = 0.01  # Elasticity
        beta = 0.1  # Rigidity
        gamma = 0.1  # Image force

        snake = ActiveContour(
            self.org_ImgWidget.image,
            x_center, y_center, radius,
            iterations, alpha, beta, gamma
        )

        result = snake.evolve()
        chain_code = snake.get_chain_code()
        perimeter = snake.get_perimeter()
        area = snake.get_area()

        self.Output_Widget_1.Set_Image(result)
        self.Output_Widget_1.display_RGBImg()

        # Print results (you could display these in UI)
        print(f"Chain Code: {chain_code}")
        print(f"Perimeter: {perimeter:.2f} pixels")
        print(f"Area: {area:.2f} square pixels")