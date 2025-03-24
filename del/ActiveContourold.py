import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
from scipy.ndimage import gaussian_filter, sobel
from skimage.feature import canny


class ActiveContourModel:
    def __init__(self, image, initial_points, alpha=0.1, beta=0.1, gamma=0.1, w_line=0.0, w_edge=1.0, w_term=0.0,
                 max_iterations=100):
        """
        Initialize the Active Contour Model
        Parameters:
        -----------
        image : numpy.ndarray
            Edge-detected image on which to evolve the contour
        initial_points : numpy.ndarray
            Initial contour points (x, y) coordinates
        alpha : float
            Weight for continuity energy
        beta : float
            Weight for curvature energy
        gamma : float
            Weight for image energy
        w_line : float
            Weight for intensity energy term
        w_edge : float
            Weight for edge energy term
        w_term : float
            Weight for termination energy term
        max_iterations : int
            Maximum number of iterations
        """
        self.image = image
        self.contour_points = np.array(initial_points)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.w_line = w_line
        self.w_edge = w_edge
        self.w_term = w_term
        self.max_iterations = max_iterations
        self.gradients = self._compute_gradients()
        self.contour_history = [self.contour_points.copy()]

    def _compute_gradients(self):
        """Compute image gradients using Sobel filter"""
        dx = sobel(self.image, axis=1, mode='constant')
        dy = sobel(self.image, axis=0, mode='constant')
        magnitude = np.sqrt(dx ** 2 + dy ** 2)
        return {
            'dx': dx,
            'dy': dy,
            'magnitude': magnitude
        }

    def _compute_E_cont(self, i, x, y):
        """Compute continuity energy at point (x, y) for contour point i"""
        n = len(self.contour_points)
        prev_point = self.contour_points[(i - 1) % n]
        next_point = self.contour_points[(i + 1) % n]
        avg_dist = np.mean([np.linalg.norm(self.contour_points[j] - self.contour_points[(j + 1) % n])
                            for j in range(n)])
        d = np.linalg.norm(np.array([x, y]) - prev_point)
        return (d - avg_dist) ** 2

    def _compute_E_curv(self, i, x, y):
        """Compute curvature energy at point (x, y) for contour point i"""
        n = len(self.contour_points)
        prev_point = self.contour_points[(i - 1) % n]
        next_point = self.contour_points[(i + 1) % n]
        return np.linalg.norm(prev_point - 2 * np.array([x, y]) + next_point) ** 2

    def _compute_E_image(self, x, y):
        """Compute image energy at point (x, y)"""
        if x < 0 or y < 0 or x >= self.image.shape[1] or y >= self.image.shape[0]:
            return float('inf')
        x_int, y_int = int(x), int(y)
        E_line = self.image[y_int, x_int]
        E_edge = -self.gradients['magnitude'][y_int, x_int]
        E_term = 0
        return self.w_line * E_line + self.w_edge * E_edge + self.w_term * E_term

    def _compute_energy(self, i, x, y):
        """Compute total energy at point (x, y) for contour point i"""
        E_cont = self._compute_E_cont(i, x, y)
        E_curv = self._compute_E_curv(i, x, y)
        E_image = self._compute_E_image(x, y)
        return self.alpha * E_cont + self.beta * E_curv + self.gamma * E_image

    def evolve(self):
        """Evolve the snake using greedy algorithm"""
        for iteration in range(self.max_iterations):
            moved = False
            new_contour = self.contour_points.copy()
            for i in range(len(self.contour_points)):
                x, y = self.contour_points[i]
                current_energy = self._compute_energy(i, x, y)
                min_energy = current_energy
                min_point = (x, y)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if (nx < 0 or ny < 0 or
                                nx >= self.image.shape[1] or
                                ny >= self.image.shape[0]):
                            continue
                        energy = self._compute_energy(i, nx, ny)
                        if energy < min_energy:
                            min_energy = energy
                            min_point = (nx, ny)
                if min_point != (x, y):
                    new_contour[i] = np.array(min_point)
                    moved = True
            self.contour_points = new_contour
            self.contour_history.append(self.contour_points.copy())
            if not moved:
                print(f"Converged after {iteration + 1} iterations")
                break
        return self.contour_points

    def to_chain_code(self):
        """Convert contour to chain code representation (8-directional)"""
        chain_code = []
        n = len(self.contour_points)
        for i in range(n):
            current = self.contour_points[i]
            next_point = self.contour_points[(i + 1) % n]
            dx = next_point[0] - current[0]
            dy = next_point[1] - current[1]
            if dx == 1 and dy == 0:
                code = 0
            elif dx == 1 and dy == 1:
                code = 1
            elif dx == 0 and dy == 1:
                code = 2
            elif dx == -1 and dy == 1:
                code = 3
            elif dx == -1 and dy == 0:
                code = 4
            elif dx == -1 and dy == -1:
                code = 5
            elif dx == 0 and dy == -1:
                code = 6
            elif dx == 1 and dy == -1:
                code = 7
            else:
                angle = math.atan2(dy, dx)
                code = round(4 * angle / math.pi) % 8
            chain_code.append(code)
        return chain_code

    def compute_perimeter(self):
        """Compute the perimeter of the contour"""
        perimeter = 0
        n = len(self.contour_points)
        for i in range(n):
            current = self.contour_points[i]
            next_point = self.contour_points[(i + 1) % n]
            perimeter += np.linalg.norm(next_point - current)
        return perimeter

    def compute_area(self):
        """Compute the area inside the contour using the shoelace formula"""
        x = self.contour_points[:, 0]
        y = self.contour_points[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return area

    def visualize_evolution(self, original_image, step=1, save_path=None):
        """Visualize the evolution of the contour on the original image"""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(original_image, cmap='gray')  # Use the original image for visualization
        iterations_to_show = np.linspace(0, len(self.contour_history) - 1,
                                         min(5, len(self.contour_history)),
                                         dtype=int)
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        for i, iter_idx in enumerate(iterations_to_show):
            contour = self.contour_history[iter_idx]
            ax.plot(contour[:, 0], contour[:, 1], 'o-',
                    color=colors[i % len(colors)],
                    linewidth=1, markersize=3,
                    label=f'Iteration {iter_idx}')
        final_contour = self.contour_history[-1]
        ax.plot(final_contour[:, 0], final_contour[:, 1], 'o-',
                color='k', linewidth=2, markersize=5,
                label='Final Contour')
        ax.set_title('Active Contour Model Evolution')
        ax.legend()
        if save_path:
            plt.savefig(save_path)
        plt.show()


def run_active_contour_demo(image_path, initial_contour=None, alpha=0.1, beta=0.1, gamma=0.3,
                            max_iterations=100, visualize=True):
    """Run the Active Contour Model demonstration"""
    # Load and preprocess the image
    original_image = np.array(Image.open(image_path).convert('L'))  # Keep the original image
    img = gaussian_filter(original_image, sigma=2)  # Smooth the image
    edges = canny(img, sigma=1.5, low_threshold=0.1, high_threshold=0.2) # Detect edges using Canny
    img = edges.astype(float)  # Use edge-detected image for active contour

    # Create initial contour if not provided
    if initial_contour is None:
        # Create an elliptical contour centered in the image
        height, width = img.shape
        center_x, center_y = width // 2, height // 2
        a = width // 2  # Semi-major axis
        b = height // 2  # Semi-minor axis
        theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        x = center_x + a * np.cos(theta)
        y = center_y + b * np.sin(theta)
        initial_contour = np.column_stack([x, y]).astype(int)

    # Initialize and run the active contour model
    snake = ActiveContourModel(
        img, initial_contour,
        alpha=alpha, beta=beta, gamma=gamma,
        w_line=0.5, w_edge=3.0,  # Emphasize edge energy
        max_iterations=max_iterations
    )

    # Evolve the contour
    final_contour = snake.evolve()

    # Compute chain code, perimeter, and area
    chain_code = snake.to_chain_code()
    perimeter = snake.compute_perimeter()
    area = snake.compute_area()

    print(f"Number of iterations: {len(snake.contour_history)}")
    print(f"Chain code: {chain_code}")
    print(f"Perimeter: {perimeter:.2f} pixels")
    print(f"Area: {area:.2f} square pixels")

    # Visualize the results on the original image
    if visualize:
        snake.visualize_evolution(original_image)

    return {
        'contour': final_contour,
        'chain_code': chain_code,
        'perimeter': perimeter,
        'area': area,
        'contour_history': snake.contour_history
    }


if __name__ == "__main__":
    image_path = "../img.jpeg"  # Replace with your image path
    results = run_active_contour_demo(
        image_path,
        initial_contour=None,
        alpha=0.1,
        beta=0.2,
        gamma=0.9,
        max_iterations=200,
        visualize=True
    )
