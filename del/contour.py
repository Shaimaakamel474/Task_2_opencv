import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
from scipy.ndimage import gaussian_filter, sobel
from skimage.feature import canny

class ActiveContourModel:
    def __init__(self, image, initial_points, alpha=0.1, beta=0.1, gamma=0.1, w_line=0.0, w_edge=1.0, w_term=0.0,
                 max_iterations=100):
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
        return {'dx': dx, 'dy': dy, 'magnitude': magnitude}

    def _compute_E_cont(self, i, x, y):
        """Compute continuity energy"""
        n = len(self.contour_points)
        prev_point = self.contour_points[(i - 1) % n]
        avg_dist = np.mean([np.linalg.norm(self.contour_points[j] - self.contour_points[(j + 1) % n])
                            for j in range(n)])
        d = np.linalg.norm(np.array([x, y]) - prev_point)
        return (d - avg_dist) ** 2

    def _compute_E_curv(self, i, x, y):
        """Compute curvature energy"""
        n = len(self.contour_points)
        prev_point = self.contour_points[(i - 1) % n]
        next_point = self.contour_points[(i + 1) % n]
        return np.linalg.norm(prev_point - 2 * np.array([x, y]) + next_point) ** 2

    def _compute_E_image(self, x, y):
        """Compute image energy"""
        if x < 0 or y < 0 or x >= self.image.shape[1] or y >= self.image.shape[0]:
            return float('inf')
        x_int, y_int = int(x), int(y)
        E_line = self.image[y_int, x_int]
        E_edge = -self.gradients['magnitude'][y_int, x_int]
        return self.w_line * E_line + self.w_edge * E_edge

    def _compute_energy(self, i, x, y):
        """Compute total energy"""
        E_cont = self._compute_E_cont(i, x, y)
        E_curv = self._compute_E_curv(i, x, y)
        E_image = self._compute_E_image(x, y)
        return self.alpha * E_cont + self.beta * E_curv + self.gamma * E_image

    def evolve(self, max_iterations=250, return_intermediate=False):
        """Evolve the contour using the active contour model"""
        contours = []

        for i in range(max_iterations):
            # Update contour points
            self.snake = self.evolve()

            # Store intermediate contours at key iterations
            if return_intermediate and i in [0, 50, 100, 150, 200]:
                contours.append(self.snake.copy())

        # Always store the final contour
        contours.append(self.snake.copy())

        return contours if return_intermediate else self.snake

    def visualize_evolution(self, original_image, step=1):
        """Visualize the evolution of the contour"""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(original_image, cmap='gray')
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        iterations_to_show = np.linspace(0, len(self.contour_history) - 1,
                                         min(5, len(self.contour_history)),
                                         dtype=int)
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
        plt.show()

def run_active_contour_demo(image_path, alpha=0.1, beta=0.1, gamma=0.3, max_iterations=100):
    """Run the Active Contour Model demonstration"""
    original_image = np.array(Image.open(image_path).convert('L'))
    img = gaussian_filter(original_image, sigma=2)
    edges = canny(img, sigma=1.5, low_threshold=0.1, high_threshold=0.2)
    img = edges.astype(float)

    height, width = img.shape
    center_x, center_y = width // 2, height // 2
    a = width // 3
    b = height // 3
    theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    x = center_x + a * np.cos(theta)
    y = center_y + b * np.sin(theta)
    initial_contour = np.column_stack([x, y]).astype(int)

    snake = ActiveContourModel(img, initial_contour, alpha, beta, gamma, w_line=0.5, w_edge=3.0, max_iterations=max_iterations)
    final_contour = snake.evolve()
    snake.visualize_evolution(original_image)

# Example usage:
# run_active_contour_demo("path/to/image.jpg")
