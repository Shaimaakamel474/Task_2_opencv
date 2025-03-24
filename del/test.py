from ActiveContour import run_active_contour_demo

image_path = "apple.jpg"  # Replace with your image path
results = run_active_contour_demo(
    image_path,
    initial_contour=None,
    alpha=0.1,
    beta=0.2,
    gamma=0.8,
    max_iterations=200,
    visualize=True,
    output_path="contour_result.png"
)