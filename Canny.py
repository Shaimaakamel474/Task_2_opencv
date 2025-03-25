import numpy as np
import cv2
import scipy.ndimage
from scipy.ndimage import convolve



def gaussian_filter(image, sigma):
    size = int(2 * np.ceil(3 * sigma) + 1)
    x, y = np.meshgrid(np.arange(-size//2 + 1, size//2 + 1),
                       np.arange(-size//2 + 1, size//2 + 1))
    gaussian_filter_matrix = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian_filter_matrix /= np.sum(gaussian_filter_matrix)
    image_gaussian = convolve(image, gaussian_filter_matrix, mode='constant', cval=0.0)

    return image_gaussian

def sobel_filters( smoothed_image):
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])
    Ix = cv2.filter2D(smoothed_image, -1, kernel_x)
    Iy = cv2.filter2D(smoothed_image, -1, kernel_y)
    
    # calculate gradient magnitude
    G = np.hypot(Ix, Iy)
    # normalize the gradient values[0:255]
    G = G / G.max() * 255
    # calculate the direction of the gradients
    theta = np.arctan2(Iy, Ix)

    return G, theta

def non_maximum_suppression( gradient_magnitude, gradient_direction):
    M, N = gradient_magnitude.shape
    # create a new image with the same shape as the original image 
    suppressed_image = np.zeros((M, N), dtype=np.int32)
    # convert radian to degree
    angle = np.rad2deg(gradient_direction) % 180
    
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                # init by white color
                q = 255
                r = 255

                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180): #0 degree , 180 degree
                    q = gradient_magnitude[i, j + 1]  #left
                    r = gradient_magnitude[i, j - 1]  #right
                
                elif (22.5 <= angle[i, j] < 67.5):  #45 degree
                    q = gradient_magnitude[i + 1, j - 1]  #bottom left
                    r = gradient_magnitude[i - 1, j + 1]  #top right
                
                elif (67.5 <= angle[i, j] < 112.5): #90 degree
                    q = gradient_magnitude[i + 1, j]  #bottom
                    r = gradient_magnitude[i - 1, j]  #top
                
                elif (112.5 <= angle[i, j] < 157.5):#135 degree
                    q = gradient_magnitude[i - 1, j - 1] #top left
                    r = gradient_magnitude[i + 1, j + 1] #bottom right

                
                #compare the current pixel with the pixels in the gradient direction
                if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                    suppressed_image[i, j] = gradient_magnitude[i, j]
                else:
                    suppressed_image[i, j] = 0

            except IndexError as e:
                pass

    return suppressed_image

def Double_thresholding(image  , low_threshold, high_threshold):
    
    M, N = image.shape
    # create a new image with the same shape as the original image 
    thresholded_image = np.zeros((M, N), dtype=np.int32)

    high_threshold = image.max() * high_threshold
    low_threshold = high_threshold * low_threshold
    # loop over the image
    strong_i, strong_j = np.where(image >= high_threshold)
    zeros_i, zeros_j = np.where(image < low_threshold)
    weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold))
    # get the edge pixels
    thresholded_image[strong_i, strong_j] = 255
    thresholded_image[zeros_i, zeros_j] = 0
    thresholded_image[weak_i, weak_j] = 75
    return thresholded_image

def hysteresis(image):
    M, N = image.shape
    weak = 75
    strong = 255
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (image[i, j] == weak):
                try:
                    if ((image[i + 1, j - 1] == strong) or (image[i + 1, j] == strong) or (image[i + 1, j + 1] == strong)
                            or (image[i, j - 1] == strong) or (image[i, j + 1] == strong)
                            or (image[i - 1, j - 1] == strong) or (image[i - 1, j] == strong) or (image[i - 1, j + 1] == strong)):
                        image[i, j] = strong
                    else:
                        image[i, j] = 0
                except IndexError as e:
                    pass
    return image

def canny_edge_detection(image, sigma, low_threshold, high_threshold):
    # Step 1: Smooth the image using a Gaussian filter
    smoothed_image = gaussian_filter(image, sigma)
    # Step 2: Find the intensity gradients of the image
    gradient_magnitude, gradient_direction = sobel_filters(smoothed_image)
    # Step 3: Apply non-maximum suppression
    suppressed_image = non_maximum_suppression(gradient_magnitude, gradient_direction)
    # Step 4: Apply double thresholding
    thresholded_image = Double_thresholding(suppressed_image, low_threshold, high_threshold)
    # Step 5: Track edge by hysteresis
    canny_image = hysteresis(thresholded_image)
    return np.array(canny_image  , dtype=np.uint8)
