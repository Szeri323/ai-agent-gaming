import cv2
import numpy as np
from PIL import Image

def load_image(image_path):
    """Load an image using OpenCV."""
    return cv2.imread(image_path)

def convert_to_gray(image):
    """Convert image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image, kernel_size=(5,5)):
    """Apply Gaussian blur to the image."""
    return cv2.GaussianBlur(image, kernel_size, 0)

def detect_edges(image):
    """Detect edges using Canny edge detection."""
    return cv2.Canny(image, 100, 200)

def main():
    # Example usage
    try:
        # Replace 'path_to_your_image.jpg' with your image path
        image_path = './game_states/walker-attack.png'
        
        # Load the image
        image = load_image(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return

        # Convert to grayscale
        gray_image = convert_to_gray(image)
        
        # Apply blur
        blurred_image = apply_gaussian_blur(gray_image)
        
        # Detect edges
        edges = detect_edges(blurred_image)
        
        # Display results
        cv2.imshow('Original Image', image)
        cv2.imshow('Grayscale Image', gray_image)
        cv2.imshow('Blurred Image', blurred_image)
        cv2.imshow('Edges', edges)
        
        # Wait for key press and close windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
