from ultralytics import YOLO
import cv2
import numpy as np

def load_model():
    """Load the YOLOv8 model."""
    # Download and load the YOLOv8n model
    model = YOLO('yolov8n.pt')
    return model

def detect_objects(model, image_path, conf_threshold=0.25):
    """
    Detect objects in an image using YOLOv8.
    
    Args:
        model: YOLOv8 model
        image_path: Path to the image file
        conf_threshold: Confidence threshold for detections
    
    Returns:
        image: Image with detections drawn
        results: Detection results
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Run detection
    results = model(image, conf=conf_threshold)[0]
    
    # Draw detections on the image
    for box in results.boxes:
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Get confidence and class
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = results.names[cls]
        
        # Draw rectangle and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{class_name} {conf:.2f}'
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image, results

def main():
    try:
        # Load the model
        print("Loading YOLOv8 model...")
        model = load_model()
        print("Model loaded successfully!")
        
        # Replace with your image path
        image_path = './game_states/crow-attack.png'
        
        # Detect objects
        print("Processing image...")
        image_with_detections, results = detect_objects(model, image_path)
        
        # Display results
        cv2.imshow('Object Detection', image_with_detections)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Print detection summary
        print("\nDetection Summary:")
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = results.names[cls]
            print(f"Detected {class_name} with confidence: {conf:.2f}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 