import cv2
import numpy as np
import pyautogui
import mss
import keyboard
import time
from typing import Tuple, Dict, List
import tensorflow as tf
from dataclasses import dataclass
from collections import deque
import win32gui
import win32con

@dataclass
class GameState:
    health: float = 100.0
    stamina: float = 100.0
    ammo: int = 0
    enemies_detected: List[Tuple[int, int, int, int]] = None  # List of (x, y, w, h) bounding boxes
    fps: float = 0.0

class DaysGoneAgent:
    def __init__(self):
        self.screen_capture = mss.mss()
        self.model = None
        self.actions = {
            'forward': 'w',
            'backward': 's',
            'left': 'a',
            'right': 'd',
            'jump': 'space',
            'attack': 'left',
            'aim': 'right'
        }
        # FPS calculation
        self.frame_times = deque(maxlen=30)
        # Color ranges for detection
        self.health_color_range = np.array([[0, 100, 100], [10, 255, 255]])  # Red color range
        self.stamina_color_range = np.array([[100, 100, 100], [130, 255, 255]])  # Blue color range
        self.enemy_color_range = np.array([[0, 0, 0], [180, 255, 30]])  # Dark color range for enemies
        
    def find_game_window(self) -> Tuple[int, int, int, int]:
        """Find the Days Gone window and return its position and size."""
        def callback(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd):
                window_title = win32gui.GetWindowText(hwnd)
                if "Days Gone" in window_title:
                    rect = win32gui.GetWindowRect(hwnd)
                    extra.append(rect)
            return True
        
        window_rects = []
        win32gui.EnumWindows(callback, window_rects)
        
        if not window_rects:
            raise Exception("Days Gone window not found! Make sure the game is running.")
        
        # Get the first matching window
        x, y, x1, y1 = window_rects[0]
        return x, y, x1, y1
    
    def capture_game_state(self) -> np.ndarray:
        """Capture only the game window."""
        try:
            # Get game window position and size
            x, y, x1, y1 = self.find_game_window()
            width = x1 - x
            height = y1 - y
            
            # Define the region to capture
            monitor = {
                "top": y,
                "left": x,
                "width": width,
                "height": height
            }
            
            # Capture the region
            screenshot = self.screen_capture.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
            
        except Exception as e:
            print(f"Error capturing game window: {e}")
            # Fallback to full screen capture if window not found
            monitor = self.screen_capture.monitors[1]
            screenshot = self.screen_capture.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess the captured frame for the model."""
        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0
        return frame
    
    def detect_health_and_stamina(self, frame: np.ndarray) -> Tuple[float, float]:
        """Detect health and stamina bars from the UI."""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define regions where health and stamina bars are typically located
        # Adjusted for window capture
        health_region = frame[50:70, 50:250]  # Adjust these coordinates based on your game UI
        stamina_region = frame[80:100, 50:250]
        
        # Create masks for health (red) and stamina (blue)
        health_mask = cv2.inRange(hsv[50:70, 50:250], self.health_color_range[0], self.health_color_range[1])
        stamina_mask = cv2.inRange(hsv[80:100, 50:250], self.stamina_color_range[0], self.stamina_color_range[1])
        
        # Calculate percentages
        health_percent = np.sum(health_mask > 0) / health_mask.size * 100
        stamina_percent = np.sum(stamina_mask > 0) / stamina_mask.size * 100
        
        return health_percent, stamina_percent
    
    def detect_enemies(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect enemies using basic motion and color detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to detect dark objects (potential enemies)
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on size and shape
        enemy_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 20 and h > 20:  # Minimum size threshold
                enemy_boxes.append((x, y, w, h))
        
        return enemy_boxes
    
    def detect_ammo(self, frame: np.ndarray) -> int:
        """Detect ammo count from the UI."""
        # Define region where ammo count is typically displayed
        ammo_region = frame[frame.shape[0]-100:frame.shape[0]-50, frame.shape[1]-200:frame.shape[1]-50]
        
        # Convert to grayscale
        gray = cv2.cvtColor(ammo_region, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Count white pixels (potential numbers)
        ammo_count = np.sum(thresh > 0) // 100  # Rough estimation, needs calibration
        
        return max(0, min(999, ammo_count))  # Clamp between 0 and 999
    
    def take_action(self, action: str):
        """Execute the given action in the game."""
        if action in self.actions:
            keyboard.press(self.actions[action])
            time.sleep(0.1)
            keyboard.release(self.actions[action])
    
    def run(self):
        """Main loop for the agent."""
        print("Starting Days Gone AI Agent...")
        print("Press 'q' to quit")
        
        # Create windows for visualization
        cv2.namedWindow('Raw Capture', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Processed Frame', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Detection View', cv2.WINDOW_NORMAL)
        
        # Set window sizes
        cv2.resizeWindow('Raw Capture', 800, 600)
        cv2.resizeWindow('Processed Frame', 224, 224)
        cv2.resizeWindow('Detection View', 800, 600)
        
        while True:
            if keyboard.is_pressed('q'):
                break
            
            # Calculate FPS
            frame_start_time = time.time()
            
            # Capture and preprocess game state
            frame = self.capture_game_state()
            processed_frame = self.preprocess_frame(frame)
            
            # Detect game state
            health, stamina = self.detect_health_and_stamina(frame)
            enemies = self.detect_enemies(frame)
            ammo = self.detect_ammo(frame)
            
            # Create detection visualization
            detection_view = frame.copy()
            
            # Draw health and stamina bars
            cv2.rectangle(detection_view, (50, 50), (250, 70), (0, 0, 0), 2)
            cv2.rectangle(detection_view, (50, 50), (50 + int(health * 2), 70), (0, 0, 255), -1)
            
            cv2.rectangle(detection_view, (50, 80), (250, 100), (0, 0, 0), 2)
            cv2.rectangle(detection_view, (50, 80), (50 + int(stamina * 2), 100), (255, 0, 0), -1)
            
            # Draw enemy bounding boxes
            for (x, y, w, h) in enemies:
                cv2.rectangle(detection_view, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add text information
            info_texts = [
                f"Health: {health:.1f}%",
                f"Stamina: {stamina:.1f}%",
                f"Ammo: {ammo}",
                f"Enemies: {len(enemies)}"
            ]
            
            for i, text in enumerate(info_texts):
                cv2.putText(detection_view, text, (10, 30 + i*30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Calculate and display FPS
            self.frame_times.append(time.time() - frame_start_time)
            fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
            cv2.putText(detection_view, f"FPS: {fps:.1f}", 
                       (detection_view.shape[1]-200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frames
            cv2.imshow('Raw Capture', frame)
            cv2.imshow('Processed Frame', (processed_frame * 255).astype(np.uint8))
            cv2.imshow('Detection View', detection_view)
            
            # Break loop if 'q' is pressed or windows are closed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.1)  # Prevent excessive CPU usage
        
        # Clean up
        cv2.destroyAllWindows()

if __name__ == "__main__":
    agent = DaysGoneAgent()
    agent.run() 