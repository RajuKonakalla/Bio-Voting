import cv2
import pickle
import numpy as np
import os
import time
import threading
import argparse
from datetime import datetime
import logging

class FacialRecognitionSystem:
    """Advanced facial recognition system with improved user experience and error handling."""
    
    def __init__(self):
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("facial_recognition.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Set up data directory
        self.data_dir = 'data/'
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Configuration settings
        self.frame_width = 740
        self.frame_height = 580
        self.resize_face = (50, 50)
        self.frames_to_capture = 27
        self.capture_interval = 2
        self.cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        # Camera setup will be done in a separate method to ensure proper initialization
        self.video = None
        self.is_camera_ready = False
        
        # Initialize face detector
        try:
            self.face_detector = cv2.CascadeClassifier(self.cascade_file)
            if self.face_detector.empty():
                self.logger.error("Failed to load face cascade classifier!")
                raise Exception("Face detector initialization failed")
            self.logger.info("Face detector initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing face detector: {str(e)}")
            raise
    
    def initialize_camera(self):
        """Initialize camera with multiple attempts if needed."""
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts and not self.is_camera_ready:
            attempt += 1
            self.logger.info(f"Initializing camera - attempt {attempt}/{max_attempts}")
            
            try:
                self.video = cv2.VideoCapture(0)
                self.video.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                
                # Check if camera opened successfully
                if not self.video.isOpened():
                    self.logger.error(f"Could not open webcam - attempt {attempt}")
                    time.sleep(1)
                    continue
                
                # Read a test frame to verify camera is working
                ret, frame = self.video.read()
                if not ret or frame is None:
                    self.logger.error(f"Could not read frame from camera - attempt {attempt}")
                    self.video.release()
                    time.sleep(1)
                    continue
                
                self.is_camera_ready = True
                self.logger.info("Camera initialized successfully")
                return True
            
            except Exception as e:
                self.logger.error(f"Error initializing camera: {str(e)} - attempt {attempt}")
                if self.video is not None:
                    self.video.release()
                time.sleep(1)
        
        if not self.is_camera_ready:
            self.logger.error("Failed to initialize camera after multiple attempts")
            return False
    
    def get_user_input(self):
        """Get user ID with validation."""
        while True:
            user_id = input("\nEnter your Aadhar number (12 digits): ")
            if len(user_id) == 12 and user_id.isdigit():
                return user_id
            print("Invalid Aadhar number! Please enter a 12-digit number.")
    
    def capture_facial_data(self, user_id):
        """Capture facial data from webcam."""
        if not self.initialize_camera():
            self.logger.error("Could not capture facial data - camera not available")
            return None
        
        faces_data = []
        frame_count = 0
        processing_count = 0
        start_time = time.time()
        
        # Create window with focus
        window_name = 'Facial Recognition System'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        
        self.logger.info(f"Starting face capture for user ID: {user_id}")
        print(f"\nCapturing face data... Please look at the camera.")
        print(f"Target: {self.frames_to_capture} frames")
        
        try:
            while True:
                ret, frame = self.video.read()
                if not ret:
                    self.logger.warning("Failed to grab frame")
                    continue
                
                frame_count += 1
                
                # Create a copy for display
                display_frame = frame.copy()
                
                # Face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # Process the largest face if multiple faces are detected
                if len(faces) > 0:
                    # Find the largest face based on area
                    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                    x, y, w, h = largest_face
                    
                    # Draw rectangle around the face
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Process face at specified intervals
                    if processing_count < self.frames_to_capture and frame_count % self.capture_interval == 0:
                        # Extract and process face
                        face_img = frame[y:y+h, x:x+w]
                        resized_face = cv2.resize(face_img, self.resize_face)
                        faces_data.append(resized_face)
                        processing_count += 1
                
                # Display progress information
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                cv2.putText(display_frame, f"Captured: {processing_count}/{self.frames_to_capture}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"FPS: {fps:.2f}", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"User ID: {user_id}", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display instructions
                instruction = "Press 'q' to quit or wait for automatic completion"
                cv2.putText(display_frame, instruction, 
                            (10, display_frame.shape[0] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Add progress bar
                progress = int((processing_count / self.frames_to_capture) * display_frame.shape[1])
                cv2.rectangle(display_frame, (0, display_frame.shape[0] - 10), 
                             (progress, display_frame.shape[0]), (0, 255, 0), -1)
                
                # Show frame
                cv2.imshow(window_name, display_frame)
                
                # Check for exit conditions
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or processing_count >= self.frames_to_capture:
                    break
                    
        except Exception as e:
            self.logger.error(f"Error during face capture: {str(e)}")
            
        finally:
            # Clean up
            if self.video:
                self.video.release()
            cv2.destroyAllWindows()
        
        self.logger.info(f"Face capture complete. Captured {len(faces_data)} frames.")
        
        # Ensure we have the right number of frames
        if len(faces_data) < self.frames_to_capture:
            self.logger.warning(f"Captured only {len(faces_data)} frames, expected {self.frames_to_capture}")
            missing = self.frames_to_capture - len(faces_data)
            # If we have at least one face, duplicate it to fill the gap
            if len(faces_data) > 0:
                self.logger.info(f"Duplicating existing frames to reach target count")
                while len(faces_data) < self.frames_to_capture:
                    faces_data.append(faces_data[-1])
            else:
                self.logger.error("No faces captured")
                return None
        
        return np.array(faces_data)
    
    def save_data(self, user_id, faces_data):
        """Save the captured facial data and associated user ID."""
        if faces_data is None or len(faces_data) == 0:
            self.logger.error("No face data to save")
            return False
        
        try:
            # Reshape data for storage
            faces_data_reshaped = faces_data.reshape((self.frames_to_capture, -1))
            
            # Save user IDs
            names_file = os.path.join(self.data_dir, 'names.pkl')
            names = [user_id] * self.frames_to_capture
            
            if os.path.exists(names_file):
                with open(names_file, 'rb') as f:
                    existing_names = pickle.load(f)
                names = existing_names + names
            
            with open(names_file, 'wb') as f:
                pickle.dump(names, f)
            
            # Save face data
            faces_file = os.path.join(self.data_dir, 'faces_data.pkl')
            
            if os.path.exists(faces_file):
                with open(faces_file, 'rb') as f:
                    existing_faces = pickle.load(f)
                faces_data_combined = np.append(existing_faces, faces_data_reshaped, axis=0)
            else:
                faces_data_combined = faces_data_reshaped
            
            with open(faces_file, 'wb') as f:
                pickle.dump(faces_data_combined, f)
            
            self.logger.info(f"Successfully saved data for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            return False
    
    def run(self):
        """Run the facial recognition data collection process."""
        print("\n===== Advanced Facial Recognition System =====")
        print("This system will capture your facial data for authentication.")
        
        # Get user information
        user_id = self.get_user_input()
        
        # Capture faces
        start_time = time.time()
        faces_data = self.capture_facial_data(user_id)
        
        if faces_data is not None and len(faces_data) > 0:
            # Save data
            if self.save_data(user_id, faces_data):
                elapsed = time.time() - start_time
                print(f"\n✅ Registration successful for user {user_id}")
                print(f"Captured {len(faces_data)} facial images in {elapsed:.2f} seconds")
                print(f"Data saved in {os.path.abspath(self.data_dir)}")
                return True
            else:
                print("\n❌ Failed to save data. Please try again.")
                return False
        else:
            print("\n❌ Failed to capture facial data. Please ensure your face is visible to the camera.")
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Facial Recognition System")
    parser.add_argument("--frames", type=int, default=51, 
                        help="Number of frames to capture (default: 51)")
    parser.add_argument("--interval", type=int, default=2, 
                        help="Frame interval for capture (default: 2)")
    args = parser.parse_args()
    
    try:
        system = FacialRecognitionSystem()
        system.frames_to_capture = args.frames
        system.capture_interval = args.interval
        system.run()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        logging.error(f"Unhandled exception: {str(e)}", exc_info=True)