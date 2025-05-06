from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch
import tkinter as tk
from tkinter import messagebox
import pandas as pd

def speak(str1):
    try:
        speak = Dispatch("SAPI.SpVoice")
        speak.Speak(str1)
    except:
        print(str1)  # Fallback if speech fails

def center_window(window, width, height):
    """Center a tkinter window on the screen"""
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    window.geometry(f"{width}x{height}+{x}+{y}")

def show_already_voted_window(voter_name):
    """Show a window when voter has already voted"""
    already_voted = tk.Tk()
    already_voted.title("ALREADY VOTED")
    already_voted.configure(bg="#f0f0f0")
    center_window(already_voted, 500, 400)
    
    # Add a border frame
    frame = tk.Frame(already_voted, bd=5, relief=tk.RIDGE, bg="#ffffff")
    frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
    
    # Header
    header = tk.Label(frame, text="ALREADY VOTED", font=("Arial", 16, "bold"), 
                     bg="#ff0000", fg="white", pady=10)
    header.pack(fill=tk.X)
    
    # Content
    content_frame = tk.Frame(frame, bg="white")
    content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
    
    # Information
    tk.Label(content_frame, text=f"Voter ID: {voter_name}", font=("Arial", 12), 
             bg="white", anchor="w").pack(fill=tk.X, pady=5)
    
    tk.Label(content_frame, text="You have already cast your vote!", font=("Arial", 12, "bold"), 
             bg="white", fg="#ff0000", anchor="w").pack(fill=tk.X, pady=5)
    
    # Verification info
    verify_frame = tk.Frame(content_frame, bg="#fff0f0", bd=1, relief=tk.GROOVE)
    verify_frame.pack(fill=tk.X, pady=10, padx=5)
    
    tk.Label(verify_frame, text="Each voter is allowed to vote only once.", 
             font=("Arial", 11), bg="#fff0f0", fg="#660000", pady=5).pack()
    
    # View previous vote button
    def view_previous_vote():
        try:
            df = pd.read_csv("Votes.csv")
            voter_row = df[df['NAME'] == voter_name].iloc[0]
            messagebox.showinfo("Previous Vote", 
                               f"You voted for {voter_row['VOTE']} on {voter_row['DATE']} at {voter_row['TIME']}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not retrieve previous vote: {str(e)}")
    
    # Close button
    btn_frame = tk.Frame(frame, bg="white")
    btn_frame.pack(fill=tk.X, pady=10)
    
    tk.Button(btn_frame, text="View My Previous Vote", command=view_previous_vote, 
              bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), 
              padx=10).pack(side=tk.LEFT, padx=10)
    
    tk.Button(btn_frame, text="Exit", command=already_voted.destroy, 
              bg="#f44336", fg="white", font=("Arial", 10, "bold"), 
              padx=10).pack(side=tk.RIGHT, padx=10)
    
    already_voted.mainloop()

def show_receipt(voter_name, party, date, time):
    """Show a graphical receipt window with vote details"""
    receipt_window = tk.Tk()
    receipt_window.title("VOTE CONFIRMATION RECEIPT")
    receipt_window.configure(bg="#f0f0f0")
    center_window(receipt_window, 500, 400)
    
    # Add a border frame
    frame = tk.Frame(receipt_window, bd=5, relief=tk.RIDGE, bg="#ffffff")
    frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
    
    # Header
    header = tk.Label(frame, text="OFFICIAL VOTING RECEIPT", font=("Arial", 16, "bold"), 
                     bg="#0066cc", fg="white", pady=10)
    header.pack(fill=tk.X)
    
    # Content
    content_frame = tk.Frame(frame, bg="white")
    content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
    
    # Receipt details
    tk.Label(content_frame, text=f"Voter ID: {voter_name}", font=("Arial", 12), 
             bg="white", anchor="w").pack(fill=tk.X, pady=5)
    
    tk.Label(content_frame, text=f"Vote Cast For: {party}", font=("Arial", 12, "bold"), 
             bg="white", fg="#009900", anchor="w").pack(fill=tk.X, pady=5)
    
    tk.Label(content_frame, text=f"Date: {date}", font=("Arial", 12), 
             bg="white", anchor="w").pack(fill=tk.X, pady=5)
    
    tk.Label(content_frame, text=f"Time: {time}", font=("Arial", 12), 
             bg="white", anchor="w").pack(fill=tk.X, pady=5)
    
    # Unique transaction ID
    import hashlib
    transaction_id = hashlib.md5(f"{voter_name}{party}{date}{time}".encode()).hexdigest()[:8].upper()
    tk.Label(content_frame, text=f"Transaction ID: {transaction_id}", font=("Arial", 12), 
             bg="white", anchor="w").pack(fill=tk.X, pady=5)
    
    # Verification message
    verify_frame = tk.Frame(content_frame, bg="#e6ffe6", bd=1, relief=tk.GROOVE)
    verify_frame.pack(fill=tk.X, pady=10, padx=5)
    
    tk.Label(verify_frame, text="Your vote has been securely recorded", 
             font=("Arial", 11), bg="#e6ffe6", fg="#006600", pady=5).pack()
    
    # Buttons frame
    btn_frame = tk.Frame(frame, bg="white")
    btn_frame.pack(fill=tk.X, pady=10)
    
    def verify_vote():
        # Read the CSV and check if the vote exists
        try:
            df = pd.read_csv("Votes.csv")
            vote_exists = any((df['NAME'] == voter_name) & (df['VOTE'] == party))
            if vote_exists:
                messagebox.showinfo("Verification", "Your vote is confirmed in the database!")
            else:
                messagebox.showerror("Verification Failed", "Vote record not found. Please contact election officials.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not verify: {str(e)}")
    
    def print_receipt():
        """Save receipt to text file"""
        try:
            with open(f"receipt_{voter_name}_{transaction_id}.txt", "w") as f:
                f.write(f"OFFICIAL VOTING RECEIPT\n")
                f.write(f"=====================\n\n")
                f.write(f"Voter ID: {voter_name}\n")
                f.write(f"Vote Cast For: {party}\n")
                f.write(f"Date: {date}\n")
                f.write(f"Time: {time}\n")
                f.write(f"Transaction ID: {transaction_id}\n\n")
                f.write(f"Your vote has been securely recorded.\n")
            messagebox.showinfo("Receipt Saved", f"Receipt saved as receipt_{voter_name}_{transaction_id}.txt")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save receipt: {str(e)}")
    
    def close_app():
        receipt_window.destroy()
    
    tk.Button(btn_frame, text="Verify Vote", command=verify_vote, 
              bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), 
              padx=10).pack(side=tk.LEFT, padx=10)
    
    tk.Button(btn_frame, text="Save Receipt", command=print_receipt, 
              bg="#2196F3", fg="white", font=("Arial", 10, "bold"), 
              padx=10).pack(side=tk.LEFT, padx=10)
    
    tk.Button(btn_frame, text="Exit", command=close_app, 
              bg="#f44336", fg="white", font=("Arial", 10, "bold"), 
              padx=10).pack(side=tk.RIGHT, padx=10)
    
    receipt_window.mainloop()

# Main program
def initialize_camera(max_attempts=3):
    """Initialize webcam with retries and checks."""
    attempt = 0
    video = None
    while attempt < max_attempts:
        attempt += 1
        video = cv2.VideoCapture(0)
        if video.isOpened():
            ret, frame = video.read()
            if ret and frame is not None:
                print(f"Webcam initialized successfully on attempt {attempt}")
                return video
            else:
                print(f"Failed to read frame from webcam on attempt {attempt}")
                video.release()
        else:
            print(f"Failed to open webcam on attempt {attempt}")
            if video is not None:
                video.release()
        time.sleep(1)
    print("Failed to initialize webcam after multiple attempts")
    return None

def main():
    # Initialize video capture and face recognition
    video = initialize_camera()
    if video is None:
        print("Error: Could not open webcam. Exiting.")
        speak("Error: Could not open webcam. Please check your camera and try again.")
        messagebox.showerror("Error", "Could not open webcam. Please check your camera and try again.")
        return

    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if not os.path.exists('data/'):
        os.makedirs('data/')

    try:
        with open('data/names.pkl', 'rb') as f:
            LABELS = pickle.load(f)

        with open('data/faces_data.pkl', 'rb') as f:
            FACES = pickle.load(f)
    except FileNotFoundError:
        print("Error: Training data files not found!")
        speak("Error: Required face recognition data not found. Please run the training program first.")
        messagebox.showerror("Error", "Training data not found! Please run the face registration program first.")
        return

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)

    # Load background image
    imgBackground = cv2.imread("background.png")
    imgBackground = cv2.resize(imgBackground, (1550, 780))
    # Check if background image loaded correctly
    if imgBackground is None:
        print("Error: Could not load 'background.png'. Using plain background instead.")
        speak("Background image not found. Using plain background.")
        imgBackground = np.zeros((900, 1400, 3), dtype=np.uint8)
        imgBackground[:] = (50, 50, 50)  # Dark gray background

    COL_NAMES = ['NAME', 'VOTE', 'DATE', 'TIME']

    # Ensure Votes.csv exists with headers
    if not os.path.isfile("Votes.csv"):
        with open("Votes.csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(COL_NAMES)

    def check_if_exists(value):
        try:
            with open("Votes.csv", "r") as csvfile:
                reader = csv.reader(csvfile)
                next(reader, None)  # Skip header
                for row in reader:
                    if row and row[0] == value:
                        return True
        except FileNotFoundError:
            print("Votes.csv not found. Creating a new one.")
        return False

    identified_person = None
    voting_mode = False
    vote_instructions_given = False
    recognition_time = 0
    recognition_countdown = 0
    recognition_failed = False
    already_voted_status = False

    speak("Welcome to the secure electronic voting system")

    # Main loop
    while True:
        ret, frame = video.read()
        if not ret:
            print("Failed to capture video frame. Check camera connection.")
            time.sleep(1)
            continue
            
        # Create a copy of the original frame for display
        display_frame = frame.copy()
        
        # Status bar at the top
        cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 40), (0, 0, 0), -1)
        
        # Status message
        if identified_person is None:
            status_msg = "Please look at the camera for identification"
            status_color = (0, 255, 255)  # Yellow
        elif already_voted_status:
            status_msg = f"VOTER: {identified_person} - ALREADY VOTED"
            status_color = (0, 0, 255)  # Red
        elif voting_mode:
            status_msg = f"VOTER: {identified_person} - Please cast your vote"
            status_color = (0, 255, 0)  # Green
        else:
            status_msg = "Processing identification..."
            status_color = (255, 255, 255)  # White
            
        # Display status message
        cv2.putText(display_frame, status_msg, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Perform face detection - THIS WAS MISSING
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        
        # Face detection logic
        if len(faces) == 0:
            # Prompt when no face is detected
            if not voting_mode and not already_voted_status:
                cv2.putText(display_frame, "No face detected", 
                            (display_frame.shape[1]//2 - 100, display_frame.shape[0]//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            # Only process if we haven't identified someone yet
            if identified_person is None:
                if recognition_countdown == 0:
                    recognition_time = time.time()
                    recognition_countdown = 3  # 3 seconds to recognize
                
                elapsed = time.time() - recognition_time
                remaining = max(0, recognition_countdown - elapsed)
                
                # Display countdown
                cv2.rectangle(display_frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
                cv2.putText(display_frame, f"Identifying: {remaining:.1f}s", (x, y-15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Recognition completed
                if elapsed >= recognition_countdown:
                    crop_img = frame[y:y+h, x:x+w]
                    resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
                    try:
                        output = knn.predict(resized_img)
                        confidence = knn.predict_proba(resized_img).max()
                        
                        if confidence > 0.6:  # Minimum confidence threshold
                            identified_person = output[0]
                            
                            # Check if already voted
                            voter_exist = check_if_exists(identified_person)
                            if voter_exist:
                                speak("You have already voted. Each person can vote only once.")
                                already_voted_status = True
                                # Will show the already voted window later
                            else:
                                speak(f"You are identified as {identified_person}. Please prepare to cast your vote.")
                                voting_mode = True
                        else:
                            speak("Identity verification failed. Please try again.")
                            recognition_failed = True
                            recognition_time = time.time()  # Start error display timer
                            recognition_countdown = 0
                    except Exception as e:
                        print(f"Recognition error: {e}")
                        recognition_failed = True
                        recognition_time = time.time()  # Start error display timer
                        recognition_countdown = 0
            else:
                # Display name above the face if already identified
                cv2.rectangle(display_frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
                cv2.putText(display_frame, str(identified_person), (x, y-15), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
            
        # Handle failed recognition message
        if recognition_failed:
            elapsed = time.time() - recognition_time
            if elapsed < 3:  # Show error for 3 seconds
                cv2.putText(display_frame, "IDENTIFICATION FAILED", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(display_frame, "Please try again", (50, 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                recognition_failed = False
                recognition_countdown = 0
                
        # Show already voted notification
        if already_voted_status:
            # Display "Already Voted" text
            cv2.putText(display_frame, "YOU HAVE ALREADY VOTED", 
                       (display_frame.shape[1]//2 - 200, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(display_frame, "Press ESC to exit or SPACE to see details", 
                       (display_frame.shape[1]//2 - 240, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display voting options if in voting mode
        if voting_mode:
            if not vote_instructions_given:
                speak("Please cast your vote. Press 1 for YCP, 2 for JSP, 3 for TDP, or 4 for NOTA")
                vote_instructions_given = True
            
            # Create voting panel - make it shorter to fit all options
            panel_height = 180
            cv2.rectangle(display_frame, (0, display_frame.shape[0] - panel_height), 
                         (display_frame.shape[1], display_frame.shape[0]), (0, 0, 0), -1)
            
            # Add voting instructions
            cv2.putText(display_frame, "PRESS KEY TO VOTE:", (20, display_frame.shape[0] - panel_height + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Party options with visual buttons - organize in a grid for better visibility
            button_width = 200
            button_height = 40
            button_gap = 10
            
            # First row - two buttons
            # 1: BJP
            cv2.rectangle(display_frame, 
                         (20, display_frame.shape[0] - panel_height + 50), 
                         (20 + button_width, display_frame.shape[0] - panel_height + 50 + button_height), 
                         (255, 153, 51), -1)
            cv2.putText(display_frame, "1: YCP", 
                       (20 + 20, display_frame.shape[0] - panel_height + 50 + 28), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # 2: CONGRESS
            cv2.rectangle(display_frame, 
                         (20 + button_width + button_gap, display_frame.shape[0] - panel_height + 50), 
                         (20 + 2*button_width + button_gap, display_frame.shape[0] - panel_height + 50 + button_height), 
                         (0, 102, 255), -1)
            cv2.putText(display_frame, "2: JSP", 
                       (20 + button_width + button_gap + 20, display_frame.shape[0] - panel_height + 50 + 28), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Second row - two buttons
            # 3: AAP
            cv2.rectangle(display_frame, 
                         (20, display_frame.shape[0] - panel_height + 50 + button_height + button_gap), 
                         (20 + button_width, display_frame.shape[0] - panel_height + 50 + 2*button_height + button_gap), 
                         (0, 255, 255), -1)
            cv2.putText(display_frame, "3: TDP", 
                       (20 + 20, display_frame.shape[0] - panel_height + 50 + button_height + button_gap + 28), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # 4: NOTA
            cv2.rectangle(display_frame, 
                         (20 + button_width + button_gap, display_frame.shape[0] - panel_height + 50 + button_height + button_gap), 
                         (20 + 2*button_width + button_gap, display_frame.shape[0] - panel_height + 50 + 2*button_height + button_gap), 
                         (200, 200, 200), -1)
            cv2.putText(display_frame, "4: NOTA", 
                       (20 + button_width + button_gap + 20, display_frame.shape[0] - panel_height + 50 + button_height + button_gap + 28), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Prepare frame for display on background
        try:
            # Resize to fit the background
            display_frame = cv2.resize(display_frame, (640, 449))
            
            # Add party logos on the right side of the background if needed
            if imgBackground.shape[1] >= 900:  # Only if background is wide enough
                # This is the right sidebar with party logos
                sidebar_width = 150
                
                # Get the right portion of the background image
                right_portion = imgBackground[:, imgBackground.shape[1]-sidebar_width:].copy()
                
                # Draw party logos onto this portion if needed
                # (You can add logo drawing code here if desired)
                
                # Place the display frame in the appropriate area of the background
                imgBackground[240:240 + 449, 100:100 + 640] = display_frame
            else:
                # Background image isn't wide enough, just overlay the display frame
                imgBackground[240:240 + 449, 100:100 + 640] = display_frame

            cv2.imshow('Electronic Voting System', imgBackground)
        except Exception as e:
            # Fallback if resizing/overlay fails
            cv2.imshow('Electronic Voting System', display_frame)
            print(f"Error displaying on background: {e}")

        # Process keypresses
        k = cv2.waitKey(1)
        
        # Handle exit
        if k == 27:  # ESC key
            break
        
        # Handle showing already voted details
        if already_voted_status and k == 32:  # SPACE key
            video.release()
            cv2.destroyAllWindows()
            show_already_voted_window(identified_person)
            break
            
        # Process votes if in voting mode
        if voting_mode:
            vote = None
            party_name = None
            
            if k == ord('1'):
                vote = "YCP"
                party_name = "YSR Congress Party (YCP)"
            elif k == ord('2'):
                vote = "JSP"
                party_name = "Jana Sena Party (JSP)"
            elif k == ord('3'):
                vote = "TDP"
                party_name = "Telugu Desam Party (TDP)"
            elif k == ord('4'):
                vote = "NOTA"
                party_name = "None Of The Above (NOTA)"

            if vote:
                # Get current date and time
                now = datetime.now()
                date = now.strftime("%d-%m-%Y")
                timestamp = now.strftime("%H:%M:%S")
                
                # Record vote in CSV
                with open("Votes.csv", "a", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([identified_person, vote, date, timestamp])
                
                # Clean up OpenCV windows
                video.release()
                cv2.destroyAllWindows()
                
                # Final confirmation via speech
                speak(f"Your vote for {party_name} has been recorded. Thank you for participating in the elections.")
                
                # Show receipt window
                show_receipt(identified_person, vote, date, timestamp)
                return

    # Cleanup
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Last resort error handling
        print(f"Program error: {e}")
        messagebox.showerror("Error", f"An error occurred: {str(e)}")