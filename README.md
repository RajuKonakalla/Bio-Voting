# Bio-Voting System

## Overview

This project implements a secure electronic voting system using facial recognition technology. The system captures facial data of voters, identifies them via webcam, and allows them to cast their vote securely. It ensures that each voter can vote only once and provides a receipt for verification.

## Features

- Facial recognition-based voter identification using OpenCV and K-Nearest Neighbors (KNN) classifier.
- Webcam integration with robust initialization and error handling.
- GUI windows for vote confirmation and already voted notifications using Tkinter.
- Vote recording in CSV format with date and time stamps.
- Speech feedback using Windows SAPI for user interaction.
- Background image overlay for a polished user interface.
- Detailed logging of facial recognition processes.

## Components

### 1. Facial Data Capture (`add_face.py`)

- Captures multiple frames of the voter's face using the webcam.
- Uses Haar Cascade classifier for face detection.
- Saves facial data and associated user IDs in pickle files for training.
- Implements robust webcam initialization with retries and error handling.
- Provides a progress bar and user instructions during capture.

### 2. Voting Process (`give_vote.py`)

- Initializes webcam with retries to ensure it opens directly.
- Loads trained facial data and trains a KNN classifier.
- Detects faces in real-time and identifies voters.
- Checks if the voter has already voted; if yes, shows a notification window.
- Allows voting for predefined parties or NOTA.
- Records votes in a CSV file with timestamps.
- Provides speech feedback and graphical receipt after voting.
- Displays voting options and status messages on the video feed.

## How to Use

1. **Register Faces:**
   - Run `add_face.py` to capture facial data for each voter.
   - Follow on-screen instructions to complete registration.

2. **Start Voting:**
   - Run `give_vote.py` to start the voting system.
   - The system will open the webcam and identify voters.
   - Voters can cast their vote using keyboard inputs.
   - Receipts and confirmations are provided after voting.

## Dependencies

- Python 3.x
- OpenCV (`cv2`)
- scikit-learn
- numpy
- pandas
- tkinter (usually included with Python)
- pywin32 (for Windows speech API)
- Other standard Python libraries

## Notes

- Ensure the webcam is connected and accessible.
- Training data files (`names.pkl` and `faces_data.pkl`) must be present before starting the voting process.
- Votes are stored in `Votes.csv`.
- Background images (`background.png`) enhance the UI but are optional.

## Troubleshooting

- If the webcam does not open, the system attempts multiple retries.
- Error messages and logs are provided for troubleshooting.
- For any issues with face recognition, ensure proper lighting and clear face visibility.

## License

This project is open source and available under the MIT License.

## Author

Rajukonkalla
