import cv2
import pytesseract
from PIL import Image
import time

def extract_text_from_frame(frame):
    # Convert the frame to an image
    img = Image.fromarray(frame)
    # Use pytesseract to extract text
    text = pytesseract.image_to_string(img)
    return text

def extract_metadata(filepath):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise ValueError(f"Error: Cannot open video {filepath}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / frame_rate

    print(f"Processing video: {filepath}")
    print(f"Total frames: {frame_count}, Frame rate: {frame_rate}, Duration: {duration} seconds")

    text_data = ""
    success, frame = cap.read()
    count = 0
    start_time = time.time()
    while success:
        # Extract text from the frame
        text_data += extract_text_from_frame(frame)
        count += 1
        if count % 10 == 0:  # Print progress every 10 frames
            elapsed_time = time.time() - start_time
            print(f"Processed {count} frames in {elapsed_time:.2f} seconds")
        success, frame = cap.read()

    cap.release()
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")

    # Parse the text data to extract camera ID and other information
    camera_id = "Unknown"
    for line in text_data.split("\n"):
        if "Camera ID:" in line:
            camera_id = line.split("Camera ID:")[1].strip()
            break

    return duration, camera_id

# Test with your video file
video_path = "uploads/DSCF0089.mov"
extract_metadata(video_path)
