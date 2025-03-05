import os
import cv2

def extract_frames(video_path, output_dir, frame_rate=1):
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.basename(video_path).split('.')[0]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count = 0
    success = True
    while success:
        success, frame = cap.read()
        if count % frame_rate == 0 and success:
            frame_filename = os.path.join(output_dir, f"{video_name}_frame_{count}.jpg")
            cv2.imwrite(frame_filename, frame)
        count += 1

    cap.release()
    print(f"Frames extracted to {output_dir}")

def process_directory(base_dir, animal_classes):
    for animal in animal_classes:
        animal_dir = os.path.join(base_dir, animal)
        if not os.path.exists(animal_dir):
            print(f"Directory {animal_dir} does not exist.")
            continue
        print(f"Checking directory: {animal_dir}")
        print(f"Contents: {os.listdir(animal_dir)}")
        video_files = [f for f in os.listdir(animal_dir) if f.lower().endswith(('mp4', 'avi', 'mov'))]
        if not video_files:
            print(f"No video files found in {animal_dir}")
        for video in video_files:
            video_path = os.path.join(animal_dir, video)
            output_dir = animal_dir  # Output frames to the same directory
            extract_frames(video_path, output_dir)

if __name__ == "__main__":
    data_dir = "/Users/josiahericksen/PycharmProjects/Trailtracker/data"  # Set this to the correct path

    animal_classes = ["Bear", "Turkey", "Boar", "Bobcat", "Deer", "Unidentified"]

    # Process train directory
    train_dir = os.path.join(data_dir, 'train')
    if os.path.exists(train_dir):
        print("Processing train directory")
        process_directory(train_dir, animal_classes)
    else:
        print("Train directory does not exist")

    # Process val directory
    val_dir = os.path.join(data_dir, 'val')
    if os.path.exists(val_dir):
        print("Processing val directory")
        process_directory(val_dir, animal_classes)
    else:
        print("Val directory does not exist")
