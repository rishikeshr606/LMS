import cv2
import os

def extract_frames(video_path, output_folder, frames_per_second):
    """Extract frames from video at given FPS."""
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 1
    frames_per_second = min(frames_per_second, fps)
    interval = round(fps / frames_per_second)

    count = saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved += 1
        count += 1
    cap.release()
    print(f"Extracted {saved} frames to {output_folder}")
