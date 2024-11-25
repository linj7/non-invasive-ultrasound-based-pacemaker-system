import os
import cv2
import csv
import sys
import numpy as np
import pandas as pd

def resize_with_aspect_ratio(frame, width, height):
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    output_frame = np.zeros((height, width, 3), dtype=np.uint8)
    x_offset = (width - new_w) // 2
    y_offset = (height - new_h) // 2
    output_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
    return output_frame

def resize_video(input_file, width=112, height=112):
    cap = cv2.VideoCapture(input_file)
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    file_path_without_extension = os.path.splitext(input_file)[0]
    output_file = file_path_without_extension + ".avi"
    
    if not cap.isOpened():
        print(f"Can't open {input_file}.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 50.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {frame_count}")
    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = resize_with_aspect_ratio(frame, width, height)
        out.write(resized_frame)

        current_frame += 1
        
    cap.release()
    out.release()
    os.remove(input_file)
    print(f"{file_name} has resized to {width}x{height}.")
    return output_file

def adjust_video_fps(video_path, target_fps=50):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Unable to open the video file to adjust fps.")
        return
    
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    temp_output_path = "temp_adjusted_video.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_output_path, fourcc, target_fps, (width, height))
    
    if input_fps > target_fps:
        # Downsampling: Save only one frame every (input_fps / target_fps) frames.
        frame_interval = int(input_fps / target_fps)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Save only one frame at the specified interval
            if frame_count % frame_interval == 0:
                out.write(frame)
            frame_count += 1

    else:
        # Frame interpolation: Interpolate frames to reach the target frame rate
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        num_original_frames = len(frames)
        new_total_frames = int(num_original_frames * (target_fps / input_fps))
        
        # Perform linear interpolation on the original frames
        for i in range(new_total_frames):
            original_index = i * (num_original_frames - 1) / (new_total_frames - 1)
            lower_index = int(np.floor(original_index))
            upper_index = min(int(np.ceil(original_index)), num_original_frames - 1)
            alpha = original_index - lower_index
            
            # Interpolate to generate new frames
            interpolated_frame = cv2.addWeighted(frames[lower_index], 1 - alpha, frames[upper_index], alpha, 0)
            out.write(interpolated_frame)

    cap.release()
    out.release()
    os.remove(video_path)
    os.rename(temp_output_path, video_path)
    print(f"Video has been adjusted to {target_fps} fps and saved as '{video_path}'.")

def append_to_csv(video_path, filelist_path, volumetracings_path):
    output_file = resize_video(video_path) 
    adjust_video_fps(output_file) 
    cap = cv2.VideoCapture(output_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release() 
    
    file_name_no_ext = os.path.splitext(os.path.basename(video_path))[0] # Get the file name (excluding the extension and path)
    file_name_with_ext = file_name_no_ext + ".avi" # Get the file name (including the extension, but excluding the path)
    
    # Data wrote to filelist_path
    row = [file_name_no_ext, 35, 35, 35, 112, 112, 50, total_frames, "TEST"]
    
    # Write row data to the csv file in filelist_path
    with open(filelist_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

    # Write row data to the csv file in volumetracings_path
    with open(volumetracings_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for _ in range(21): # first 21 rows
            row = [file_name_with_ext, 60, 50, 80, 50, 1]
            writer.writerow(row)
        for _ in range(21): # last 21 rows
            row = [file_name_with_ext, 55, 45, 75, 45, total_frames - 2]
            writer.writerow(row)

    result_str = f"Successfully written to {filelist_path} and {volumetracings_path}!"
    print(result_str)
    return result_str

if __name__ == "__main__":
    video_path = sys.argv[1]
    filelist_path = sys.argv[2]
    volumetracings_path = sys.argv[3]
    append_to_csv(video_path, filelist_path, volumetracings_path)


