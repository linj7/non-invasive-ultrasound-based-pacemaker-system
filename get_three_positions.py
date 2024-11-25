import os
import csv
import sys
import cv2
import json
import numpy as np

def crop_model_output_top_right_corner(input_video_path, output_video_path, 
                                frame_width=224, frame_height=224, 
                                crop_width=112, crop_height=112):
    """
    Crop the top-right corner region of each frame in the model output video file and save it as a new video file.
    
    :param input_video_path: Path to the input video file.
    :param output_video_path: Path to the output video file.
    :param frame_width: Width of the input video frames (default is 224).
    :param frame_height: Height of the input video frames (default is 224).
    :param crop_width: Width of the cropped region (default is 112).
    :param crop_height: Height of the cropped region (default is 56).
    """
    
    if not os.path.exists(input_video_path):
        # print(f"'{input_video_path}' doesn't exist.")
        return
    
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        # print(f"Unable to open '{input_video_path}'.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 50  # Default fps
        print("Warning：Unable to get the frame rate of the video, use 50 fps as default. ")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(f"Total frames: {total_frames}")
    
    # Set the encoder for the output video: for .avi files, use the 'XVID' encoder
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    cropped_size = (crop_width, crop_height)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, cropped_size)
    
    if not out.isOpened():
        # print(f"Unable to create video file '{output_video_path}'.")
        cap.release()
        return
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Check if the frame dimensions meet the requirements
        if frame.shape[1] != frame_width or frame.shape[0] != frame_height:
            frame = cv2.resize(frame, (frame_width, frame_height))
        
        x_start = frame_width - crop_width
        y_start = 0
        x_end = frame_width
        y_end = crop_height
        
        cropped_frame = frame[y_start:y_end, x_start:x_end]
        out.write(cropped_frame)
        frame_count += 1

    cap.release()
    out.release()
    # print(f"Save cropped video to '{output_video_path}'.")

def calculate_three_positions(input_video_path, output_video_path):
    if not os.path.exists(input_video_path):
        # print(f"Input file doesn't exist: {input_video_path}")
        exit()
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Can't open file.")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS)  
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    frame_size = (frame_width, frame_height)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    frame_count = 0  
    last_frame_markers = [] 

    while True:
        ret, frame = cap.read()  
        if not ret:
            break  

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)

        red_mask = mask1 | mask2
        ys = np.where(red_mask.any(axis=1))[0]

        if ys.size == 0:
            annotated_frame = frame.copy()
            out.write(annotated_frame)
            frame_count += 1
            continue

        min_y = ys.min()
        max_y = ys.max()
        three_quarter_height = int(min_y + (max_y - min_y) / 4)
        markers = []

        # Mark Position 2: at halfway of the left boundary.
        half_height = frame_height // 2
        target_point_2 = next(((min_x, y) for y in range(half_height - 5, half_height + 5)
                               if (x_indices := np.where(red_mask[y] != 0)[0]).size > 0
                               for min_x in [x_indices.min()]), None)

        if target_point_2:
            markers.append(('2', target_point_2))

        # Mark Position 3: at the lowest point of the left boundary
        left_boundary_points = [(np.where(red_mask[y] != 0)[0].min(), y)
                                for y in range(frame_height) if np.where(red_mask[y] != 0)[0].size > 0]
        target_point_3 = max(left_boundary_points, key=lambda pt: pt[1], default=None)
        if target_point_3:
            markers.append(('3', target_point_3))

        # Mark position 4: three-quarters up the right boundary
        target_point_4 = next(((max_x, y) for y in range(three_quarter_height - 5, three_quarter_height + 5)
                               if (x_indices := np.where(red_mask[y] != 0)[0]).size > 0
                               for max_x in [x_indices.max()]), None)

        if target_point_4:
            markers.append(('4', target_point_4))

        last_frame_markers = markers if frame_count == total_frames - 1 else last_frame_markers
        annotated_frame = frame.copy()

        for label, point in markers:
            cv2.circle(annotated_frame, point, radius=13, color=(10, 255, 0), thickness=-1)
            cv2.putText(annotated_frame, label, (point[0] - 8, point[1] + 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            coordinate_text = f"({point[0]}, {point[1]})"
            cv2.putText(annotated_frame, coordinate_text, (point[0] + 15, point[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)

        out.write(annotated_frame)

        frame_count += 1

    cap.release()
    out.release()

    # print(f"Save annotated video to {output_video_path}.")

    # Return last frame's coordinate of three positions
    return {label: point for label, point in last_frame_markers}

def main():
    input_video_path = sys.argv[1]
    cropped_video_path = os.path.splitext(input_video_path)[0] + "_cropped" + os.path.splitext(input_video_path)[1]
    annotated_video_path = os.path.splitext(input_video_path)[0] + "_annotated" + os.path.splitext(input_video_path)[1]

    crop_model_output_top_right_corner(input_video_path, cropped_video_path)
    result = calculate_three_positions(cropped_video_path, annotated_video_path)
    result = {key: (int(value[0]), int(value[1])) for key, value in result.items()}
    print(json.dumps(result))

if __name__ == "__main__":
    main()


