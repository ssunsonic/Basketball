"""
This is a script that utilizes YOLO object detection and OpenCV to detect basketball shots and makes.

It uses a pretrained YOLO model to detect the classes 'shoot', 'made', 'rim', and 'ball'. 
"""

# Imports
import time
import math
import ipdb

from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque

from src.utils.distances import is_increasing_distances
from src.utils.output import get_available_filename
from src.utils.trajectory import draw_shot_arc
from src.utils.court_transform import detect_court_lines, detect_three_point_line

# Load the trained model
model = YOLO("models/bballvision.pt")
labels = model.names # Classes in dataset

# Initialize the video capture object
# cap = cv2.VideoCapture("videos/vid1.mp4")
cap = cv2.VideoCapture("videos/frankie.mp4")
# cap = cv2.VideoCapture(0)

# Stuff for output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = get_available_filename('output_vids', 'output', 'mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Variables
shot_counter = 0
shots_made = 0
FRAME_MAKE_COOLDOWN = 30
FRAME_SINCE_LAST_MAKE = 0

# Positions
ball_position = deque(maxlen=30)    
rim_position = deque(maxlen=30)
shot_position = deque(maxlen=30)

# Mask and Frames
ball_mask= None
img = 0

# While video not terminated
while cap.isOpened():
    # ipdb.set_trace()
    ret, frame = cap.read()
    if not ret:
          break
    
    # For portrait videos
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # YOLO inference
    results = model(frame, stream=False)

    # Initialize frame counter
    FRAME_SINCE_LAST_MAKE += 1
    print(FRAME_SINCE_LAST_MAKE)

    # For counting makes/attempts
    # shot_detected = False
    shot_made = False

    # Heatmap
    heatmap = np.zeros_like(frame, dtype=np.float32)

    # Check if label 'shoot' is detected
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1 # Width and height of bounding box

            # Get confidence score
            conf = math.ceil(box.conf[0] * 100) / 100

            # Class Names
            cls = int(box.cls[0])
            current_class = labels.get(cls)

            # Get centered coordinates
            cx, cy = x1+w // 2, y1+h // 2

            # Draw rectangle
            cv2.rectangle(frame, (5, 0), (315, 150), (255, 255, 255), -1) # White rectangle

            # Display text
            cv2.putText(frame, f"{box.conf[0]:.2f}", (max(0, x1), max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2) # Confidence score
            cv2.putText(frame, f"{current_class}", (max(0, x1), max(0, y1-30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2) # Class label; 5 classes
            cv2.putText(frame, f"Shot Attempted: {shot_counter}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3) # Shots attempted counter            
            cv2.putText(frame, f"Shots Made: {shots_made}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3) # Shots made counter

            # Logic for class handling
            if current_class == "shoot" and conf > 0.8:
                shot_position.append([cx, cy, img]) # Append shot position to deque

            if current_class == "made":
                shot_made = True       

            if current_class == "rim" and conf > 0.70:
                y1 += 20
                rim_position.append([x1, x2, y1, y2, img]) # Append rim position to deque
                # cv2.line(frame, (x1, y1), (x2, y1), (0, 0, 255), 2) # Draw line               
            
            if current_class == "ball" and conf > 0.5:
                 ball_position.append([cx, cy, img]) # Append ball position to deque
                #  cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1) # Draw circle


    # Checks if distance from shoot position and ball keeps increasing after shot attempt
    # Checks if last time "shoot" was detected was five frames ago
    if shot_position and shot_position[-1][2] == img - 5:
        last_ball_pos = [(cx, cy) for cx, cy, img in list(ball_position)[-5:]]
        if is_increasing_distances((shot_position[-1][0], shot_position[-1][1]), last_ball_pos):
            shot_counter += 1

    # If a make is detected, check if cooldown has passed
    if shot_made and FRAME_SINCE_LAST_MAKE > FRAME_MAKE_COOLDOWN:
        shots_made += 1
        FRAME_SINCE_LAST_MAKE = 0

    # Adds circles on ball position every 7 frames
    if ball_mask is None:
        ball_mask = np.zeros_like(frame, dtype=np.uint8)

    # Draws a path for the balls
    if img % 2 == 0:
        # Clear the overlay (reset to transparent)
        ball_mask = np.zeros_like(frame, dtype=np.uint8)
        
        for pos in ball_position:
            cx, cy, pos_frame = pos
            if pos_frame % 2 == 0:
                cv2.circle(ball_mask, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
    
    img += 1

    # Draw trajectory
    # draw_trajectory(list(ball_position)[-10:], frame)

    # Shooting PCT%
    if shot_counter > 0:
        shooting_pct = shots_made / shot_counter
        cv2.putText(frame, f"Shooting PCT: {100 * shooting_pct:.2f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    else:
        cv2.putText(frame, f"Shooting PCT: 0%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    

    # HEATMAP
    # Update heatmap based on ball positions
    # for x, y, _ in ball_position:
    #     cv2.circle(heatmap, (x, y), 15, (255,), -1)  # Larger radius for more impact

    # Normalize and overlay heatmap
    # heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)
    # heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # cv2.addWeighted(frame, 0.7, heatmap_color, 0.3, 0, frame)

    # Draw shot arc
    # draw_shot_arc(frame, ball_position)

    # Blend the overlay onto the main frame
    blended_img = cv2.addWeighted(frame, 1, ball_mask, 0.75, 0.5)
    cv2.imshow("YOLOv8 Detection", blended_img)

    # Write to output video
    out.write(blended_img)

    if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    
# Sanity
print(f"Shots attempted: {shot_counter}")
print(f"Shots made: {shots_made}")

cap.release()
out.release()
cv2.destroyAllWindows()

