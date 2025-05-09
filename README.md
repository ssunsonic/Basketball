# VizBuckets

Basketball + Computer Vision for tracking single player and 1v1 metrics, such as shots attempted
and scores. The current implementation assumes a fixed background with the rim, players, and ball
in frame.

UPDATES:
- 
Prior versions of this software was limited to single player tracking. Now adapted to track 2 players,
specifically 1 on 1s. The tracking is done by utilizing the SORT package, developed by Alex Bewley. 
SORT is a simple real time tracking algorithm for 2d multiple object tracking in video sequences,
and has streamlined the process of tracking multiple players in video.

### Current Caveats
- Occlusion and Re-entering object tracking issues
- "Made" class can be best described as loose fitting 
  -   Does not detect "make" class well
- Assumes all points scored are equal: (1)
  - Cannot distinguish between 2s vs 3s


## Details

This project uses Ultralytics, OpenCV, and SORT packages. The goal is to be able to accurately 
detect the movement of player(s) & ball(s) in a video, displaying the shots attempted
and made on a basketball court. The magic behind this lies in training a YOLO detection model,
which iterated over 9000+ images containing 5 core classes: [Ball, Made, Person, Shoot, Rim]. 
The model's best weights were used for real-time detection. More on how this works below:

### The Process
---
High level:

Frame looping -> YOLOv8 trained model + OpenCV overlay -> Logic for handling "made" shots and
"shot" detection -> Special logic for handling "made" detections -> SORT algorithm for multiple player tracking 

Dataset: [Roboflow][]

<br>

# Demo

Current functionality of the software! 
<br>
<br>
*Features: Single Player and 1v1 tracking, Shot Tracking, Shot Trajectory Mapping*

[![Video](https://img.youtube.com/vi/3Jv9w7-SAtk/3.jpg)](https://www.youtube.com/watch?v=3Jv9w7-SAtk)

[Roboflow]: https://universe.roboflow.com/basketball-kipnz/basketball-bs0zc-g9xgj/dataset/1

## Future Work

- Perspective transforms to track court lines, distinguishing between 2 & 3 pointers
- Pose estimation for determining common moves
  - Crossovers, hesis, behind the backs
- Solving the occlusion issue
- App development, deploying to actual users
  - Storing tracked data into a csv/database for accessible metrics

<br>

*Last Updated: 4/8/25*












