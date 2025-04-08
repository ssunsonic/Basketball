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

## Demo

Current functionality of the software! 
<br>
<br>
*Features: Single Player and 1v1 tracking, Shot Tracking, Shot Trajectory Mapping*

[![Video](https://img.youtube.com/vi/3Jv9w7-SAtk/3.jpg)](https://www.youtube.com/watch?v=3Jv9w7-SAtk)

[Roboflow]: https://universe.roboflow.com/basketball-kipnz/basketball-bs0zc-g9xgj/dataset/1


## Installation

### Pixi

If you would like to recreate this project, please download 
[pixi][]: a fast package manager based on the
conda ecosystem. If you prefer not to use pixi,
it's also possible to manually install the packages using conda or mamba.

[pixi]: https://pixi.sh/

The `pixi.toml` file in this repo lists required packages, while the
`pixi.lock` file lists package versions for each platform. When the lock file
is present, pixi will attempt to install the exact versions listed. Deleting
the lock file allows pixi to install other versions, which might help if
installation fails (but beware of inconsistencies between package versions).

To install the required packages, open a terminal and navigate to this repo's
directory. Then run:

```sh
pixi install
```

This will automatically create a virtual environment and install the packages.

To open a shell in the virtual environment, run:

```sh
pixi shell
```

You can run the `pixi shell` command from the repo directory or any of its
subdirectories. Use the virtual environment to run any commands related to this
repo. When you're finished using the virtual environment, you can use the
`exit` command to exit the shell.

<br>

*Last Updated: 4/8/25*












