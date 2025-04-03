# VizBuckets

This is an implementation of basketball + computer vision for tracking single player shots.
The current implementation assumes a fixed background with a single player shooting around
the court.

FUTURE USE: 
- Track multiple players
- Keep score for various formats (1v1s, 5v5s)
- Shot angles, areas of improvement

<br>

## Details

This project uses Ultralytics, OpenCV, and PyTorch and is able to accurately 
detect the movement of player(s) & ball(s) in a video, displaying the shots attempted
and made on a basketball court. The magic behind this lies in training a YOLO detection model,
which iterated over 9000+ images containing 5 core classes: [Ball, Made, Person, Shoot, Rim]. 
The model's best weights were used for real-time detection. 

Dataset: [Roboflow][]


## Demo

Current functionality of the software. 
<br>
<br>
*Features: Single Player tracking, Shot Tracking, Shot Trajectory Mapping*

![Video](assets/shootaround.gif)


[Roboflow]: https://universe.roboflow.com/basketball-kipnz/basketball-bs0zc-g9xgj/dataset/1


## Installation

### Pixi

For this project, I used [pixi][]: a fast package manager based on the
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












