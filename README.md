# Visual Navigation Game (Example Player Code)

This is the course project platform for NYU ROB-GY 6203 Robot Perception. 
For more information, please reach out to AI4CE lab (cfeng at nyu dot edu).

# Instructions for Players
1. Install
```commandline
conda update conda
git clone https://github.com/ai4ce/vis_nav_player.git
cd vis_nav_player
conda env create -f environment.yaml
conda activate game
```

2. Play using the default keyboard player
```commandline
python player.py
```

3. Modify the player.py to implement your own solutions, 
unless you have photographic memories!

# Implementation

## Depth Estimation 
Reference: https://github.com/noahzn/Lite-Mono

### Install
```commandline
git clone https://github.com/noahzn/Lite-Mono.git
```

1. [Download](https://surfdrive.surf.nl/files/index.php/s/CUjiK221EFLyXDY) lite-Mono weights
2. Modify the path in config in [player.py](./player.py) line 13
3. Modify the system path of Lite-Mono in [test_simple_modified.py](./test_simple_modified.py) line 5


## Run 
```commandline
python player.py
```

## Implement
### Find the camera rotation angle
When game start: rotate 360 to find the angles between frames, must be continuous 360; if not, should restart the game

logic: 
1. Set the rotate_flag. 0: no totation; 1: rotate right (positive value); 2: rotate left (negative value).
2. Rotate the camera 360 degrees, count the total frames captured in 360 degrees in radian.
3. Calculate the rotation angle using self.rotate_angle = (self.frames_angle * len(self.fpv_frames)) % (2 * math.pi): Clockwise positive, counterclockwise negative.
4. Get the camera rotation angle using self.camera_angle += self.rotate_angle.


### Update the camera position
1. Set the initial camera position (0, 0) in x y coordinate system
2. -360 < theta < -180 = 0 < theta < 180; -180 < theta < 0  = 180 < theta < 360


### if it is close to wall
1. If there is a path, the mean depth in the central region of the depth map should larger than 0.9
2. Try not to hit the camera back against the wall
