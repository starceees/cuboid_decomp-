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

## Map generation
### Find the camera rotation angle
When game start: rotate 360 to find the angles between frames, must be continuous 360; if not, should restart the game

logic: 
1. Save frames when Action.LEFT or Action.RIGHT into self.fpv_frames
2. Set the rotate_flag. 0: no totation; 1: rotate right (positive value); 2: rotate left (negative value).
3. Rotate the camera 360 degrees, count the total frames captured in 360 degrees in radian.
4. Calculate the rotation angle using self.rotate_angle = (self.frames_angle * len(self.fpv_frames)) % (2 * math.pi): Clockwise positive, counterclockwise negative.
5. Get the camera rotation angle using self.camera_angle += self.rotate_angle.


### Update the camera position
1. Save number of steps when Action.FORWARD or Action.BACKWARD into self.move_step (set each step moves 1 meter)
2. Set the initial camera position (0, 0) in x y coordinate system
3. -360 < theta < -180 = 0 < theta < 180; -180 < theta < 0  = 180 < theta < 360
4. Update camera position using polar coordinates



## Run 
```commandline
python player.py
```