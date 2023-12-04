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