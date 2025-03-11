# Occupancy Grid Mapping and A* Path Planning

This project demonstrates a simple approach to build a 2D occupancy grid from robot pose data recorded in a simulated environment and then run A* path planning on the generated map.

## Overview

The project consists of two main phases:

1. **Mapping Phase:**  
   - The robot records its poses (skipping pure turning) using a `vis_nav_game` simulation.
   - A fine occupancy grid is built from the recorded path data.  
     *Cells are represented internally as:*  
     - **-1:** Unknown (displayed as white)  
     - **0:** Wall (blocked, displayed as red)  
     - **1:** Path (traversable, displayed as yellow)
   - The fine grid is merged to a coarser resolution.
   - Morphological dilation is applied to widen the traversable corridors.
   - The final occupancy grid is saved as a `.npy` file.

2. **Planning Phase:**  
   - A separate script loads the saved occupancy grid.
   - A* path planning is performed on the grid (treating both path and unknown as free, and walls as obstacles).
   - The planned path is displayed over the occupancy grid.

## Requirements

- Python 3.7+
- [numpy](https://numpy.org/)
- [opencv-python](https://pypi.org/project/opencv-python/)
- [matplotlib](https://matplotlib.org/)
- [scikit-image](https://scikit-image.org/)
- [pygame](https://www.pygame.org/)
- [tqdm](https://github.com/tqdm/tqdm)
- [vis_nav_game](https://pypi.org/project/vis-nav-game/)

Install dependencies with:

```bash
pip install numpy opencv-python matplotlib scikit-image pygame tqdm vis_nav_game
```
