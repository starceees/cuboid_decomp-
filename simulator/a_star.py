import numpy as np
import math, random
import matplotlib.pyplot as plt

# same codes for OCC_VAL
PATH_VAL    = 1
WALL_VAL    = 0
UNKNOWN_VAL = -1

LOAD_FILENAME = "my_occupancy_map.npy"

def astar_occupancy(occ, start, goal):
    """
    A* on occupancy array where occ[i,j]:
      -1 => unknown => treat as free
       0 => wall => blocked
       1 => path => free
    start, goal: (i, j) grid indices
    Returns list of (i, j) or None
    """
    def heuristic(a,b):
        return math.hypot(b[0]-a[0], b[1]-a[1])
    neighbors = [(-1,0),(1,0),(0,-1),(0,1),
                 (-1,-1),(1,-1),(-1,1),(1,1)]
    import heapq
    closed_set = set()
    came_from  = {}
    gscore     = {start:0}
    fscore     = {start:heuristic(start,goal)}
    oheap      = []
    heapq.heappush(oheap,(fscore[start],start))
    
    def is_free(i,j):
        if i<0 or j<0 or i>=occ.shape[0] or j>=occ.shape[1]:
            return False
        # treat path(1) or unknown(-1) as free
        return (occ[i,j] != WALL_VAL)
    
    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            path=[]
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        closed_set.add(current)
        for di,dj in neighbors:
            ni, nj = current[0]+di, current[1]+dj
            if not is_free(ni,nj):
                continue
            tg = gscore[current] + heuristic(current,(ni,nj))
            if (ni,nj) in closed_set and tg>=gscore.get((ni,nj),float('inf')):
                continue
            if tg<gscore.get((ni,nj),float('inf')):
                came_from[(ni,nj)] = current
                gscore[(ni,nj)]    = tg
                fscore[(ni,nj)]    = tg + heuristic((ni,nj),goal)
                heapq.heappush(oheap,(fscore[(ni,nj)],(ni,nj)))
    return None

if __name__=="__main__":
    occ = np.load(LOAD_FILENAME)
    print(f"Loaded occupancy map shape={occ.shape}, dtype={occ.dtype}")

    # find all free cells => path(1) or unknown(-1)
    free_cells = np.argwhere( (occ==PATH_VAL) | (occ==UNKNOWN_VAL) )
    print(f"Number of free/unknown cells={len(free_cells)}")
    if len(free_cells)<2:
        print("Not enough free cells to do A*.")
        exit(0)
    
    start_idx = tuple(random.choice(free_cells))
    goal_idx  = tuple(random.choice(free_cells))
    print(f"Start={start_idx}, Goal={goal_idx}")
    plan = astar_occupancy(occ, start_idx, goal_idx)
    if plan is None:
        print("A* could not find path in loaded occupancy.")
    else:
        print(f"A* path length={len(plan)}")
        # For a quick debug view: show occupancy with path
        import cv2
        # map each cell to color: white(unknown), red(wall), yellow(path)
        H,W = occ.shape
        rgb = np.zeros((H,W,3), dtype=np.uint8)
        # color them
        rgb[occ==UNKNOWN_VAL] = [255,255,255]  # white
        rgb[occ==WALL_VAL]    = [255,0,0]      # red
        rgb[occ==PATH_VAL]    = [255,255,0]    # yellow
        # color the path in blue
        for (i,j) in plan:
            rgb[i,j] = [0,0,255]  # BGR => redish or we can do blue
        # show
        rgb_bgr = rgb  # we used BGR above
        cv2.imshow("Loaded Occupancy with A* Path", rgb_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
