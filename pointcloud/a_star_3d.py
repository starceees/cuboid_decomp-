import numpy as np
import open3d as o3d
import heapq
from tqdm import tqdm
import time
import scipy.ndimage  # for 3D morphological dilation

#############################################
# 1. Obstacle Expansion (Safety Margin)
#############################################

def expand_obstacles_3d(occupancy, safety_voxels=2):
    """
    Expands obstacles in a 3D occupancy grid by 'safety_voxels' using a morphological dilation.
    This ensures the planner keeps a safety margin from obstacles.
    
    occupancy: 3D numpy array (1=obstacle, 0=free).
    safety_voxels: how many voxels to dilate around each obstacle.
    Returns: a new 3D array with obstacles expanded.
    """
    # Create a 3D structuring element (6-connected or 26-connected).
    # For a "radius" style expansion, we can do multiple iterations or use a larger footprint.
    # Below is a simple 6-connected struct, repeated 'safety_voxels' times.
    structure = scipy.ndimage.generate_binary_structure(3, 1)  # 6-connected
    # Perform dilation
    expanded = scipy.ndimage.binary_dilation(
        occupancy.astype(bool),
        structure=structure,
        iterations=safety_voxels
    )
    return expanded.astype(np.uint8)

#############################################
# 2. A* Path Planning on the 3D Occupancy Grid
#############################################

def a_star_3d(occupancy, start, goal):
    """
    Performs A* search on a 3D occupancy grid.
    occupancy: 3D numpy array with obstacles=1, free=0.
    start, goal: (ix, iy, iz) grid indices in the occupancy.
    Returns a list of grid indices from start to goal (inclusive), or None if no path is found.
    """
    dims = occupancy.shape
    
    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))
    
    # 26-connected neighbors
    neighbors = [(dx, dy, dz)
                 for dx in (-1, 0, 1)
                 for dy in (-1, 0, 1)
                 for dz in (-1, 0, 1)
                 if not (dx == 0 and dy == 0 and dz == 0)]
    
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    closed_set = set()
    
    pbar = tqdm(desc="A* Processing", unit="iter")
    while open_set:
        _, current_g, current = heapq.heappop(open_set)
        pbar.update(1)
        
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            pbar.close()
            return path[::-1]
        
        closed_set.add(current)
        cx, cy, cz = current
        for dx, dy, dz in neighbors:
            nx = cx + dx
            ny = cy + dy
            nz = cz + dz
            if not (0 <= nx < dims[0] and 0 <= ny < dims[1] and 0 <= nz < dims[2]):
                continue
            if occupancy[nx, ny, nz] == 1:
                continue
            if (nx, ny, nz) in closed_set:
                continue
            cost = np.linalg.norm([dx, dy, dz])
            tentative_g = current_g + cost
            if (nx, ny, nz) not in g_score or tentative_g < g_score[(nx, ny, nz)]:
                came_from[(nx, ny, nz)] = current
                g_score[(nx, ny, nz)] = tentative_g
                f_score = tentative_g + heuristic((nx, ny, nz), goal)
                heapq.heappush(open_set, (f_score, tentative_g, (nx, ny, nz)))
    pbar.close()
    return None

#############################################
# 3. Grid-to-World Conversion
#############################################

def grid_to_world_3d(idx, global_min, resolution):
    """
    Convert grid indices (ix, iy, iz) to world coords (x, y, z).
    """
    return np.array([
        global_min[0] + (idx[0] + 0.5) * resolution,
        global_min[1] + (idx[1] + 0.5) * resolution,
        global_min[2] + (idx[2] + 0.5) * resolution
    ])

#############################################
# 4. Open3D Animation with Thicker Path
#############################################

def animate_path_open3d(path_world, obstacles_pc, sleep_time=0.1, line_width=5.0):
    """
    Animates the path in Open3D, showing:
      - A blue sphere (robot) moving along 'path_world',
      - A red point cloud 'obstacles_pc' for obstacles,
      - A green line for the path, with adjustable 'line_width'.
    """
    # Blue sphere for robot
    robot = o3d.geometry.TriangleMesh.create_sphere(radius=5)
    robot.compute_vertex_normals()
    robot.paint_uniform_color([0, 0, 1])
    robot.translate(path_world[0], relative=False)
    
    # Green line for path
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(path_world)
    lines = [[i, i+1] for i in range(len(path_world)-1)]
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in lines])
    
    # Coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    
    # Create Open3D window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Path Animation", width=800, height=600)
    # Increase line width for better visibility
    vis.get_render_option().line_width = line_width
    
    vis.add_geometry(frame)
    vis.add_geometry(obstacles_pc)
    vis.add_geometry(line_set)
    vis.add_geometry(robot)
    
    path_world = [np.array(p) for p in path_world]
    counter = [0]
    
    def animation_callback(vis):
        if counter[0] < len(path_world):
            new_pos = path_world[counter[0]]
            current_center = np.array(robot.get_center())
            translation = new_pos - current_center
            robot.translate(translation, relative=True)
            vis.update_geometry(robot)
            counter[0] += 1
            time.sleep(sleep_time)
        return False
    
    vis.register_animation_callback(animation_callback)
    vis.run()
    vis.destroy_window()

#############################################
# 5. Main Routine
#############################################

if __name__ == "__main__":
    # 1) Load your existing occupancy grid
    occupancy = np.load("occupancy_3d.npy")  # 3D array, 1=obs,0=free
    # 2) Expand obstacles for safety
    from scipy.ndimage import binary_dilation, generate_binary_structure
    safety_voxels = 2  # e.g. expand by 2 voxels
    occupancy_safety = expand_obstacles_3d(occupancy, safety_voxels=safety_voxels)
    
    # 3) Provide the bounding box info used to build occupancy_3d
    #    so that grid_to_world_3d lines up. Must match the code that built occupancy_3d.
    global_min = np.array([-48.46652245, -27.08304417, -4.99])  # Example
    resolution = 0.2
    
    # 4) Randomly pick start & goal from free cells in the expanded grid
    free_cells = np.argwhere(occupancy_safety == 0)
    if free_cells.size == 0:
        print("No free space found in safety occupancy!")
        exit()
    np.random.shuffle(free_cells)
    start_idx = tuple(free_cells[0])
    goal_idx  = tuple(free_cells[1])
    print("Random start:", start_idx, "Random goal:", goal_idx)
    
    # 5) A* planning on the expanded grid
    start_time = time.time()
    path_indices = a_star_3d(occupancy_safety, start_idx, goal_idx)
    end_time = time.time()
    print(f"A* search took {end_time - start_time:.2f} seconds.")
    if path_indices is None:
        print("No path found.")
        exit(0)
    print("Path length:", len(path_indices))
    
    # 6) Convert path to world coords
    path_world = [grid_to_world_3d(idx, global_min, resolution) for idx in path_indices]
    
    # 7) Create a red point cloud for obstacles from the *expanded* occupancy grid
    occ_indices = np.argwhere(occupancy_safety == 1)
    max_obstacles = 10000
    if occ_indices.shape[0] > max_obstacles:
        factor = int(np.ceil(occ_indices.shape[0] / max_obstacles))
        occ_indices = occ_indices[::factor]
    occ_points = np.array([grid_to_world_3d(tuple(idx), global_min, resolution) for idx in occ_indices])
    obstacles_pc = o3d.geometry.PointCloud()
    obstacles_pc.points = o3d.utility.Vector3dVector(occ_points)
    obstacles_pc.paint_uniform_color([1, 0, 0])
    
    # 8) Animate the path with thicker lines in Open3D
    animate_path_open3d(path_world, obstacles_pc, sleep_time=0.1, line_width=5.0)
