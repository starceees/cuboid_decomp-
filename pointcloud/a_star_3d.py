import numpy as np
import heapq
from tqdm import tqdm
import time
import open3d as o3d

#############################################
# 3D A* Path Planning (using occupancy grid)
#############################################

def a_star_3d(occupancy, start, goal):
    """
    Performs A* search on a 3D occupancy grid.
    occupancy: 3D numpy array with obstacles=1, free=0.
    start, goal: tuple indices (i,j,k) in the occupancy grid.
    Returns a list of grid indices from start to goal (inclusive), or None if no path is found.
    """
    dims = occupancy.shape
    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))
    
    # 26-connected neighborhood.
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
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            pbar.close()
            return path[::-1]
        closed_set.add(current)
        for dx, dy, dz in neighbors:
            neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
            if not (0 <= neighbor[0] < dims[0] and 0 <= neighbor[1] < dims[1] and 0 <= neighbor[2] < dims[2]):
                continue
            if occupancy[neighbor] == 1:
                continue  # Obstacle
            if neighbor in closed_set:
                continue
            cost = np.linalg.norm([dx, dy, dz])
            tentative_g = current_g + cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))
    pbar.close()
    return None

#############################################
# Helper: Grid-to-World conversion
#############################################

def grid_to_world_3d(idx, global_min, resolution):
    """
    Convert grid indices to world coordinates (center of voxel).
    """
    return np.array([
        global_min[0] + (idx[0] + 0.5) * resolution,
        global_min[1] + (idx[1] + 0.5) * resolution,
        global_min[2] + (idx[2] + 0.5) * resolution
    ])

#############################################
# Part: Animate Path with Open3D Visualizer (with Obstacles)
#############################################

def animate_path_open3d(path_world, obstacles_pc, sleep_time=0.1):
    """
    Animates the given path (a list of world-coordinate points) using Open3D.
    Displays a blue sphere (robot) moving along the path and a red point cloud for obstacles.
    """
    # Create a blue sphere to represent the robot.
    robot = o3d.geometry.TriangleMesh.create_sphere(radius=5)
    robot.compute_vertex_normals()
    robot.paint_uniform_color([0, 0, 1])
    robot.translate(path_world[0], relative=False)
    
    # Create a coordinate frame for reference.
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Path Animation", width=800, height=600)
    vis.add_geometry(frame)
    vis.add_geometry(obstacles_pc)
    vis.add_geometry(robot)
    
    # Convert path_world to a list of numpy arrays.
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
# Main Routine
#############################################

if __name__ == "__main__":
    # Load occupancy grid (generated from your point cloud processing) from file.
    occupancy = np.load('occupancy_3d.npy')
    # Recompute (or define) global_min from your occupancy grid generation.
    # Replace the value below with the correct global_min for your current occupancy grid.
    global_min = np.array([-48.46652245, -27.08304417, -4.99])
    resolution = 0.2  # Must match the occupancy grid resolution.
    
    # Randomly select two free cells as start and goal.
    free_cells = np.argwhere(occupancy == 0)
    if free_cells.size == 0:
        print("No free space found in the occupancy grid!")
        exit()
    np.random.shuffle(free_cells)
    start_idx = tuple(free_cells[0])
    goal_idx = tuple(free_cells[1])
    print("Random Grid start:", start_idx, "Random Grid goal:", goal_idx)
    
    start_time = time.time()
    path_indices = a_star_3d(occupancy, start_idx, goal_idx)
    end_time = time.time()import numpy as np
import heapq
from tqdm import tqdm
import time
import open3d as o3d

#############################################
# 3D A* Path Planning (using occupancy grid)
#############################################

def a_star_3d(occupancy, start, goal):
    """
    Performs A* search on a 3D occupancy grid.
    occupancy: 3D numpy array with obstacles=1, free=0.
    start, goal: tuple indices (i,j,k) in the occupancy grid.
    Returns a list of grid indices from start to goal (inclusive), or None if no path is found.
    """
    dims = occupancy.shape
    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))
    
    # 26-connected neighborhood.
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
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            pbar.close()
            return path[::-1]
        closed_set.add(current)
        for dx, dy, dz in neighbors:
            neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
            if not (0 <= neighbor[0] < dims[0] and 0 <= neighbor[1] < dims[1] and 0 <= neighbor[2] < dims[2]):
                continue
            if occupancy[neighbor] == 1:
                continue  # Obstacle
            if neighbor in closed_set:
                continue
            cost = np.linalg.norm([dx, dy, dz])
            tentative_g = current_g + cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))
    pbar.close()
    return None

#############################################
# Helper: Grid-to-World conversion
#############################################

def grid_to_world_3d(idx, global_min, resolution):
    """
    Convert grid indices to world coordinates (center of voxel).
    """
    return np.array([
        global_min[0] + (idx[0] + 0.5) * resolution,
        global_min[1] + (idx[1] + 0.5) * resolution,
        global_min[2] + (idx[2] + 0.5) * resolution
    ])

#############################################
# Part: Animate Path with Open3D Visualizer (with Obstacles)
#############################################

def animate_path_open3d(path_world, obstacles_pc, sleep_time=0.1):
    """
    Animates the given path (a list of world-coordinate points) using Open3D.
    Displays a blue sphere (robot) moving along the path and a red point cloud for obstacles.
    """
    # Create a blue sphere to represent the robot.
    robot = o3d.geometry.TriangleMesh.create_sphere(radius=5)
    robot.compute_vertex_normals()
    robot.paint_uniform_color([0, 0, 1])
    robot.translate(path_world[0], relative=False)
    
    # Create a coordinate frame for reference.
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Path Animation", width=800, height=600)
    vis.add_geometry(frame)
    vis.add_geometry(obstacles_pc)
    vis.add_geometry(robot)
    
    # Convert path_world to a list of numpy arrays.
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
# Main Routine
#############################################

if __name__ == "__main__":
    # Load occupancy grid (generated from your point cloud processing) from file.
    occupancy = np.load('occupancy_3d.npy')
    # Recompute (or define) global_min from your occupancy grid generation.
    # Replace the value below with the correct global_min for your current occupancy grid.
    global_min = np.array([-48.46652245, -27.08304417, -4.99])
    resolution = 0.2  # Must match the occupancy grid resolution.
    
    # Randomly select two free cells as start and goal.
    free_cells = np.argwhere(occupancy == 0)
    if free_cells.size == 0:
        print("No free space found in the occupancy grid!")
        exit()
    np.random.shuffle(free_cells)
    start_idx = tuple(free_cells[0])
    goal_idx = tuple(free_cells[1])
    print("Random Grid start:", start_idx, "Random Grid goal:", goal_idx)
    
    start_time = time.time()
    path_indices = a_star_3d(occupancy, start_idx, goal_idx)
    end_time = time.time()
    print("A* search took {:.2f} seconds.".format(end_time - start_time))
    if path_indices is None:
        print("No path found between start and goal.")
        exit(0)
    print("Path found with", len(path_indices), "steps.")
    
    # Convert grid indices in the path to world coordinates.
    path_world = [grid_to_world_3d(idx, global_min, resolution) for idx in path_indices]
    
    # Build obstacles point cloud from occupancy grid.
    occ_indices = np.argwhere(occupancy == 1)
    # Downsample obstacles if necessary.
    max_obstacles = 10000
    if occ_indices.shape[0] > max_obstacles:
        factor = int(np.ceil(occ_indices.shape[0] / max_obstacles))
        occ_indices = occ_indices[::factor]
    occ_points = np.array([grid_to_world_3d(idx, global_min, resolution) for idx in occ_indices])
    obstacles_pc = o3d.geometry.PointCloud()
    obstacles_pc.points = o3d.utility.Vector3dVector(occ_points)
    obstacles_pc.paint_uniform_color([1, 0, 0])
    
    # Free occupancy grid memory.
    occupancy = None
    
    # Animate the path (robot and obstacles) using Open3D.
    animate_path_open3d(path_world, obstacles_pc, sleep_time=0.1)

    print("A* search took {:.2f} seconds.".format(end_time - start_time))
    if path_indices is None:
        print("No path found between start and goal.")
        exit(0)
    print("Path found with", len(path_indices), "steps.")
    
    # Convert grid indices in the path to world coordinates.
    path_world = [grid_to_world_3d(idx, global_min, resolution) for idx in path_indices]
    
    # Build obstacles point cloud from occupancy grid.
    occ_indices = np.argwhere(occupancy == 1)
    # Downsample obstacles if necessary.
    max_obstacles = 10000
    if occ_indices.shape[0] > max_obstacles:
        factor = int(np.ceil(occ_indices.shape[0] / max_obstacles))
        occ_indices = occ_indices[::factor]
    occ_points = np.array([grid_to_world_3d(idx, global_min, resolution) for idx in occ_indices])
    obstacles_pc = o3d.geometry.PointCloud()
    obstacles_pc.points = o3d.utility.Vector3dVector(occ_points)
    obstacles_pc.paint_uniform_color([1, 0, 0])
    
    # Free occupancy grid memory.
    occupancy = None
    
    # Animate the path (robot and obstacles) using Open3D.
    animate_path_open3d(path_world, obstacles_pc, sleep_time=0.1)
