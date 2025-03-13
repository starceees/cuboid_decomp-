import numpy as np
import heapq
from tqdm import tqdm
import time
import open3d as o3d
import scipy.ndimage as ndi

#############################################
# 1. Obstacle Expansion (Safety Margin)
#############################################

def expand_obstacles_3d(occupancy, safety_voxels=2):
    """
    Expands obstacles in a 3D occupancy grid by 'safety_voxels' using morphological dilation.
    Returns a new 3D array with obstacles expanded.
    """
    structure = ndi.generate_binary_structure(3, 1)  # 6-connected structure
    expanded = ndi.binary_dilation(occupancy.astype(bool), structure=structure, iterations=safety_voxels)
    return expanded.astype(np.uint8)

#############################################
# 2. RRT-based 3D Path Planning (with Z-limit)
#############################################

def collision_free(occupancy, p1, p2, interp_resolution=0.5):
    """
    Checks if the straight-line path between grid points p1 and p2 is collision free.
    Interpolates along the line with steps of size 'interp_resolution'.
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    dist = np.linalg.norm(p2 - p1)
    n_steps = int(np.ceil(dist / interp_resolution)) + 1
    for t in np.linspace(0, 1, n_steps):
        pt = p1 + t*(p2 - p1)
        idx = tuple(np.round(pt).astype(int))
        # Check bounds
        if (idx[0] < 0 or idx[0] >= occupancy.shape[0] or
            idx[1] < 0 or idx[1] >= occupancy.shape[1] or
            idx[2] < 0 or idx[2] >= occupancy.shape[2]):
            return False
        if occupancy[idx] == 1:
            return False
    return True

def rrt_3d(occupancy, start, goal, max_iter=5000000, step_size=20, z_upper=None):
    """
    A simple RRT planner in grid space, restricting z to be <= z_upper.
    occupancy: 3D numpy array (1=obstacle, 0=free).
    start, goal: grid indices (tuple of ints).
    max_iter: maximum iterations.
    step_size: maximum extension (in grid cells) per iteration.
    z_upper: maximum z index to allow (if None, use full range).
    Returns a list of grid indices representing the path, or None if planning fails.
    """
    dims = occupancy.shape
    if z_upper is None or z_upper > dims[2]:
        z_upper = dims[2]
    print("running rrt planner: max_iter =", max_iter, 
          "step_size =", step_size, "z_upper =", z_upper)
    
    def sample_free():
        # Sample random free cell, but limit z <= z_upper-1.
        while True:
            ix = np.random.randint(0, dims[0])
            iy = np.random.randint(0, dims[1])
            iz = np.random.randint(0, z_upper)  # restricted z range
            if occupancy[ix, iy, iz] == 0:
                return (ix, iy, iz)

    tree = {start: None}
    nodes = [start]
    
    for i in range(max_iter):
        rand_node = sample_free()
        nearest = min(nodes, key=lambda n: np.linalg.norm(np.array(n) - np.array(rand_node)))
        vec = np.array(rand_node) - np.array(nearest)
        d = np.linalg.norm(vec)
        if d == 0:
            continue
        # Extend from nearest toward rand_node by step_size.
        if d > step_size:
            new_node = np.round(np.array(nearest) + (vec / d) * step_size).astype(int)
        else:
            new_node = np.array(rand_node)
        
        # If new_node's z is beyond z_upper, skip it.
        if new_node[2] >= z_upper:
            continue
        
        new_node = tuple(new_node)
        if occupancy[new_node] == 1:
            continue
        if not collision_free(occupancy, nearest, new_node):
            continue
        tree[new_node] = nearest
        nodes.append(new_node)
        # If new_node is close enough to the goal, try connecting directly.
        if np.linalg.norm(np.array(new_node) - np.array(goal)) <= step_size:
            # Also clamp goal's z if needed.
            if goal[2] < z_upper and collision_free(occupancy, new_node, goal):
                tree[goal] = new_node
                path = []
                current = goal
                while current is not None:
                    path.append(current)
                    current = tree[current]
                return path[::-1]
    return None

#############################################
# 3. Grid-to-World Conversion
#############################################

def grid_to_world_3d(idx, global_min, resolution):
    """
    Converts grid indices (ix, iy, iz) to world coordinates (center of voxel).
    """
    return np.array([
        global_min[0] + (idx[0] + 0.5) * resolution,
        global_min[1] + (idx[1] + 0.5) * resolution,
        global_min[2] + (idx[2] + 0.5) * resolution
    ])

#############################################
# 4. Open3D Animation with Thicker Path Visualization
#############################################

def animate_path_open3d(path_world, obstacles_pc, sleep_time=0.1, line_width=5.0):
    """
    Animates the given path (a list of world-coordinate points) using Open3D.
    Displays:
      - A blue sphere (robot) moving along the path,
      - A red point cloud for obstacles,
      - A green line (LineSet) representing the full path, with increased line width.
    """
    # Create a blue sphere for the robot.
    robot = o3d.geometry.TriangleMesh.create_sphere(radius=5)
    robot.compute_vertex_normals()
    robot.paint_uniform_color([0, 0, 1])
    robot.translate(path_world[0], relative=False)
    
    # Create a LineSet for the path.
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(path_world)
    lines = [[i, i+1] for i in range(len(path_world)-1)]
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in lines])  # Green
    
    # Create a coordinate frame.
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Path Animation", width=800, height=600)
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
    # Load the raw point cloud from file to build the occupancy grid.
    pc_file = "/home/raghuram/ARPL/cuboid_decomp/cuboid_decomp-/pointcloud/pointcloud_gq/point_cloud_gq.npy"
    points = np.load(pc_file)
    if points.dtype.names is not None:
        points = np.vstack([points[name] for name in ('x', 'y', 'z')]).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Compute occupancy grid from the point cloud.
    resolution = 0.2
    margin = 0.0  # No extra margin in occupancy grid generation.
    aabb = pcd.get_axis_aligned_bounding_box()
    min_bound = aabb.min_bound
    max_bound = aabb.max_bound
    print("Point Cloud Bounding Box:")
    print("  Min:", min_bound)
    print("  Max:", max_bound)
    
    extent = max_bound - min_bound
    nx = int(np.ceil(extent[0] / resolution))
    ny = int(np.ceil(extent[1] / resolution))
    nz = int(np.ceil(extent[2] / resolution))
    occupancy = np.zeros((nx, ny, nz), dtype=np.uint8)
    pts = np.asarray(pcd.points)
    idxs = np.floor((pts - min_bound) / resolution).astype(int)
    idxs = np.clip(idxs, 0, [nx-1, ny-1, nz-1])
    for ix, iy, iz in idxs:
        occupancy[ix, iy, iz] = 1
    global_min = min_bound
    print(f"Occupancy grid shape {occupancy.shape}. global_min: {global_min}")
    
    # Expand obstacles for planning.
    safety_voxels = 2
    occupancy_safety = expand_obstacles_3d(occupancy, safety_voxels=safety_voxels)
    
    # Randomly select free cells from the expanded grid as start and goal.
    free_cells = np.argwhere(occupancy_safety == 0)
    if free_cells.size == 0:
        print("No free space found in the safety occupancy grid!")
        exit()
    np.random.shuffle(free_cells)
    start_idx = tuple(free_cells[0])
    goal_idx = tuple(free_cells[1])
    print("Random Grid start:", start_idx, "Random Grid goal:", goal_idx)
    
    # Create a point cloud from the original occupancy grid (undilated) for visualization.
    occ_indices = np.argwhere(occupancy == 1)
    max_obstacles = 10000
    if occ_indices.shape[0] > max_obstacles:
        factor = int(np.ceil(occ_indices.shape[0] / max_obstacles))
        occ_indices = occ_indices[::factor]
    occ_points = np.array([grid_to_world_3d(tuple(idx), global_min, resolution) for idx in occ_indices])
    obstacles_pc = o3d.geometry.PointCloud()
    obstacles_pc.points = o3d.utility.Vector3dVector(occ_points)
    obstacles_pc.paint_uniform_color([1, 0, 0])  # Red obstacles.
    
    # Show the random start and goal in a debug visualization (before planning).
    start_marker = o3d.geometry.TriangleMesh.create_sphere(radius=7)
    start_marker.paint_uniform_color([0, 0, 1])  # Blue
    start_marker.compute_vertex_normals()
    start_marker.translate(grid_to_world_3d(start_idx, global_min, resolution), relative=False)
    
    goal_marker = o3d.geometry.TriangleMesh.create_sphere(radius=7)
    goal_marker.paint_uniform_color([0, 1, 0])  # Green
    goal_marker.compute_vertex_normals()
    goal_marker.translate(grid_to_world_3d(goal_idx, global_min, resolution), relative=False)
    
    vis_debug = o3d.visualization.Visualizer()
    vis_debug.create_window(window_name="Pre-planning Visualization", width=800, height=600)
    vis_debug.add_geometry(obstacles_pc)
    vis_debug.add_geometry(start_marker)
    vis_debug.add_geometry(goal_marker)
    vis_debug.run()
    vis_debug.destroy_window()
    
    # Restrict the z-range to 2/3 of the total height (in grid cells).
    # So the RRT won't attempt to sample above that fraction.
    z_upper = int(nz * (2.0/3.0))
    print(f"Restricting z range to [0, {z_upper}) out of {nz}.")
    
    # RRT planning on the safety occupancy grid with the z-limit.
    start_time = time.time()
    path_indices = rrt_3d(occupancy_safety, start_idx, goal_idx,
                          max_iter=5000000, step_size=20, z_upper=z_upper)
    end_time = time.time()
    print("RRT search took {:.2f} seconds.".format(end_time - start_time))
    if path_indices is None:
        print("No path found between start and goal.")
        exit(0)
    print("Path found with", len(path_indices), "steps.")
    
    # Convert path grid indices to world coordinates.
    path_world = [grid_to_world_3d(idx, global_min, resolution) for idx in path_indices]
    
    # Animate the planned path using Open3D.
    animate_path_open3d(path_world, obstacles_pc, sleep_time=0.1, line_width=5.0)
