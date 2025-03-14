#!/usr/bin/env python3
import os
import numpy as np
import open3d as o3d
import random
import scipy.ndimage as ndi
import heapq

#############################################
# Debug Helper
#############################################
def debug_print(msg, *values):
    """Simple debug print function."""
    print("[DEBUG]", msg, *values)

#############################################
# 1. Grid-to-World Conversion
#############################################
def grid_to_world_3d(idx, global_min, resolution):
    """
    Converts grid indices (i, j, k) to world coordinates (center of voxel).
    Here we assume the occupancy grid is stored as shape (nx, ny, nz) with
    i -> x, j -> y, k -> z.
    """
    return np.array([
        global_min[0] + (idx[0] + 0.5) * resolution,
        global_min[1] + (idx[1] + 0.5) * resolution,
        global_min[2] + (idx[2] + 0.5) * resolution
    ])

#############################################
# 2. Expand Obstacles (Safety Margin)
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
# 3. Uniform Free-Space Decomposition
#############################################
def uniform_free_decomposition(occupancy, block_size, free_thresh=0.9):
    """
    Uniformly partitions the occupancy grid into non-overlapping blocks (cuboids) of fixed size.
    Each block is accepted as free if the fraction of free cells is at least free_thresh.
    
    Parameters:
      occupancy: 3D numpy array with 1=obstacle, 0=free.
      block_size: tuple (Bx, By, Bz) in voxels.
      free_thresh: required free fraction (0<=free_thresh<=1) to accept the block.
    
    Returns a list of dictionaries. Each dictionary has:
      'min_idx': (i, j, k) grid index (lower corner)
      'dimensions': (w, h, d) block dimensions in voxels
    """
    nx, ny, nz = occupancy.shape
    Bx, By, Bz = block_size
    free_blocks = []
    for i in range(0, nx, Bx):
        for j in range(0, ny, By):
            for k in range(0, nz, Bz):
                i_end = min(i + Bx, nx)
                j_end = min(j + By, ny)
                k_end = min(k + Bz, nz)
                block = occupancy[i:i_end, j:j_end, k:k_end]
                total = block.size
                free_count = np.count_nonzero(block == 0)
                free_ratio = free_count / total
                if free_ratio >= free_thresh:
                    dims = np.array([i_end - i, j_end - j, k_end - k])
                    free_blocks.append({
                        'min_idx': np.array([i, j, k]),
                        'dimensions': dims
                    })
    return free_blocks

#############################################
# 4. Convert Free Blocks to World Cuboids
#############################################
def free_blocks_to_cuboids(free_blocks, global_min, resolution):
    """
    Converts a list of free blocks (in grid indices and voxel dimensions)
    to cuboids in world coordinates.
    Each cuboid is represented as a dictionary with keys:
      'lower': world coordinate of lower corner,
      'upper': world coordinate of upper corner,
      'dimensions': (w, h, d) in world units.
    """
    cuboids = []
    for block in free_blocks:
        min_idx = block['min_idx']
        dims = block['dimensions']
        lower = grid_to_world_3d(min_idx, global_min, resolution)
        upper = grid_to_world_3d(min_idx + dims - 1, global_min, resolution)
        dimensions = upper - lower
        cuboids.append({
            'min_idx': min_idx,
            'dimensions': dims,
            'lower': lower,
            'upper': upper,
            'dimensions_world': dimensions
        })
    return cuboids

#############################################
# 5. Build Connectivity Graph over Cuboids
#############################################
def cuboids_connected(cuboid1, cuboid2, tol=0.1):
    """
    Two cuboids are considered connected if their world bounding boxes overlap or touch
    within a tolerance tol in all three dimensions.
    """
    for i in range(3):
        if cuboid1['upper'][i] < cuboid2['lower'][i] - tol or cuboid2['upper'][i] < cuboid1['lower'][i] - tol:
            return False
    return True

def build_graph(cuboids, tol=0.1):
    """
    Constructs a graph from a list of cuboids.
    Each cuboid is a node. An edge exists if the cuboids are connected (overlap/touch).
    The edge cost is the Euclidean distance between their centers.
    Returns a dictionary: {node_index: [(neighbor_index, cost), ...]}.
    """
    graph = {i: [] for i in range(len(cuboids))}
    for i in range(len(cuboids)):
        for j in range(i+1, len(cuboids)):
            if cuboids_connected(cuboids[i], cuboids[j], tol):
                center_i = (cuboids[i]['lower'] + cuboids[i]['upper']) / 2
                center_j = (cuboids[j]['lower'] + cuboids[j]['upper']) / 2
                cost = np.linalg.norm(center_i - center_j)
                graph[i].append((j, cost))
                graph[j].append((i, cost))
    return graph

def a_star_graph(graph, start_idx, goal_idx, cuboids):
    """
    Performs A* search on the graph.
    Heuristic is the Euclidean distance between cuboid centers.
    Returns (path, total_cost) where path is a list of cuboid indices.
    """
    def heuristic(idx):
        center = (cuboids[idx]['lower'] + cuboids[idx]['upper']) / 2
        goal_center = (cuboids[goal_idx]['lower'] + cuboids[goal_idx]['upper']) / 2
        return np.linalg.norm(center - goal_center)
    
    open_set = [(heuristic(start_idx), 0, start_idx, [])]
    closed = set()
    
    while open_set:
        f, g, node, path = heapq.heappop(open_set)
        if node in closed:
            continue
        closed.add(node)
        path = path + [node]
        if node == goal_idx:
            return path, g
        for neighbor, cost in graph[node]:
            if neighbor in closed:
                continue
            g_new = g + cost
            f_new = g_new + heuristic(neighbor)
            heapq.heappush(open_set, (f_new, g_new, neighbor, path))
    return None, np.inf

#############################################
# 6. Create Open3D Box from a Cuboid
#############################################
def create_box_from_cuboid(cuboid):
    """
    Creates an Open3D TriangleMesh box from cuboid data.
    The box is axis-aligned and its lower corner is at cuboid['lower'].
    """
    dims = cuboid['dimensions_world']
    if np.any(dims <= 0):
        return None
    box = o3d.geometry.TriangleMesh.create_box(width=dims[0], height=dims[1], depth=dims[2])
    box.translate(cuboid['lower'], relative=False)
    box.compute_vertex_normals()
    return box

#############################################
# 7. Main Routine
#############################################
if __name__ == "__main__":
    # Load point cloud (Nx3 numpy array)
    pc_file = "/home/raghuram/ARPL/cuboid_decomp/cuboid_decomp-/pointcloud/pointcloud_gq/point_cloud_gq.npy"
    points = np.load(pc_file)
    if points.dtype.names is not None:
        points = np.vstack([points[name] for name in ('x', 'y', 'z')]).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Compute point cloud bounding box
    aabb = pcd.get_axis_aligned_bounding_box()
    min_bound = aabb.min_bound
    max_bound = aabb.max_bound
    debug_print("Point Cloud bounding box: Min:", min_bound, "Max:", max_bound)

    # Build occupancy grid from point cloud (assume grid shape is (nx, ny, nz) with i->x, j->y, k->z)
    resolution = 0.2
    extent = max_bound - min_bound
    nx = int(np.ceil(extent[0] / resolution))
    ny = int(np.ceil(extent[1] / resolution))
    nz = int(np.ceil(extent[2] / resolution))
    debug_print("Grid dims (nx, ny, nz):", (nx, ny, nz))

    occupancy = np.zeros((nx, ny, nz), dtype=np.uint8)
    idxs = np.floor((points - min_bound) / resolution).astype(int)
    idxs = np.clip(idxs, 0, [nx-1, ny-1, nz-1])
    for (ix, iy, iz) in idxs:
        occupancy[ix, iy, iz] = 1
    debug_print("Occupancy grid created. Occupied cells:", int(occupancy.sum()))

    # Expand obstacles (optional safety margin)
    safety_voxels = 2
    occupancy_expanded = expand_obstacles_3d(occupancy, safety_voxels=safety_voxels)
    debug_print("After expansion, occupancy shape:", occupancy_expanded.shape)

    # Uniform free-space decomposition:
    # Partition the occupancy grid into fixed-size blocks.
    block_size = (10, 10, 5)  # in voxels (Bx, By, Bz)
    free_thresh = 0.9         # require 90% free to accept a block
    free_blocks = []
    for i in range(0, nx, block_size[0]):
        for j in range(0, ny, block_size[1]):
            for k in range(0, nz, block_size[2]):
                i_end = min(i + block_size[0], nx)
                j_end = min(j + block_size[1], ny)
                k_end = min(k + block_size[2], nz)
                block = occupancy_expanded[i:i_end, j:j_end, k:k_end]
                total = block.size
                free_count = np.count_nonzero(block==0)
                if free_count / total >= free_thresh:
                    dims = np.array([i_end - i, j_end - j, k_end - k])
                    free_blocks.append({
                        'min_idx': np.array([i, j, k]),
                        'dimensions': dims
                    })
    debug_print("Uniform free blocks found:", len(free_blocks))

    # Convert free blocks to cuboids in world coordinates
    free_cuboids = free_blocks_to_cuboids(free_blocks, min_bound, resolution)
    debug_print("Total free-space cuboids (after conversion):", len(free_cuboids))

    # Build connectivity graph over free cuboids
    def cuboids_connected(c1, c2, tol=0.1):
        for i in range(3):
            if c1['upper'][i] < c2['lower'][i] - tol or c2['upper'][i] < c1['lower'][i] - tol:
                return False
        return True

    def build_graph(cuboids, tol=0.1):
        graph = {i: [] for i in range(len(cuboids))}
        for i in range(len(cuboids)):
            for j in range(i+1, len(cuboids)):
                if cuboids_connected(cuboids[i], cuboids[j], tol):
                    center_i = (cuboids[i]['lower'] + cuboids[i]['upper']) / 2
                    center_j = (cuboids[j]['lower'] + cuboids[j]['upper']) / 2
                    cost = np.linalg.norm(center_i - center_j)
                    graph[i].append((j, cost))
                    graph[j].append((i, cost))
        return graph

    graph = build_graph(free_cuboids, tol=0.1)
    debug_print("Graph built with", len(graph), "nodes.")

    # Choose start and goal cuboids (for instance, the one with minimum x and maximum x)
    # Here we choose the free cuboid with the smallest x-coordinate as start
    # and the one with the largest x-coordinate as goal.
    centers = [(i, (cub['lower'] + cub['upper'])/2) for i, cub in enumerate(free_cuboids)]
    centers.sort(key=lambda t: t[1][0])
    start_idx_cub = centers[0][0]
    goal_idx_cub = centers[-1][0]
    debug_print("Chosen start cuboid index:", start_idx_cub, "and goal cuboid index:", goal_idx_cub)

    # A* search on the graph
    def a_star_graph(graph, start_idx, goal_idx, cuboids):
        def heuristic(idx):
            center = (cuboids[idx]['lower'] + cuboids[idx]['upper']) / 2
            goal_center = (cuboids[goal_idx]['lower'] + cuboids[goal_idx]['upper']) / 2
            return np.linalg.norm(center - goal_center)
        open_set = [(heuristic(start_idx), 0, start_idx, [])]
        closed = set()
        while open_set:
            f, g, node, path = heapq.heappop(open_set)
            if node in closed:
                continue
            closed.add(node)
            path = path + [node]
            if node == goal_idx:
                return path, g
            for neighbor, cost in graph[node]:
                if neighbor in closed:
                    continue
                g_new = g + cost
                f_new = g_new + heuristic(neighbor)
                heapq.heappush(open_set, (f_new, g_new, neighbor, path))
        return None, np.inf

    path_nodes, total_cost = a_star_graph(graph, start_idx_cub, goal_idx_cub, free_cuboids)
    if path_nodes is None:
        print("No path found between the chosen free-space regions.")
        exit(0)
    debug_print("A* path (cuboid indices):", path_nodes, "with cost:", total_cost)

    # Build planned path as sequence of cuboid centers (world coordinates)
    planned_path = []
    for idx in path_nodes:
        center = (free_cuboids[idx]['lower'] + free_cuboids[idx]['upper']) / 2
        planned_path.append(center)

    # Create a thick green LineSet for the planned path
    path_line = o3d.geometry.LineSet()
    path_line.points = o3d.utility.Vector3dVector(planned_path)
    lines = [[i, i+1] for i in range(len(planned_path)-1)]
    path_line.lines = o3d.utility.Vector2iVector(lines)
    path_line.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in lines])

    # Create spheres for start and goal centers
    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
    start_sphere.paint_uniform_color([0, 0, 1])
    start_center = (free_cuboids[start_idx_cub]['lower'] + free_cuboids[start_idx_cub]['upper']) / 2
    start_sphere.translate(start_center, relative=False)
    goal_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
    goal_sphere.paint_uniform_color([1, 0, 0])
    goal_center = (free_cuboids[goal_idx_cub]['lower'] + free_cuboids[goal_idx_cub]['upper']) / 2
    goal_sphere.translate(goal_center, relative=False)

    # Create Open3D boxes for each free-space cuboid for visualization
    free_boxes = []
    for cuboid in free_cuboids:
        if np.any(cuboid['dimensions_world'] < 0.01):  # skip very small blocks
            continue
        box = create_box_from_cuboid({
            'lower': cuboid['lower'],
            'upper': cuboid['upper'],
            'dimensions_world': cuboid['upper'] - cuboid['lower']
        })
        if box is not None:
            box.paint_uniform_color([0.6, 0.8, 1.0])
            free_boxes.append(box)

    # Color the original point cloud in light gray
    pcd.paint_uniform_color([0.7, 0.7, 0.7])

    # Visualize in Open3D: point cloud, free-space boxes, and planned path with start/goal
    o3d.visualization.draw_geometries(
        [pcd] + free_boxes + [path_line, start_sphere, goal_sphere],
        window_name="Graph-based Free-Space Decomposition & A* Path",
        width=1200, height=800)
