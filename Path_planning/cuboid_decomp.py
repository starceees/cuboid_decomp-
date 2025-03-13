import os
import numpy as np
import open3d as o3d
import random
import scipy.ndimage as ndi

#############################################
# Debug Helper
#############################################
def debug_print(msg, *values):
    """Simple debug print function."""
    print("[DEBUG]", msg, *values)

#############################################
# 1. Convert Grid Indices to World Coordinates
#############################################
def grid_to_world_3d(idx, global_min, resolution):
    """
    Converts grid indices (z, y, x) to world coordinates (center of voxel)
    using global_min and resolution.
    """
    return np.array([
        global_min[0] + (idx[2] + 0.5) * resolution,  # x = idx[2]
        global_min[1] + (idx[1] + 0.5) * resolution,  # y = idx[1]
        global_min[2] + (idx[0] + 0.5) * resolution   # z = idx[0]
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
# 3. Obstacle Decomposition (Connected Components of occupancy==1)
#############################################
def obstacle_decomposition(occupancy, global_min, resolution, min_size=5):
    """
    Decomposes the obstacles (occupancy==1) into axis-aligned cuboids by labeling
    connected obstacle components (in 3D) and computing minimal bounding boxes (z,y,x).
    
    Components with fewer than min_size voxels are discarded.
    Returns a list of dictionaries with bounding info.
    """
    # Label connected obstacle cells
    labeled, num_features = ndi.label(occupancy == 1)
    debug_print("Found connected obstacle components:", num_features)

    counts = np.bincount(labeled.flatten())
    comp_sizes = counts[1:]  # skip background (label=0)

    cuboids = []
    for label_idx in range(1, num_features+1):
        size = comp_sizes[label_idx - 1]
        if size < min_size:
            continue
        coords = np.argwhere(labeled == label_idx)
        if coords.size == 0:
            continue
        min_idx = coords.min(axis=0)
        max_idx = coords.max(axis=0)
        lower = grid_to_world_3d(min_idx, global_min, resolution)
        upper = grid_to_world_3d(max_idx, global_min, resolution)
        dims = upper - lower
        debug_print(f"Obstacle label={label_idx}, size={size}")
        debug_print("  min_idx=", min_idx, " max_idx=", max_idx)
        debug_print("  lower=", lower, " upper=", upper, " dims=", dims)
        cuboids.append({
            'min_idx': min_idx,
            'max_idx': max_idx,
            'lower': lower,
            'upper': upper,
            'dimensions': dims
        })
    return cuboids

#############################################
# 4. Free-Space Decomposition (Naive 3D expansions)
#############################################
def solveSafeFlight3D(occupancy, global_min, resolution):
    """
    Naïve 3D rectangular decomposition of free space (occupancy==0).
    For each unvisited free cell, expand a rectangular block in x,y,z.
    Returns a list of (z0, y0, x0, dz, dy, dx).
    """
    nz, ny, nx = occupancy.shape
    visited = np.zeros_like(occupancy, dtype=bool)
    rect_list = []

    def expand_block(z0, y0, x0):
        # Expand in ±x from (z0,y0,x0)
        x_left = x0
        while x_left >= 0 and occupancy[z0,y0,x_left]==0 and not visited[z0,y0,x_left]:
            x_left -= 1
        x_left += 1
        x_right = x0
        while x_right < nx and occupancy[z0,y0,x_right]==0 and not visited[z0,y0,x_right]:
            x_right += 1
        x_right -= 1
        width_x = x_right - x_left + 1

        # Expand in ±y
        y_up = y0
        while y_up >= 0:
            row = occupancy[z0, y_up, x_left:x_right+1]
            row_visited = visited[z0, y_up, x_left:x_right+1]
            if np.any(row!=0) or np.any(row_visited):
                break
            y_up -= 1
        y_up += 1

        y_down = y0
        while y_down < ny:
            row = occupancy[z0, y_down, x_left:x_right+1]
            row_visited = visited[z0, y_down, x_left:x_right+1]
            if np.any(row!=0) or np.any(row_visited):
                break
            y_down += 1
        y_down -= 1
        height_y = y_down - y_up + 1

        # Expand in ±z
        z_front = z0
        while z_front >= 0:
            block = occupancy[z_front, y_up:y_down+1, x_left:x_right+1]
            block_visited = visited[z_front, y_up:y_down+1, x_left:x_right+1]
            if np.any(block!=0) or np.any(block_visited):
                break
            z_front -= 1
        z_front += 1

        z_back = z0
        while z_back < nz:
            block = occupancy[z_back, y_up:y_down+1, x_left:x_right+1]
            block_visited = visited[z_back, y_up:y_down+1, x_left:x_right+1]
            if np.any(block!=0) or np.any(block_visited):
                break
            z_back += 1
        z_back -= 1
        depth_z = z_back - z_front + 1

        # Mark visited
        visited[z_front:z_back+1, y_up:y_down+1, x_left:x_right+1] = True
        return (z_front, y_up, x_left, depth_z, height_y, width_x)

    for z in range(nz):
        for y in range(ny):
            x = 0
            while x < nx:
                if occupancy[z,y,x]==0 and not visited[z,y,x]:
                    # Expand a block
                    block = expand_block(z,y,x)
                    rect_list.append(block)
                    # skip ahead in x
                    x_left = block[2]
                    w_x = block[5]
                    x = x_left + w_x
                else:
                    x +=1

    # Convert these blocks to actual bounding boxes in world coords
    boxes_data = []
    for (z0, y0, x0, dz, dy, dx) in rect_list:
        # Lower corner (z0,y0,x0)
        min_idx = np.array([z0, y0, x0])
        max_idx = min_idx + np.array([dz, dy, dx]) - 1
        lower = grid_to_world_3d(min_idx, global_min, resolution)
        upper = grid_to_world_3d(max_idx, global_min, resolution)
        dims = upper - lower
        boxes_data.append({
            'min_idx': min_idx,
            'max_idx': max_idx,
            'lower': lower,
            'upper': upper,
            'dimensions': dims
        })
    return boxes_data

#############################################
# 5. Create Open3D Box from a bounding data
#############################################
def create_box_from_cuboid(cuboid):
    width, height, depth = cuboid['dimensions']
    if width<=0 or height<=0 or depth<=0:
        return None
    box = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
    box.translate(cuboid['lower'], relative=False)
    box.compute_vertex_normals()
    return box

#############################################
# Main Routine
#############################################
if __name__ == "__main__":
    # 1) Load point cloud from a NumPy file ( Nx3 shape)
    pc_file = "/home/raghuram/ARPL/cuboid_decomp/cuboid_decomp-/pointcloud/pointcloud_gq/point_cloud_gq.npy"
    points = np.load(pc_file)
    if points.dtype.names is not None:
        points = np.vstack([points[name] for name in ('x','y','z')]).T
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # bounding box
    aabb = pcd.get_axis_aligned_bounding_box()
    min_bound = aabb.min_bound
    max_bound = aabb.max_bound
    debug_print("Point Cloud bounding box: Min:", min_bound, "Max:", max_bound)

    # 2) Build occupancy grid
    resolution = 0.2
    extent = max_bound - min_bound
    nx = int(np.ceil(extent[0]/resolution))
    ny = int(np.ceil(extent[1]/resolution))
    nz = int(np.ceil(extent[2]/resolution))
    debug_print("Grid dims:", (nz, ny, nx), "Note: We'll store as (z,y,x)")

    # We'll store occupancy in shape (nz, ny, nx)
    occupancy = np.zeros((nz, ny, nx), dtype=np.uint8)
    # Convert each point to (z,y,x) index
    # world coords are (x,y,z), but we store as (z,y,x)
    # so z = floor((z - min_z)/res), y = floor((y-min_y)/res), x= floor((x-min_x)/res)
    # Then clamp
    diffs = points - min_bound
    idxs_z = np.floor(diffs[:,2]/resolution).astype(int)
    idxs_y = np.floor(diffs[:,1]/resolution).astype(int)
    idxs_x = np.floor(diffs[:,0]/resolution).astype(int)
    idxs_z = np.clip(idxs_z, 0, nz-1)
    idxs_y = np.clip(idxs_y, 0, ny-1)
    idxs_x = np.clip(idxs_x, 0, nx-1)
    for i in range(points.shape[0]):
        occupancy[idxs_z[i], idxs_y[i], idxs_x[i]] = 1
    debug_print("Occupancy grid created. #occupied=", occupancy.sum())

    # 3) Expand obstacles
    safety_voxels=2
    occupancy_expanded = expand_obstacles_3d(occupancy, safety_voxels)
    debug_print("After expansion, occupancy shape:", occupancy_expanded.shape)

    # 4a) Obstacle decomposition approach
    obs_cuboids = obstacle_decomposition(occupancy_expanded, min_bound, resolution, min_size=5)
    debug_print("Obstacle cuboids:", len(obs_cuboids))

    # Convert to boxes
    obs_boxes = []
    for c in obs_cuboids:
        if np.any(c['dimensions'] < 0.1):
            continue
        box = create_box_from_cuboid(c)
        if box is not None:
            box.paint_uniform_color([random.random(), random.random(), random.random()])
            obs_boxes.append(box)
    
    debug_print(f"Obstacle boxes = {len(obs_boxes)}")

    # 4b) Free-space naive expansions approach
    free_cuboids = solveSafeFlight3D(occupancy_expanded, min_bound, resolution)
    debug_print("Free-space blocks from solveSafeFlight3D:", len(free_cuboids))

    free_boxes = []
    for c in free_cuboids:
        if np.any(c['dimensions'] < 0.1):
            continue
        box = create_box_from_cuboid(c)
        if box is not None:
            box.paint_uniform_color([random.random(), random.random(), random.random()])
            free_boxes.append(box)
    debug_print(f"Free-space boxes = {len(free_boxes)}")

    # 5) Visualize everything in Open3D
    # Color the original point cloud in gray
    pcd.paint_uniform_color([0.6, 0.6, 0.6])

    # Option A: Show obstacles only
    # o3d.visualization.draw_geometries([pcd] + obs_boxes,
    #                                   window_name="Obstacle Decomposition",
    #                                   width=1200, height=800)

    # Option B: Show free space only
    # o3d.visualization.draw_geometries([pcd] + free_boxes,
    #                                   window_name="Free-Space Decomposition",
    #                                   width=1200, height=800)

    # Option C: Show both
    # We'll color obstacles boxes + free boxes
    # Just combine them all
    all_boxes = obs_boxes + free_boxes
    o3d.visualization.draw_geometries([pcd] + all_boxes,
                                      window_name="Obstacles (various colors) + Free Space (various colors)",
                                      width=1200,
                                      height=800)
