#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
import sensor_msgs_py.point_cloud2 as pc2
import random
import heapq
import os
import pickle
import time
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

#############################################
# Saving / Loading Functions
#############################################
def save_cuboids(cuboids, filename="my_cuboids.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(cuboids, f)
        
def load_cuboids(filename="my_cuboids.pkl"):
    with open(filename, 'rb') as f:
        cuboids = pickle.load(f)
    return cuboids

#############################################
# 1. Obstacle Expansion (3D Safety Margin)
#############################################
def expand_obstacles_3d(occupancy, safety_voxels=2):
    from scipy.ndimage import generate_binary_structure, binary_dilation
    structure = generate_binary_structure(rank=3, connectivity=1)
    expanded = binary_dilation(occupancy.astype(bool), structure=structure, iterations=safety_voxels)
    return expanded.astype(np.uint8)

#############################################
# 2. Build 3D Occupancy Grid from Point Cloud
#############################################
def build_occupancy_grid(points, global_min, resolution):
    extent = np.max(points, axis=0) - global_min
    nx_dim = int(np.ceil(extent[0] / resolution))
    ny_dim = int(np.ceil(extent[1] / resolution))
    nz_dim = int(np.ceil(extent[2] / resolution))
    occupancy = np.zeros((nx_dim, ny_dim, nz_dim), dtype=np.uint8)
    
    idxs = np.floor((points - global_min) / resolution).astype(int)
    idxs = np.clip(idxs, 0, [nx_dim-1, ny_dim-1, nz_dim-1])
    for (ix, iy, iz) in idxs:
        occupancy[ix, iy, iz] = 1
    return occupancy

#############################################
# 3. Region Growing for Free Cuboids
#############################################
def region_growing_3d(occupancy, max_z_thickness=5):
    nx, ny, nz = occupancy.shape
    visited = np.zeros_like(occupancy, dtype=bool)
    cuboids = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if occupancy[i, j, k] == 0 and not visited[i, j, k]:
                    i_min, i_max = i, i
                    j_min, j_max = j, j
                    k_min, k_max = k, k
                    changed = True
                    while changed:
                        changed = False
                        if i_max < nx - 1:
                            candidate = occupancy[i_max+1, j_min:j_max+1, k_min:k_max+1]
                            if np.all(candidate == 0):
                                i_max += 1
                                changed = True
                        if j_max < ny - 1:
                            candidate = occupancy[i_min:i_max+1, j_max+1, k_min:k_max+1]
                            if np.all(candidate == 0):
                                j_max += 1
                                changed = True
                        current_z = k_max - k_min + 1
                        if current_z < max_z_thickness and k_max < nz - 1:
                            candidate = occupancy[i_min:i_max+1, j_min:j_max+1, k_max+1]
                            if np.all(candidate == 0):
                                k_max += 1
                                changed = True
                    visited[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1] = True
                    cuboids.append({
                        'min_idx': np.array([i_min, j_min, k_min]),
                        'dimensions': np.array([i_max - i_min + 1,
                                                j_max - j_min + 1,
                                                k_max - k_min + 1])
                    })
    return cuboids

#############################################
# 4. Convert Grid Cuboid to World Coordinates
#############################################
def block_to_world_cuboid(block, global_min, resolution):
    min_idx = block['min_idx']
    dims = block['dimensions']
    lower = global_min + (min_idx * resolution)
    upper = global_min + ((min_idx + dims) * resolution)
    return {
        'lower': lower,
        'upper': upper,
        'dimensions_world': upper - lower
    }

#############################################
# 5. Strict Line-of-Sight Checking (Cuboid-level)
#############################################
def line_of_sight_in_cuboids(cub1, cub2, step=0.05):
    c1 = (cub1['lower'] + cub1['upper']) / 2.0
    c2 = (cub2['lower'] + cub2['upper']) / 2.0
    vec = c2 - c1
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return True
    steps = int(np.ceil(1.0 / step))
    for i in range(steps + 1):
        t = i * step
        if t > 1.0:
            t = 1.0
        point = c1 + t * vec
        if not point_in_union_of_two_cuboids(point, cub1, cub2):
            return False
    return True

def point_in_union_of_two_cuboids(pt, cub1, cub2):
    return point_in_cuboid(pt, cub1) or point_in_cuboid(pt, cub2)

def point_in_cuboid(pt, cub):
    return np.all(pt >= cub['lower']) and np.all(pt <= cub['upper'])

#############################################
# 6. Build Cuboids with Connectivity (Graph on the Fly)
#############################################
def build_cuboids_with_connectivity(free_cuboids_blocks, global_min, resolution, step=0.05):
    cuboids = []
    for block in tqdm(free_cuboids_blocks, desc="Building cuboid connectivity"):
        cub = block_to_world_cuboid(block, global_min, resolution)
        cub['neighbors'] = []
        for idx, existing in enumerate(cuboids):
            if cuboids_touch_or_overlap(cub, existing, tol=0.0):
                if line_of_sight_in_cuboids(cub, existing, step=step):
                    cub['neighbors'].append(idx)
                    existing.setdefault('neighbors', []).append(len(cuboids))
        cuboids.append(cub)
    return cuboids

def cuboids_touch_or_overlap(c1, c2, tol=0.0):
    for i in range(3):
        if c1['upper'][i] < c2['lower'][i] - tol or c2['upper'][i] < c1['lower'][i] - tol:
            return False
    return True

#############################################
# 7. A* Path Planning on the Cuboid Graph (Coarse Path)
#############################################
def astar_on_cuboids(cuboids, start_idx, goal_idx):
    def center(cub):
        return (cub['lower'] + cub['upper']) / 2.0
    def heuristic(i, j):
        return np.linalg.norm(center(cuboids[i]) - center(cuboids[j]))
    open_set = []
    best_cost = {start_idx: 0}
    visited = set()
    heapq.heappush(open_set, (heuristic(start_idx, goal_idx), 0, start_idx, [start_idx]))
    while open_set:
        open_set.sort(key=lambda x: x[0])
        f, g, current, path = open_set.pop(0)
        if current == goal_idx:
            return path
        if current in visited:
            continue
        visited.add(current)
        for nb in cuboids[current]['neighbors']:
            if nb in visited:
                continue
            cost_new = g + heuristic(current, nb)
            if nb not in best_cost or cost_new < best_cost[nb]:
                best_cost[nb] = cost_new
                f_new = cost_new + heuristic(nb, goal_idx)
                new_path = path + [nb]
                heapq.heappush(open_set, (f_new, cost_new, nb, new_path))
    return None

#############################################
# 8. BFS Corridor Extraction
#############################################
def bfs_cuboid_corridor(cuboids, start_idx, goal_idx):
    queue = deque([[start_idx]])
    visited = set()
    while queue:
        path = queue.popleft()
        current = path[-1]
        if current == goal_idx:
            return path
        if current in visited:
            continue
        visited.add(current)
        for nb in cuboids[current].get('neighbors', []):
            if nb not in visited:
                queue.append(path + [nb])
    return None

#############################################
# 9. Build a 2D Corridor Grid + Fine A* in the Corridor
#############################################
def create_2d_corridor_grid(cuboids, corridor_indices, grid_resolution=0.1):
    xs, ys = [], []
    for idx in corridor_indices:
        c = cuboids[idx]
        xs.extend([c['lower'][0], c['upper'][0]])
        ys.extend([c['lower'][1], c['upper'][1]])
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    nx = int(np.ceil(width / grid_resolution))
    ny = int(np.ceil(height / grid_resolution))
    grid = np.zeros((ny, nx), dtype=np.uint8)  # 1=free, 0=blocked
    for i in range(ny):
        for j in range(nx):
            x = min_x + (j + 0.5)*grid_resolution
            y = min_y + (i + 0.5)*grid_resolution
            free = False
            for idx in corridor_indices:
                c = cuboids[idx]
                if (x >= c['lower'][0] and x <= c['upper'][0] and
                    y >= c['lower'][1] and y <= c['upper'][1]):
                    free = True
                    break
            grid[i, j] = 1 if free else 0
    return grid, min_x, min_y, nx, ny

def astar_grid(grid, start_ij, goal_ij):
    moves = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
    open_list = []
    visited = set()
    expansions = 0
    best_cost = {}
    g0 = 0
    f0 = heuristic(start_ij, goal_ij)
    best_cost[start_ij] = 0
    heapq.heappush(open_list, (f0, g0, start_ij, [start_ij]))
    while open_list:
        open_list.sort(key=lambda x: x[0])
        f, g, current, path = open_list.pop(0)
        expansions += 1
        if current == goal_ij:
            return path, expansions, len(path)
        if current in visited:
            continue
        visited.add(current)
        for mv in moves:
            ni = current[0] + mv[0]
            nj = current[1] + mv[1]
            if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                if grid[ni, nj] == 1:
                    cost_new = g + 1.0
                    if (ni, nj) not in best_cost or cost_new < best_cost[(ni, nj)]:
                        best_cost[(ni, nj)] = cost_new
                        fn = cost_new + heuristic((ni, nj), goal_ij)
                        open_list.append((fn, cost_new, (ni, nj), path + [(ni, nj)]))
    return None, expansions, 0

def astar_in_corridor_2d(cuboids, corridor, start_xyz, goal_xyz, grid_res=0.1):
    grid, min_x, min_y, nx, ny = create_2d_corridor_grid(cuboids, corridor, grid_res)
    sx = (start_xyz[0] - min_x) / grid_res
    sy = (start_xyz[1] - min_y) / grid_res
    gx = (goal_xyz[0] - min_x) / grid_res
    gy = (goal_xyz[1] - min_y) / grid_res
    start_ij = (int(sy), int(sx))
    goal_ij = (int(gy), int(gx))
    path_ij, expansions, sol_nodes = astar_grid(grid, start_ij, goal_ij)
    if path_ij is None:
        return None, expansions, 0.0, 0, start_ij, goal_ij
    path_world = []
    for (i, j) in path_ij:
        wx = min_x + (j + 0.5) * grid_res
        wy = min_y + (i + 0.5) * grid_res
        wz = start_xyz[2]
        path_world.append([wx, wy, wz])
    path_length = 0.0
    for k in range(len(path_world)-1):
        p0 = np.array(path_world[k])
        p1 = np.array(path_world[k+1])
        path_length += np.linalg.norm(p1 - p0)
    return path_world, expansions, path_length, sol_nodes, start_ij, goal_ij

#############################################
# 10. Update Connectivity for Filtered Cuboids
#############################################
def update_connectivity(cuboids, step=0.05, tol=0.0):
    n = len(cuboids)
    for cub in cuboids:
        cub['neighbors'] = []
    for i in range(n):
        for j in range(i+1, n):
            if cuboids_touch_or_overlap(cuboids[i], cuboids[j], tol):
                if line_of_sight_in_cuboids(cuboids[i], cuboids[j], step=step):
                    cuboids[i]['neighbors'].append(j)
                    cuboids[j]['neighbors'].append(i)

#############################################
# 11. Helper Function to Find Nearest Cuboid for a Given Waypoint
#############################################
def find_nearest_cuboid_index(waypoint, cuboids):
    min_dist = float("inf")
    best_idx = None
    for i, cub in enumerate(cuboids):
        if np.all(waypoint >= cub['lower']) and np.all(waypoint <= cub['upper']):
            return i
        center = (cub['lower'] + cub['upper']) / 2.0
        dist = np.linalg.norm(waypoint - center)
        if dist < min_dist:
            min_dist = dist
            best_idx = i
    return best_idx

#############################################
# 12. ROS2 Publisher Node for RViz Visualization
#############################################
class RVizPublisher(Node):
    def __init__(self, points, cuboids, path_coords_3d, exploration_path_coords, drone_mesh_resource, waypoints, frame_id="map"):
        super().__init__('rviz_los_planner')
        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.pc_pub = self.create_publisher(PointCloud2, '/point_cloud', qos)
        self.marker_pub_all = self.create_publisher(MarkerArray, '/free_cuboids', qos)
        self.marker_pub_path = self.create_publisher(MarkerArray, '/path_cuboids', qos)
        self.line_path_pub = self.create_publisher(Marker, '/line_path', qos)
        self.exploration_path_pub = self.create_publisher(Path, '/exploration_path', qos)
        self.drone_pub = self.create_publisher(Marker, '/drone_mesh', qos)
        self.waypoint_pub = self.create_publisher(MarkerArray, '/waypoints', qos)
        
        self.points = points
        self.cuboids = cuboids
        self.path_coords = path_coords_3d  # final refined 3D path points
        self.exploration_path_coords = exploration_path_coords
        self.waypoints = waypoints
        self.drone_mesh_resource = drone_mesh_resource
        self.frame_id = frame_id
        
        self.current_segment_index = 0
        self.alpha = 0.0
        self.alpha_increment = 0.02
        
        self.timer = self.create_timer(0.1, self.publish_all)
        self.get_logger().info("LOS Planner Node initialized.")
    
    def publish_all(self):
        now = self.get_clock().now().to_msg()
        self.publish_point_cloud(now)
        self.publish_all_cuboids(now)
        self.publish_path_cuboids(now)
        self.publish_line_path(now)
        self.publish_exploration_path(now)
        self.publish_drone_marker(now)
        self.publish_waypoints(now)
    
    def publish_point_cloud(self, stamp):
        header = Header()
        header.stamp = stamp
        header.frame_id = self.frame_id
        pc_list = self.points.tolist()
        pc_msg = pc2.create_cloud_xyz32(header, pc_list)
        self.pc_pub.publish(pc_msg)
    
    def publish_all_cuboids(self, stamp):
        marker_array = MarkerArray()
        mid = 0
        for cub in self.cuboids:
            lower = cub['lower']
            upper = cub['upper']
            dims = cub['dimensions_world']
            center = (lower + upper) / 2.0
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = stamp
            m.ns = "all_cuboids"
            m.id = mid
            mid += 1
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(center[0])
            m.pose.position.y = float(center[1])
            m.pose.position.z = float(center[2])
            m.pose.orientation.w = 1.0
            m.scale.x = float(dims[0])
            m.scale.y = float(dims[1])
            m.scale.z = float(dims[2])
            m.color.r = 0.2
            m.color.g = 0.8
            m.color.b = 0.2
            m.color.a = 0.3
            marker_array.markers.append(m)
        self.marker_pub_all.publish(marker_array)
    
    def publish_path_cuboids(self, stamp):
        marker_array = MarkerArray()
        mid = 0
        for pt in self.path_coords:
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = stamp
            m.ns = "path_cuboids"
            m.id = mid
            mid += 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(pt[0])
            m.pose.position.y = float(pt[1])
            m.pose.position.z = float(pt[2])
            m.pose.orientation.w = 1.0
            m.scale.x = 0.3
            m.scale.y = 0.3
            m.scale.z = 0.3
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 1.0
            marker_array.markers.append(m)
        self.marker_pub_path.publish(marker_array)
    
    def publish_line_path(self, stamp):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.ns = "line_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.points = []
        for pt in self.path_coords:
            p = Point()
            p.x = float(pt[0])
            p.y = float(pt[1])
            p.z = float(pt[2])
            marker.points.append(p)
        self.line_path_pub.publish(marker)
    
    def publish_exploration_path(self, stamp):
        path_msg = Path()
        path_msg.header.stamp = stamp
        path_msg.header.frame_id = self.frame_id
        for pt in self.exploration_path_coords:
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = self.frame_id
            pose.pose.position.x = float(pt[0])
            pose.pose.position.y = float(pt[1])
            pose.pose.position.z = float(pt[2])
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.exploration_path_pub.publish(path_msg)
    
    def publish_drone_marker(self, stamp):
        if len(self.path_coords) < 2:
            return
        if self.current_segment_index >= len(self.path_coords) - 1:
            self.current_segment_index = 0
        pt1 = np.array(self.path_coords[self.current_segment_index])
        pt2 = np.array(self.path_coords[self.current_segment_index + 1])
        pos = (1 - self.alpha) * pt1 + self.alpha * pt2
        self.alpha += self.alpha_increment
        if self.alpha >= 1.0:
            self.alpha = 0.0
            self.current_segment_index += 1
            if self.current_segment_index >= len(self.path_coords) - 1:
                self.current_segment_index = 0
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = stamp
        m.ns = "drone"
        m.id = 0
        m.type = Marker.MESH_RESOURCE
        m.action = Marker.ADD
        m.mesh_resource = self.drone_mesh_resource
        m.pose.position.x = float(pos[0])
        m.pose.position.y = float(pos[1])
        m.pose.position.z = float(pos[2])
        m.pose.orientation.w = 1.0
        m.scale.x = 5.0
        m.scale.y = 5.0
        m.scale.z = 5.0
        m.color.a = 1.0
        self.drone_pub.publish(m)
    
    def publish_waypoints(self, stamp):
        marker_array = MarkerArray()
        for idx, wp in enumerate(self.waypoints):
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = stamp
            m.ns = "waypoints"
            m.id = idx
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(wp[0])
            m.pose.position.y = float(wp[1])
            m.pose.position.z = float(wp[2])
            m.pose.orientation.w = 1.0
            m.scale.x = 0.5
            m.scale.y = 0.5
            m.scale.z = 0.5
            m.color.r = 0.0
            m.color.g = 0.0
            m.color.b = 1.0
            m.color.a = 1.0
            marker_array.markers.append(m)
        self.waypoint_pub.publish(marker_array)

#############################################
# 13. MAIN Routine with Waypoint-based Iterative Planning and Metrics
#############################################
def main(args=None):
    rclpy.init(args=args)
    
    # 1) Load / Filter point cloud
    pc_file = "/home/raghuram/ARPL/cuboid_decomp/cuboid_decomp-/pointcloud/pointcloud_gq/point_cloud_gq.npy"
    points = np.load(pc_file)
    if points.dtype.names is not None:
        points = np.vstack([points[name] for name in ('x','y','z')]).T
    min_point_z = -0.5
    points = points[points[:,2] >= min_point_z]
    global_min = np.min(points, axis=0)
    global_max = np.max(points, axis=0)
    print("Point Cloud Bounding Box:")
    print("  Min:", global_min)
    print("  Max:", global_max)
    
    # 2) Build occupancy grid and expand obstacles
    resolution = 0.2
    occupancy = build_occupancy_grid(points, global_min, resolution)
    print(f"Occupancy grid shape: {occupancy.shape}, Occupied voxels: {occupancy.sum()}")
    safety_voxels = 2
    occupancy_expanded = expand_obstacles_3d(occupancy, safety_voxels)
    
    # 3) Region growing to find free cuboids
    max_z_thickness = 20
    blocks = region_growing_3d(occupancy_expanded, max_z_thickness)
    print("Found", len(blocks), "free cuboids.")
    
    # 4) Build connectivity among cuboids
    cuboids_file = "my_cuboids_hos.pkl"
    t0_cub = time.time()
    if os.path.exists(cuboids_file):
        print(f"Loading cuboids from {cuboids_file}...")
        cuboids = load_cuboids(cuboids_file)
    else:
        print("Cuboids file not found. Building cuboids with connectivity...")
        cuboids = build_cuboids_with_connectivity(blocks, global_min, resolution, step=0.05)
        for c in cuboids:
            c['dimensions_world'] = c['upper'] - c['lower']
        save_cuboids(cuboids, cuboids_file)
        print(f"Saved cuboids to {cuboids_file}")
    t1_cub = time.time()
    cuboid_time = t1_cub - t0_cub
    print(f"Cuboid decomposition time: {cuboid_time:.3f} seconds")
    
    print(f"Total cuboids loaded: {len(cuboids)}")
    update_connectivity(cuboids, step=0.05, tol=0.0)
    
    # 5) Select free-space waypoints (do not force z=0)
    num_waypoints = 5
    free_voxel_indices = np.argwhere(occupancy_expanded == 0)
    if len(free_voxel_indices) < num_waypoints:
        print("Not enough free voxels for waypoint selection.")
        return
    selected_indices = random.sample(range(len(free_voxel_indices)), num_waypoints)
    waypoints = []
    for idx in selected_indices:
        voxel_idx = free_voxel_indices[idx]
        wp = global_min + (voxel_idx + 0.5) * resolution
        waypoints.append(wp)
    waypoints = [np.array(wp) for wp in waypoints]
    print("Selected free-space waypoints:")
    for wp in waypoints:
        print(wp)
    
    # 6) For each waypoint, find nearest cuboid index
    waypoint_indices = [find_nearest_cuboid_index(wp, cuboids) for wp in waypoints]
    print("Nearest cuboid indices for waypoints:", waypoint_indices)
    
    # 7) For each consecutive waypoint pair, extract corridor and run fine A*
    final_3d_path = []
    path_planning_times = []
    metric_lines = []
    segment_count = 0
    for i in range(num_waypoints - 1):
        s_wp = waypoints[i]
        g_wp = waypoints[i+1]
        s_idx = waypoint_indices[i]
        g_idx = waypoint_indices[i+1]
        segment_count += 1
        
        corridor = bfs_cuboid_corridor(cuboids, s_idx, g_idx)
        if corridor is None:
            print(f"No corridor found between waypoint {i} and {i+1}!")
            continue
        
        t0_path = time.time()
        path_world, expansions, path_length, sol_nodes, startG, goalG = astar_in_corridor_2d(
            cuboids, corridor, s_wp, g_wp, grid_res=0.1
        )
        t1_path = time.time()
        compute_time = t1_path - t0_path
        if path_world is None:
            print(f"No fine path found in corridor between waypoint {i} and {i+1}!")
            path_world = []
        if i == 0:
            final_3d_path.extend(path_world)
        else:
            final_3d_path.extend(path_world[1:])
        seg_str = f"{segment_count}\t{startG}\t{goalG}\t{compute_time:.3f}\t{path_length:.3f}\t{expansions}\t{sol_nodes}"
        metric_lines.append(seg_str)
    
    with open("detailed_path_metrics.txt", "w") as f:
        f.write("Segment\tStartGrid\tGoalGrid\tComputeTime(s)\tPathLength(m)\tExpandedNodes\tSolutionNodes\n")
        for line in metric_lines:
            f.write(line + "\n")
    print("Saved path metrics to 'detailed_path_metrics.txt'")
    
    direct_path = final_3d_path
    path_coords = direct_path  # final refined 3D path
    
    drone_mesh_resource = "file:///home/raghuram/ARPL/cuboid_decomp/cuboid_decomp-/simulator/meshes/race2.stl"
    
    publisher_node = RVizPublisher(
        points=points,
        cuboids=cuboids,
        path_coords_3d=direct_path,
        exploration_path_coords=[],
        drone_mesh_resource=drone_mesh_resource,
        waypoints=waypoints,
        frame_id="map"
    )
    
    try:
        rclpy.spin(publisher_node)
    except KeyboardInterrupt:
        pass
    finally:
        publisher_node.destroy_node()
        rclpy.shutdown()

#############################################
# 14. DFS-based Exploration Path Planning with Radius (if needed)
#############################################
def build_exploration_path_radius(G, start_node, radius=10.0):
    explored = set()
    path = []
    current = start_node
    explored.add(current)
    path.append(current)
    while True:
        current_center = (G.nodes[current]['cuboid']['lower'] + G.nodes[current]['cuboid']['upper']) / 2.0
        candidates = []
        for nb in G.neighbors(current):
            if nb not in explored:
                nb_center = (G.nodes[nb]['cuboid']['lower'] + G.nodes[nb]['cuboid']['upper']) / 2.0
                d = np.linalg.norm(nb_center - current_center)
                if d <= radius:
                    candidates.append((nb, d))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[1])
        next_node = candidates[0][0]
        path.append(next_node)
        explored.add(next_node)
        current = next_node
    return path

if __name__ == "__main__":
    main()
