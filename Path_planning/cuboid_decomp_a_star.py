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
import random, heapq, os, pickle, time
from collections import deque
import networkx as nx
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
def cuboids_touch_or_overlap(c1, c2, tol=0.0):
    for i in range(3):
        if c1['upper'][i] < c2['lower'][i] - tol or c2['upper'][i] < c1['lower'][i] - tol:
            return False
    return True

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

#############################################
# 7. BFS Corridor on the Cuboid Graph
#############################################
def bfs_cuboid_corridor(cuboids, start_idx, goal_idx):
    """Return a list of cuboid indices forming a corridor from start_idx to goal_idx."""
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
# 8. 3D Corridor Grid + A* in 3D
#############################################
def create_3d_corridor_grid(cuboids, corridor_indices, grid_resolution=0.1):
    xs, ys, zs = [], [], []
    for idx in corridor_indices:
        c = cuboids[idx]
        xs.extend([c['lower'][0], c['upper'][0]])
        ys.extend([c['lower'][1], c['upper'][1]])
        zs.extend([c['lower'][2], c['upper'][2]])
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    width = max_x - min_x
    height = max_y - min_y
    depth = max_z - min_z
    nx = int(np.ceil(width / grid_resolution))
    ny = int(np.ceil(height / grid_resolution))
    nz = int(np.ceil(depth / grid_resolution))
    grid = np.zeros((nz, ny, nx), dtype=np.uint8)  # grid[z,y,x]

    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                x = min_x + (k + 0.5)*grid_resolution
                y = min_y + (j + 0.5)*grid_resolution
                z = min_z + (i + 0.5)*grid_resolution
                free = False
                for idx in corridor_indices:
                    c = cuboids[idx]
                    if (x >= c['lower'][0] and x <= c['upper'][0] and
                        y >= c['lower'][1] and y <= c['upper'][1] and
                        z >= c['lower'][2] and z <= c['upper'][2]):
                        free = True
                        break
                grid[i, j, k] = 1 if free else 0

    return grid, min_x, min_y, min_z, nx, ny, nz

def astar_grid_3d(grid, start_idx, goal_idx):
    """26-connected 3D A* on a voxel grid. Returns (path_of_indices, expansions, solution_nodes)."""
    moves = []
    for di in [-1,0,1]:
        for dj in [-1,0,1]:
            for dk in [-1,0,1]:
                if di==0 and dj==0 and dk==0:
                    continue
                moves.append((di, dj, dk))
    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))
    
    open_list = []
    visited = set()
    expansions = 0
    best_cost = {}
    g0 = 0
    f0 = heuristic(start_idx, goal_idx)
    best_cost[start_idx] = 0
    heapq.heappush(open_list, (f0, g0, start_idx, [start_idx]))
    
    while open_list:
        open_list.sort(key=lambda x: x[0])
        f, g, current, path = open_list.pop(0)
        expansions += 1
        if current == goal_idx:
            return path, expansions, len(path)
        if current in visited:
            continue
        visited.add(current)
        for (di, dj, dk) in moves:
            ni = current[0] + di
            nj = current[1] + dj
            nk = current[2] + dk
            if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1] and 0 <= nk < grid.shape[2]:
                if grid[ni, nj, nk] == 1:
                    cost_new = g + np.linalg.norm(np.array((ni,nj,nk)) - np.array(current))
                    if (ni, nj, nk) not in best_cost or cost_new < best_cost[(ni,nj,nk)]:
                        best_cost[(ni,nj,nk)] = cost_new
                        fn = cost_new + heuristic((ni,nj,nk), goal_idx)
                        open_list.append((fn, cost_new, (ni,nj,nk), path + [(ni,nj,nk)]))
    return None, expansions, 0

def astar_in_corridor_3d(cuboids, corridor_indices, start_xyz, goal_xyz, grid_res=0.1):
    """Build a local 3D grid for corridor_indices, run 3D A*, return path + metrics."""
    grid, min_x, min_y, min_z, nx, ny, nz = create_3d_corridor_grid(cuboids, corridor_indices, grid_res)
    sx = (start_xyz[0] - min_x) / grid_res
    sy = (start_xyz[1] - min_y) / grid_res
    sz = (start_xyz[2] - min_z) / grid_res
    gx = (goal_xyz[0] - min_x) / grid_res
    gy = (goal_xyz[1] - min_y) / grid_res
    gz = (goal_xyz[2] - min_z) / grid_res
    start_idx = (int(sz), int(sy), int(sx))
    goal_idx  = (int(gz), int(gy), int(gx))

    path_idx, expansions, sol_nodes = astar_grid_3d(grid, start_idx, goal_idx)
    if path_idx is None:
        return None

    # Convert path of grid indices back to world coords
    path_world = []
    for (iz, iy, ix) in path_idx:
        wx = min_x + (ix + 0.5)*grid_res
        wy = min_y + (iy + 0.5)*grid_res
        wz = min_z + (iz + 0.5)*grid_res
        path_world.append([wx, wy, wz])

    # Compute path length in 3D
    path_length = 0.0
    for i in range(len(path_world)-1):
        path_length += np.linalg.norm(np.array(path_world[i+1]) - np.array(path_world[i]))

    return (path_world, expansions, path_length, sol_nodes, start_idx, goal_idx)

#############################################
# 9. Update Connectivity for Cuboids
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
# 10. Find Nearest Cuboid for a Given Waypoint
#############################################
def find_nearest_cuboid_index(waypoint, cuboids):
    min_dist = float("inf")
    best_idx = None
    for i, cub in enumerate(cuboids):
        # Check if the waypoint lies within the cuboid
        if np.all(waypoint >= cub['lower']) and np.all(waypoint <= cub['upper']):
            return i
        # Otherwise measure distance to center
        center = (cub['lower'] + cub['upper']) / 2.0
        dist = np.linalg.norm(waypoint - center)
        if dist < min_dist:
            min_dist = dist
            best_idx = i
    return best_idx

#############################################
# 11. Straight-Line Check in 3D Corridor
#############################################
def is_line_in_corridor_3d(cuboids, corridor_indices, start_pt, goal_pt, num_samples=50):
    """Check if the line from start_pt to goal_pt is fully inside corridor."""
    for t in np.linspace(0, 1, num_samples):
        pt = start_pt + t*(goal_pt - start_pt)
        in_corridor = False
        for idx in corridor_indices:
            c = cuboids[idx]
            if (pt[0] >= c['lower'][0] and pt[0] <= c['upper'][0] and
                pt[1] >= c['lower'][1] and pt[1] <= c['upper'][1] and
                pt[2] >= c['lower'][2] and pt[2] <= c['upper'][2]):
                in_corridor = True
                break
        if not in_corridor:
            return False
    return True

def linear_path_3d(start_pt, goal_pt, num_points=20):
    return [start_pt + t*(goal_pt - start_pt) for t in np.linspace(0, 1, num_points)]

#############################################
# 12. RVizPublisher
#############################################
class RVizPublisher(Node):
    def __init__(self, points, cuboids, corridor_indices, path_coords_3d, frame_id="map", drone_mesh_resource=None):
        super().__init__('rviz_cuboid_astar')
        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.pc_pub = self.create_publisher(PointCloud2, '/point_cloud', qos)
        self.marker_pub_all = self.create_publisher(MarkerArray, '/free_cuboids', qos)
        self.marker_pub_corridor = self.create_publisher(MarkerArray, '/corridor_cuboids', qos)
        self.line_path_pub = self.create_publisher(Marker, '/line_path', qos)
        self.path_pub = self.create_publisher(Path, '/planned_path', qos)
        self.drone_pub = self.create_publisher(Marker, '/drone_mesh', qos)
        
        self.points = points
        self.cuboids = cuboids
        self.corridor_indices = corridor_indices
        self.path_coords = path_coords_3d  # final 3D path
        self.frame_id = frame_id
        
        self.drone_mesh_resource = drone_mesh_resource or ""
        
        self.current_segment_index = 0
        self.alpha = 0.0
        self.alpha_increment = 0.02
        
        self.timer = self.create_timer(0.1, self.publish_all)
        self.get_logger().info("Cuboid + A* Node initialized.")
    
    def publish_all(self):
        now = self.get_clock().now().to_msg()
        self.publish_point_cloud(now)
        self.publish_all_cuboids(now)
        self.publish_corridor_cuboids(now)
        self.publish_line_path(now)
        self.publish_path(now)
        self.publish_drone_marker(now)
        self.get_logger().info("Published point cloud, cuboids, path, etc.")
    
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
            center = (cub['lower'] + cub['upper']) / 2.0
            dims = cub['dimensions_world']
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
            m.color.a = 0.2
            marker_array.markers.append(m)
        self.marker_pub_all.publish(marker_array)
    
    def publish_corridor_cuboids(self, stamp):
        marker_array = MarkerArray()
        mid = 0
        for idx in self.corridor_indices:
            cub = self.cuboids[idx]
            center = (cub['lower'] + cub['upper']) / 2.0
            dims = cub['dimensions_world']
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = stamp
            m.ns = "corridor_cuboids"
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
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 0.3
            marker_array.markers.append(m)
        self.marker_pub_corridor.publish(marker_array)
    
    def publish_line_path(self, stamp):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.ns = "line_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.2
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
    
    def publish_path(self, stamp):
        path_msg = Path()
        path_msg.header.stamp = stamp
        path_msg.header.frame_id = self.frame_id
        for pt in self.path_coords:
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = self.frame_id
            pose.pose.position.x = float(pt[0])
            pose.pose.position.y = float(pt[1])
            pose.pose.position.z = float(pt[2])
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)
    
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

#############################################
# 13. MAIN
#############################################
def main(args=None):
    rclpy.init(args=args)
    
    # Load point cloud
    pc_file = "/home/raghuram/ARPL/cuboid_decomp/cuboid_decomp-/pointcloud/pointcloud_gq/point_cloud_gq.npy"
    points = np.load(pc_file)
    if points.dtype.names is not None:
        points = np.vstack([points[name] for name in ('x','y','z')]).T
    
    # Filter out below z=0 if desired
    min_point_z = 0.0
    points = points[points[:,2] >= min_point_z]
    
    global_min = np.min(points, axis=0)
    global_max = np.max(points, axis=0)
    print("Point Cloud Bounding Box:")
    print("  Min:", global_min)
    print("  Max:", global_max)
    
    # Build occupancy grid
    resolution = 0.2
    occupancy = build_occupancy_grid(points, global_min, resolution)
    print(f"Occupancy grid shape: {occupancy.shape}, Occupied voxels: {occupancy.sum()}")
    
    # Expand obstacles
    safety_voxels = 2
    occupancy_expanded = expand_obstacles_3d(occupancy, safety_voxels)
    
    # Region growing for free cuboids
    max_z_thickness = 20
    blocks = region_growing_3d(occupancy_expanded, max_z_thickness)
    print("Found", len(blocks), "free cuboids.")
    
    # Build or load cuboids
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
    print(f"Cuboid decomposition time: {t1_cub - t0_cub:.3f} seconds")
    print(f"Total cuboids loaded: {len(cuboids)}")
    
    # Update connectivity
    update_connectivity(cuboids, step=0.05, tol=0.0)
    
    # Plan successive subpaths between random cuboid indices
    num_segments = 5
    metrics = []
    overall_3d_path = []
    
    current_start = random.randint(0, len(cuboids)-1)
    current_goal = random.randint(0, len(cuboids)-1)
    while current_goal == current_start:
        current_goal = random.randint(0, len(cuboids)-1)
    
    def center_of_cuboid(cub):
        return (cub['lower'] + cub['upper']) / 2.0
    
    for seg in range(num_segments):
        print(f"\nSegment {seg+1}: start={current_start}, goal={current_goal}")
        # BFS corridor
        corridor = bfs_cuboid_corridor(cuboids, current_start, current_goal)
        start_time = time.time()
        if corridor is None:
            print("No corridor found!")
            compute_time = time.time() - start_time
            metrics.append((seg+1, current_start, current_goal, compute_time, 0.0))
        else:
            # Straight-line check from center of start cuboid to center of goal cuboid
            c_s = center_of_cuboid(cuboids[current_start])
            c_g = center_of_cuboid(cuboids[current_goal])
            if is_line_in_corridor_3d(cuboids, corridor, c_s, c_g, num_samples=50):
                # Use linear interpolation
                expansions = 0
                sol_nodes = 2
                path_length = np.linalg.norm(c_g - c_s)
                path_world = linear_path_3d(c_s, c_g, num_points=20)
                compute_time = time.time() - start_time
                print(f"Straight-line used, length={path_length:.3f}m, time={compute_time:.3f}s")
            else:
                # 3D corridor A*
                print("Line check failed; running 3D A* in corridor.")
                result = astar_in_corridor_3d(cuboids, corridor, c_s, c_g, grid_res=0.1)
                compute_time = time.time() - start_time
                if result is None:
                    print("No fine 3D path found in corridor!")
                    expansions = 0
                    sol_nodes = 0
                    path_length = 0.0
                    path_world = []
                else:
                    (path_world, expansions, path_length, sol_nodes, startG, goalG) = result
                    print(f"3D A* path found: length={path_length:.3f}m, expansions={expansions}, time={compute_time:.3f}s")
            
            # Add final path to overall
            if seg == 0:
                overall_3d_path.extend(path_world)
            else:
                overall_3d_path.extend(path_world[1:])
            
            metrics.append((seg+1, current_start, current_goal, compute_time, path_length))
        
        # Next segment
        current_start = current_goal
        current_goal = random.randint(0, len(cuboids)-1)
        while current_goal == current_start:
            current_goal = random.randint(0, len(cuboids)-1)
    
    # Write metrics to file
    with open("metrics.txt", "w") as f:
        f.write("Segment\tStartCuboid\tGoalCuboid\tComputeTime(s)\tPathLength(m)\n")
        for seg, s, g, t, L in metrics:
            f.write(f"{seg}\t{s}\t{g}\t{t:.3f}\t{L:.3f}\n")
    print("Metrics saved to metrics.txt")
    
    # Initialize RViz with final path
    # For corridor visualization, we can just union all corridors used
    # but here we skip that for brevity. We'll pass an empty corridor or a random set
    corridor_indices_list = []
    # We pass the final 3D path to the publisher
    drone_mesh_resource = "file:///home/raghuram/ARPL/cuboid_decomp/cuboid_decomp-/simulator/meshes/race2.stl"
    publisher_node = RVizPublisher(
        points=points,
        cuboids=cuboids,
        corridor_indices=corridor_indices_list,
        path_coords_3d=overall_3d_path,
        frame_id="map",
        drone_mesh_resource=drone_mesh_resource
    )
    try:
        rclpy.spin(publisher_node)
    except KeyboardInterrupt:
        pass
    finally:
        publisher_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
