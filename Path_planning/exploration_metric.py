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
import random, heapq, os, pickle, time, json, csv
from collections import deque
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
# Load Waypoints from TXT File
#############################################
def load_waypoints_txt(filename):
    """
    Expects a text file where:
      - The first line is the number of waypoints.
      - Each waypoint is represented on 7 separate lines:
          x, y, z, qx, qy, qz, qw.
    Only x, y, z are returned.
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    count = int(lines[0].strip())
    waypoints = []
    idx = 1
    for _ in range(count):
        # Each waypoint has 7 numbers; take only the first 3 (x, y, z)
        vals = [float(lines[idx + j].strip()) for j in range(7)]
        idx += 7
        waypoints.append(np.array(vals[:3]))
    return waypoints

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
# 5. Strict Line-of-Sight Checking
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
# 7. A* Path Planning on Cuboids (Connectivity Graph)
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
        f, g, current, path = heapq.heappop(open_set)
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
                heapq.heappush(open_set, (f_new, cost_new, nb, path + [nb]))
    return None

#############################################
# 8. Update Connectivity for Cuboids
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
# 9. Find Nearest Cuboid for a Given Waypoint
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
# 10. Compute Direct Path via Intersection Midpoints
#############################################
def compute_direct_path(cuboids, connectivity_path, start_waypoint, end_waypoint):
    """
    Given a connectivity path (list of cuboid indices), compute a direct path
    by taking the start waypoint, then for each adjacent pair of cuboids, compute
    the midpoint of their intersection (if they intersect; else use the center of the first),
    and finally the end waypoint.
    All points are converted to plain Python lists.
    """
    def to_list(pt):
        return pt.tolist() if isinstance(pt, np.ndarray) else list(pt)
    
    path = [to_list(start_waypoint)]
    for i in range(len(connectivity_path)-1):
        cub1 = cuboids[connectivity_path[i]]
        cub2 = cuboids[connectivity_path[i+1]]
        lower = np.maximum(cub1['lower'], cub2['lower'])
        upper = np.minimum(cub1['upper'], cub2['upper'])
        if np.all(lower <= upper):
            midpoint = (lower + upper) / 2.0
        else:
            midpoint = (cub1['lower'] + cub1['upper']) / 2.0
        path.append(to_list(midpoint))
    path.append(to_list(end_waypoint))
    return path

#############################################
# 11. RVizPublisher Node (Direct Path, Waypoints, Used Cuboids, and Free Cuboids)
#############################################
class RVizPublisher(Node):
    def __init__(self, points, cuboids, path_coords, waypoints, path_cuboids_indices, frame_id="map", drone_mesh_resource=None):
        super().__init__('rviz_cuboid_direct_path')
        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.pc_pub = self.create_publisher(PointCloud2, '/point_cloud', qos)
        self.marker_pub_all = self.create_publisher(MarkerArray, '/free_cuboids', qos)
        self.marker_pub_path_cuboids = self.create_publisher(MarkerArray, '/path_cuboids', qos)
        self.line_path_pub = self.create_publisher(Marker, '/line_path', qos)
        self.path_pub = self.create_publisher(Path, '/planned_path', qos)
        self.waypoint_pub = self.create_publisher(MarkerArray, '/waypoints', qos)
        self.drone_pub = self.create_publisher(Marker, '/drone_mesh', qos)
        
        self.points = points
        self.cuboids = cuboids
        self.path_coords = path_coords  # Direct 3D coordinates (including midpoints)
        self.waypoints = waypoints      # Loaded waypoints from file
        self.path_cuboids_indices = path_cuboids_indices  # List of cuboid indices used for planning
        self.frame_id = frame_id
        self.drone_mesh_resource = drone_mesh_resource or ""
        
        # For drone motion interpolation along the direct path
        self.current_segment_index = 0
        self.alpha = 0.0
        self.alpha_increment = 0.02
        
        self.timer = self.create_timer(0.1, self.publish_all)
        self.get_logger().info("[INFO] Cuboid Direct-Path Node initialized.")
    
    def publish_all(self):
        now = self.get_clock().now().to_msg()
        self.publish_point_cloud(now)
        self.publish_all_cuboids(now)
        self.publish_path_cuboids(now)
        self.publish_line_path(now)
        self.publish_path(now)
        self.publish_waypoints(now)
        self.publish_drone_marker(now)
        self.get_logger().info("[INFO] Published point cloud, cuboids, path, used cuboids, waypoints, and drone mesh.")
    
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
            m.color.a = 0.3
            marker_array.markers.append(m)
        self.marker_pub_all.publish(marker_array)
    
    def publish_path_cuboids(self, stamp):
        marker_array = MarkerArray()
        mid = 0
        for idx in self.path_cuboids_indices:
            cub = self.cuboids[idx]
            center = (cub['lower'] + cub['upper']) / 2.0
            dims = cub['dimensions_world']
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = stamp
            m.ns = "path_cuboids"
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
            # Use a distinct color (blue)
            m.color.r = 1.0
            m.color.g = 1.0
            m.color.b = 1.0
            m.color.a = 0.5
            marker_array.markers.append(m)
        self.marker_pub_path_cuboids.publish(marker_array)
    
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
        marker.color.g = 1.0
        marker.color.b = 0.0
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
    
    def publish_waypoints(self, stamp):
        marker_array = MarkerArray()
        for i, wp in enumerate(self.waypoints):
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = stamp
            m.ns = "waypoints"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(wp[0])
            m.pose.position.y = float(wp[1])
            m.pose.position.z = float(wp[2])
            m.pose.orientation.w = 1.0
            m.scale.x = 1.5
            m.scale.y = 1.5
            m.scale.z = 1.5
            m.color.r = 1.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 1.0
            marker_array.markers.append(m)
        self.waypoint_pub.publish(marker_array)
    
    def publish_drone_marker(self, stamp):
        if len(self.path_coords) < 2:
            pos = self.path_coords[0]
        else:
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
# 12. Main Routine: Cuboid Decomp + Direct Path via Waypoints from File
#############################################
def main(args=None):
    rclpy.init(args=args)
    
    # CONFIG: Adjust these parameters as needed
    CONFIG = {
        "POINT_CLOUD_FILE": "../pointcloud/pointcloud_minco/point_cloud_minco.npy",
        "MIN_POINT_Z": -1.0,
        "MAX_POINT_Z": 100.0, 
        "RESOLUTION": 0.2,
        "SAFETY_VOXELS": 1,
        "MAX_Z_THICKNESS": 1,
        "CUBOIDS_FILE": "my_cuboids_minco.pkl",
        "WAYPOINTS_FILE": "../waypoints/waypoints_minco.txt",  # New: file containing waypoints
        "DRONE_MESH_RESOURCE": "file://../simulator/meshes/race2.stl"
    }
    
    # Load point cloud
    points = np.load(CONFIG["POINT_CLOUD_FILE"])
    if points.dtype.names is not None:
        points = np.vstack([points[name] for name in ('x', 'y', 'z')]).T

    z_min = CONFIG["MIN_POINT_Z"]
    z_max = CONFIG["MAX_POINT_Z"]
    points = points[(points[:,2] >= z_min) & (points[:,2] <= z_max)]
    
    global_min = np.min(points, axis=0)
    global_max = np.max(points, axis=0)
    print(f"[INFO] Point Cloud Bounding Box:")
    print(f"[INFO]   Min: {global_min}")
    print(f"[INFO]   Max: {global_max}")
    
    # Record size of the point cloud for metrics
    size_point_cloud = points.shape[0]
    
    # Start cuboid decomposition timer
    start_decomp = time.time()
    
    # Build occupancy grid and expand obstacles
    occupancy = build_occupancy_grid(points, global_min, CONFIG["RESOLUTION"])
    print(f"[INFO] Occupancy grid shape: {occupancy.shape}, Occupied voxels: {occupancy.sum()}")
    occupancy_expanded = expand_obstacles_3d(occupancy, CONFIG["SAFETY_VOXELS"])
    
    # Extract free cuboids via region growing
    blocks = region_growing_3d(occupancy_expanded, CONFIG["MAX_Z_THICKNESS"])
    print(f"[INFO] Found {len(blocks)} free cuboids.")
    
    # Load or build cuboids with connectivity
    cuboids_file = CONFIG["CUBOIDS_FILE"]
    if os.path.exists(cuboids_file):
        print(f"[INFO] Loading cuboids from {cuboids_file}...")
        cuboids = load_cuboids(cuboids_file)
    else:
        print(f"[INFO] Cuboids file not found. Building cuboids with connectivity...")
        cuboids = build_cuboids_with_connectivity(blocks, global_min, CONFIG["RESOLUTION"], step=0.05)
        for c in cuboids:
            c['dimensions_world'] = c['upper'] - c['lower']
        save_cuboids(cuboids, cuboids_file)
        print(f"[INFO] Saved cuboids to {cuboids_file}")
    print(f"[INFO] Total cuboids loaded: {len(cuboids)}")
    
    update_connectivity(cuboids, step=0.05, tol=0.0)
    
    # End cuboid decomp timer
    total_decomp_time = time.time() - start_decomp
    
    # Load waypoints from the specified txt file
    waypoints = load_waypoints_txt(CONFIG["WAYPOINTS_FILE"])
    print(f"[INFO] Loaded {len(waypoints)} waypoints from {CONFIG['WAYPOINTS_FILE']}.")
    
    overall_path_coords = []  # Complete direct path (list of 3D points)
    used_path_indices = set()  # To accumulate cuboid indices used in planning
    total_connectivity_time = 0.0  # Sum of A* planning times per segment
    total_path_length = 0.0  # Sum of lengths of all planned segments
    
    # For each consecutive pair of waypoints, plan a segment
    for i in range(len(waypoints)-1):
        start_wp = waypoints[i]
        end_wp = waypoints[i+1]
        start_idx = find_nearest_cuboid_index(start_wp, cuboids)
        end_idx = find_nearest_cuboid_index(end_wp, cuboids)
        print(f"[INFO] Planning segment from waypoint {i+1} (cuboid {start_idx}) to waypoint {i+2} (cuboid {end_idx})...")
        seg_start_time = time.time()
        connectivity_path = astar_on_cuboids(cuboids, start_idx, end_idx)
        if connectivity_path is None:
            print(f"[INFO] No connectivity path found for segment {i+1}.")
            continue
        seg_time = time.time() - seg_start_time
        total_connectivity_time += seg_time
        
        used_path_indices.update(connectivity_path)
        direct_path_segment = compute_direct_path(cuboids, connectivity_path, start_wp, end_wp)
        seg_length = sum(np.linalg.norm(np.array(direct_path_segment[j+1]) - np.array(direct_path_segment[j]))
                         for j in range(len(direct_path_segment)-1))
        total_path_length += seg_length
        
        if i == 0:
            overall_path_coords.extend(direct_path_segment)
        else:
            overall_path_coords.extend(direct_path_segment[1:])
        print(f"[INFO] Segment {i+1}: time: {seg_time:.3f}s, length: {seg_length:.3f}m")
    
    # Compute graph size: count of edges (each edge counted once)
    graph_size = sum(len(cub['neighbors']) for cub in cuboids) // 2
    
    # Save overall metrics to CSV
    with open("metrics_cmu_new.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Size of point cloud", "number of cuboids", "number of cuboids for the path", 
                         "Total Cuboid Decomp time", "Path cuboid Connective time", "Path Length", "graph size"])
        writer.writerow([size_point_cloud, len(cuboids), len(used_path_indices),
                         total_decomp_time, total_connectivity_time, total_path_length, graph_size])
    print(f"[INFO] Metrics saved to metrics.csv")
    
    # Create ROS2 publisher node for visualization,
    # passing overall direct path, waypoints, and the cuboids used for planning.
    publisher_node = RVizPublisher(points, cuboids, overall_path_coords, waypoints, list(used_path_indices),
                                   frame_id="map", drone_mesh_resource=CONFIG["DRONE_MESH_RESOURCE"])
    try:
        rclpy.spin(publisher_node)
    except KeyboardInterrupt:
        pass
    finally:
        publisher_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
