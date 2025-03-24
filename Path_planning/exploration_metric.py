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

from tqdm import tqdm
import networkx as nx

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
# 7. A* Path Planning on Cuboids Using Their Connectivity
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
# 9. ROS2 Publisher Node for Cuboid Decomp + A*
#############################################
class RVizPublisher(Node):
    def __init__(self, points, cuboids, path_indices, drone_mesh_resource, frame_id="map"):
        super().__init__('rviz_cuboid_astar')
        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.pc_pub = self.create_publisher(PointCloud2, '/point_cloud', qos)
        self.marker_pub_all = self.create_publisher(MarkerArray, '/free_cuboids', qos)
        self.marker_pub_path = self.create_publisher(MarkerArray, '/path_cuboids', qos)
        self.line_path_pub = self.create_publisher(Marker, '/line_path', qos)
        self.path_pub = self.create_publisher(Path, '/planned_path', qos)
        self.drone_pub = self.create_publisher(Marker, '/drone_mesh', qos)
        
        self.points = points
        self.cuboids = cuboids
        self.path_indices = path_indices  # Direct path (cuboid indices)
        self.frame_id = frame_id
        
        self.path_coords = []
        for idx in path_indices:
            c = cuboids[idx]
            center = (c['lower'] + c['upper']) / 2.0
            self.path_coords.append(center)
        
        self.drone_mesh_resource = drone_mesh_resource  # e.g., "file:///path/to/mesh.stl"
        # For continuous drone motion, use segment interpolation:
        self.current_segment_index = 0
        self.alpha = 0.0
        self.alpha_increment = 0.02
        
        self.timer = self.create_timer(0.1, self.publish_all)
        self.get_logger().info("Cuboid + A* Node initialized.")
    
    def publish_all(self):
        now = self.get_clock().now().to_msg()
        self.publish_point_cloud(now)
        self.publish_all_cuboids(now)
        self.publish_path_cuboids(now)
        self.publish_line_path(now)
        self.publish_path(now)
        self.publish_drone_marker(now)
        self.get_logger().info("Published point cloud, cuboids, path, and drone mesh.")
    
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
        for idx in self.path_indices:
            cub = self.cuboids[idx]
            lower = cub['lower']
            upper = cub['upper']
            dims = cub['dimensions_world']
            center = (lower + upper) / 2.0
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
            m.color.r = 0.0
            m.color.g = 0.5
            m.color.b = 0.0
            m.color.a = 0.5
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
        marker.scale.x = 0.2
        marker.color.r = 1.0
        marker.color.g = 0.0
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
# 10. Main Routine: Cuboid Decomp + A* with Successive Waypoints
#############################################
def main(args=None):
    rclpy.init(args=args)
    
    # Load point cloud (Nx3 numpy array)
    pc_file = "/home/raghuram/ARPL/cuboid_decomp/cuboid_decomp-/pointcloud/pointcloud_gq/point_cloud_gq.npy"  # update as needed
    points = np.load(pc_file)
    if points.dtype.names is not None:
        points = np.vstack([points[name] for name in ('x','y','z')]).T

    # Filter point cloud to only include points with z >= 0
    min_point_z = -1.0  
    points = points[points[:,2] >= min_point_z]
    
    # Compute bounding box (of filtered points)
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
    
    # Load or build cuboids with connectivity and save them
    cuboids_file = "my_cuboids_hos.pkl"
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
    
    print(f"Total cuboids loaded: {len(cuboids)}")
    
    # Update connectivity for cuboids
    update_connectivity(cuboids, step=0.05, tol=0.0)
    
    # --- Sampling Waypoints from Free Space ---
    # Get free cell indices from occupancy grid (cells with 0)
    free_indices = np.argwhere(occupancy == 0)
    # Randomly sample 5 free cell indices
    if len(free_indices) < 5:
        raise ValueError("Not enough free cells available!")
    sampled_idx = free_indices[random.sample(range(len(free_indices)), 5)]
    # Convert these grid indices to world coordinates (center of voxel)
    def grid_to_world(idx, global_min, resolution):
        return global_min + (np.array(idx) + 0.5) * resolution
    waypoint_points = [grid_to_world(idx, global_min, resolution) for idx in sampled_idx]
    print("Random free-space waypoint coordinates:")
    for pt in waypoint_points:
        print(pt)
    
    # For each waypoint, find the nearest cuboid center (as our planning node)
    waypoint_cuboid_indices = []
    for pt in waypoint_points:
        best_idx = None
        best_dist = float("inf")
        for i, cub in enumerate(cuboids):
            center = (cub['lower'] + cub['upper']) / 2.0
            d = np.linalg.norm(center - pt)
            if d < best_dist:
                best_dist = d
                best_idx = i
        waypoint_cuboid_indices.append(best_idx)
    print("Waypoints (cuboid indices):", waypoint_cuboid_indices)
    
    # Now, plan successive subpaths between the 5 waypoints using A* on cuboids.
    num_segments = len(waypoint_cuboid_indices) - 1
    overall_path = []
    metrics = []  # (segment, start_idx, goal_idx, compute_time, path_length, start_point, goal_point)
    current_start = waypoint_cuboid_indices[0]
    for seg in range(num_segments):
        current_goal = waypoint_cuboid_indices[seg+1]
        start_time = time.time()
        subpath = astar_on_cuboids(cuboids, current_start, current_goal)
        compute_time = time.time() - start_time
        if subpath is None:
            print(f"Segment {seg+1}: No path found from {current_start} to {current_goal}")
            continue
        # Compute path length (sum of distances between consecutive cuboid centers)
        path_length = 0.0
        for i in range(len(subpath)-1):
            c1 = (cuboids[subpath[i]]['lower'] + cuboids[subpath[i]]['upper']) / 2.0
            c2 = (cuboids[subpath[i+1]]['lower'] + cuboids[subpath[i+1]]['upper']) / 2.0
            path_length += np.linalg.norm(c2 - c1)
        metrics.append((seg+1, current_start, current_goal, compute_time, path_length, 
                        waypoint_points[seg], waypoint_points[seg+1]))
        # For overall path, concatenate subpath (avoiding duplicate node at junction)
        overall_path.extend(subpath if seg == 0 else subpath[1:])
        current_start = current_goal
        print(f"Segment {seg+1}: from {current_start} to {current_goal}, time: {compute_time:.3f}s, length: {path_length:.3f}m")
    
    # Save metrics to file
    with open("metrics.txt", "w") as f:
        f.write("Segment\tStartCuboid\tGoalCuboid\tComputeTime(s)\tPathLength(m)\tStartPoint\tGoalPoint\n")
        for seg, s, g, t, L, sp, gp in metrics:
            sp_str = f"({sp[0]:.2f}, {sp[1]:.2f}, {sp[2]:.2f})"
            gp_str = f"({gp[0]:.2f}, {gp[1]:.2f}, {gp[2]:.2f})"
            f.write(f"{seg}\t{s}\t{g}\t{t:.3f}\t{L:.3f}\t{sp_str}\t{gp_str}\n")
    print("Metrics saved to metrics.txt")
    
    # Build overall path centers (list of cuboid centers)
    overall_path_coords = []
    for idx in overall_path:
        center = (cuboids[idx]['lower'] + cuboids[idx]['upper']) / 2.0
        overall_path_coords.append(center)
    
    # Define your drone mesh resource (using an STL file)
    drone_mesh_resource = "file:///home/raghuram/ARPL/cuboid_decomp/cuboid_decomp-/simulator/meshes/race2.stl"
    
    # Create ROS2 publisher node for visualization using the overall path
    publisher_node = RVizPublisher(points, cuboids, overall_path, drone_mesh_resource, frame_id="map")
    publisher_node.current_drone_index = 0
    publisher_node.current_segment_index = 0
    publisher_node.alpha = 0.0
    publisher_node.alpha_increment = 0.02  # Adjust for smooth motion
    
    try:
        rclpy.spin(publisher_node)
    except KeyboardInterrupt:
        pass
    finally:
        publisher_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
