#!/usr/bin/env python3
import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import sensor_msgs_py.point_cloud2 as pc2
import random
import heapq
import os
import pickle

# Additional imports for progress and graph plotting
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

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
    structure = generate_binary_structure(rank=3, connectivity=1)  # 6-connected 3D structure
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
                        # Expand in +x
                        if i_max < nx - 1:
                            candidate = occupancy[i_max+1, j_min:j_max+1, k_min:k_max+1]
                            if np.all(candidate == 0):
                                i_max += 1
                                changed = True
                        # Expand in +y
                        if j_max < ny - 1:
                            candidate = occupancy[i_min:i_max+1, j_max+1, k_min:k_max+1]
                            if np.all(candidate == 0):
                                j_max += 1
                                changed = True
                        # Expand in +z, but cap thickness
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
# 8. Build a Graph from Cuboids (for Exploration)
#############################################
def build_cuboid_graph_from_cuboids(cuboids, tol=0.0):
    G = nx.Graph()
    for i, cub in enumerate(cuboids):
        G.add_node(i, cuboid=cub)
        for nb in cub.get('neighbors', []):
            if not G.has_edge(i, nb):
                center_i = (cub['lower'] + cub['upper']) / 2.0
                center_j = (cuboids[nb]['lower'] + cuboids[nb]['upper']) / 2.0
                weight = np.linalg.norm(center_i - center_j)
                G.add_edge(i, nb, weight=weight)
    return G

#############################################
# 9. DFS-based Exploration Path Planning
#############################################
def build_exploration_path(G, exploration_factor, start_node=None):
    """
    Performs DFS from a start node (or random if None) on graph G.
    Then takes the first N nodes (N = exploration_factor * total nodes) from the DFS order.
    Returns a list of node indices representing the exploration path.
    """
    if start_node is None:
        start_node = random.choice(list(G.nodes))
    dfs_order = list(nx.dfs_preorder_nodes(G, source=start_node))
    target_count = max(1, int(exploration_factor * len(G.nodes)))
    exploration_path = dfs_order[:target_count]
    return exploration_path

#############################################
# 10. ROS2 Publisher Node for RViz2 Visualization + Path Publishing + Drone Mesh
#############################################
class RVizPublisher(Node):
    def __init__(self, points, cuboids, path_indices, exploration_path_coords, drone_mesh_resource, frame_id="map"):
        super().__init__('rviz_los_planner')
        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.pc_pub = self.create_publisher(PointCloud2, '/point_cloud', qos)
        self.marker_pub_all = self.create_publisher(MarkerArray, '/free_cuboids', qos)
        self.marker_pub_path = self.create_publisher(MarkerArray, '/path_cuboids', qos)
        self.path_pub = self.create_publisher(Path, '/planned_path', qos)
        self.exploration_path_pub = self.create_publisher(Path, '/exploration_path', qos)
        # New publisher for the drone mesh marker.
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
        
        self.exploration_path_coords = exploration_path_coords
        
        # NEW: Drone mesh resource URI and index for animation.
        self.drone_mesh_resource = drone_mesh_resource  # e.g. "package://my_package/meshes/quadrotor.dae"
        self.current_drone_index = 0
        
        self.timer = self.create_timer(1.0, self.publish_all)
        self.get_logger().info("LOS Planner Node initialized.")
    
    def publish_all(self):
        now = self.get_clock().now().to_msg()
        self.publish_point_cloud(now)
        self.publish_all_cuboids(now)
        self.publish_path_cuboids(now)
        self.publish_path(now)
        self.publish_exploration_path(now)
        self.publish_drone_marker(now)
        self.get_logger().info("Published point cloud, cuboid markers, paths, and drone mesh.")
    
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
            m.color.r = 0.9
            m.color.g = 0.1
            m.color.b = 0.1
            m.color.a = 0.8
            marker_array.markers.append(m)
        self.marker_pub_path.publish(marker_array)
    
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
        # Move the drone along the direct path (cycle through path_coords)
        if not self.path_coords:
            return
        pos = self.path_coords[self.current_drone_index]
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
        # Adjust scale as necessary for your drone mesh.
        m.scale.x = 1.0
        m.scale.y = 1.0
        m.scale.z = 1.0
        m.color.a = 1.0  # fully opaque
        self.drone_pub.publish(m)
        # Increment the drone index (cycle through the path)
        self.current_drone_index = (self.current_drone_index + 1) % len(self.path_coords)

#############################################
# 11. Main Routine
#############################################
def main(args=None):
    rclpy.init(args=args)
    
    # Load point cloud (Nx3 numpy array)
    pc_file = "/home/raghuram/ARPL/cuboid_decomp/cuboid_decomp-/pointcloud/pointcloud_gq/point_cloud_gq.npy"  # update as needed
    points = np.load(pc_file)
    if points.dtype.names is not None:
        points = np.vstack([points[name] for name in ('x','y','z')]).T
    
    # Compute bounding box
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
    max_z_thickness = 20  # adjust as needed
    blocks = region_growing_3d(occupancy_expanded, max_z_thickness)
    print("Found", len(blocks), "free cuboids.")
    
    # Load or build cuboids with connectivity and save them
    cuboids_file = "my_cuboids.pkl"
    if os.path.exists(cuboids_file):
        print(f"Loading cuboids from {cuboids_file}...")
        cuboids = load_cuboids(cuboids_file)
    else:
        print("Cuboids file not found. Building cuboids with connectivity...")
        cuboids = build_cuboids_with_connectivity(blocks, global_min, resolution, step=0.05)
        # Ensure each cuboid has the 'dimensions_world' field
        for c in cuboids:
            c['dimensions_world'] = c['upper'] - c['lower']
        save_cuboids(cuboids, cuboids_file)
        print(f"Saved cuboids to {cuboids_file}")
    
    print(f"Total cuboids loaded: {len(cuboids)}")
    
    # Filter cuboids to only consider those above a minimum z threshold
    min_z_threshold = 0.0  # adjust as needed
    filtered_cuboids = [cub for cub in cuboids if ((cub['lower'][2] + cub['upper'][2]) / 2.0) >= min_z_threshold]
    print(f"Cuboids after filtering with min_z >= {min_z_threshold}: {len(filtered_cuboids)}")
    cuboids = filtered_cuboids  # use filtered cuboids for planning
    
    # Build a graph from cuboids for exploration planning
    G_explore = build_cuboid_graph_from_cuboids(cuboids, tol=0.0)
    
    # For exploration, choose a random start (only one is needed)
    explore_start = random.choice(list(G_explore.nodes))
    exploration_nodes = build_exploration_path(G_explore, exploration_factor=0.7, start_node=explore_start)
    print("Exploration path (node indices):", exploration_nodes)
    
    exploration_path_coords = []
    for idx in exploration_nodes:
        cub = G_explore.nodes[idx]['cuboid']
        center_pt = (cub['lower'] + cub['upper']) / 2.0
        exploration_path_coords.append(center_pt)
    
    # For direct path planning, choose random start and goal
    start_idx = random.randint(0, len(cuboids)-1)
    goal_idx = random.randint(0, len(cuboids)-1)
    while goal_idx == start_idx:
        goal_idx = random.randint(0, len(cuboids)-1)
    print("Direct path: start =", start_idx, "goal =", goal_idx)
    
    direct_path = astar_on_cuboids(cuboids, start_idx, goal_idx)
    if direct_path is None:
        print("No direct path found!")
        direct_path = []
    else:
        print("Direct path (node indices):", direct_path)
    
    path_coords = []
    for idx in direct_path:
        cub = cuboids[idx]
        center_pt = (cub['lower'] + cub['upper']) / 2.0
        path_coords.append(center_pt)
    
    # Define your drone mesh resource (update the URI as needed)
    drone_mesh_resource = "/home/raghuram/ARPL/cuboid_decomp/cuboid_decomp-/simulator/meshes/race2.stl"
    
    # Publish to RViz: publish all cuboids, direct path cuboids, exploration path, and drone mesh.
    node = RVizPublisher(points, cuboids, direct_path, exploration_path_coords, drone_mesh_resource, frame_id="map")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

def build_cuboid_graph_from_cuboids(cuboids, tol=0.0):
    G = nx.Graph()
    for i, cub in enumerate(cuboids):
        G.add_node(i, cuboid=cub)
        for nb in cub.get('neighbors', []):
            if not G.has_edge(i, nb):
                center_i = (cub['lower'] + cub['upper']) / 2.0
                center_j = (cuboids[nb]['lower'] + cuboids[nb]['upper']) / 2.0
                weight = np.linalg.norm(center_i - center_j)
                G.add_edge(i, nb, weight=weight)
    return G

if __name__ == "__main__":
    main()
