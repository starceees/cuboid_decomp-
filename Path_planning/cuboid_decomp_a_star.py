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

# Additional imports for graph + plotting
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    nx = int(np.ceil(extent[0] / resolution))
    ny = int(np.ceil(extent[1] / resolution))
    nz = int(np.ceil(extent[2] / resolution))
    occupancy = np.zeros((nx, ny, nz), dtype=np.uint8)
    
    idxs = np.floor((points - global_min) / resolution).astype(int)
    idxs = np.clip(idxs, 0, [nx-1, ny-1, nz-1])
    for (ix, iy, iz) in idxs:
        occupancy[ix, iy, iz] = 1
    return occupancy

#############################################
# 3. Greedy 3D Region Growing for Free Cuboids 
#    with a fixed max_z_thickness
#############################################
def region_growing_3d(occupancy, max_z_thickness=5):
    """
    Greedily grows free cuboids in X/Y while capping the thickness in Z.
    occupancy: 3D numpy array (1=obstacle, 0=free).
    Returns a list of cuboid blocks (in grid indices) as dicts:
      { 'min_idx': np.array([i,j,k]), 'dimensions': np.array([dx,dy,dz]) }
    """
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
                    cuboid = {
                        'min_idx': np.array([i_min, j_min, k_min]),
                        'dimensions': np.array([i_max - i_min + 1, j_max - j_min + 1, k_max - k_min + 1])
                    }
                    cuboids.append(cuboid)
    return cuboids

#############################################
# 4. Convert Grid Cuboids to World Coordinates
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
# 5. Build Cuboids with Connectivity (Graph Built as We Go)
#############################################
def build_cuboids_with_connectivity(free_cuboids_blocks, global_min, resolution, tol=0.0):
    """
    Convert free cuboid blocks to world coordinate cuboids.
    As each cuboid is created, check for connectivity (touch/overlap)
    with previously created cuboids. Each cuboid gets a 'neighbors' list.
    """
    cuboids = []
    for block in tqdm(free_cuboids_blocks, desc="Building cuboid connectivity"):
        cub = block_to_world_cuboid(block, global_min, resolution)
        cub['neighbors'] = []
        for idx, existing in enumerate(cuboids):
            if cuboids_touch_or_overlap(cub, existing, tol):
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
# 6. A* Path Planning on Cuboids using Their Connectivity
#############################################
def astar_on_cuboids(cuboids, start_idx, goal_idx):
    """
    Run A* search over cuboids using each cuboid's 'neighbors' list.
    Returns a list of cuboid indices representing the path.
    """
    def center(cub):
        return (cub['lower'] + cub['upper']) / 2.0
    def heuristic(i, j):
        return np.linalg.norm(center(cuboids[i]) - center(cuboids[j]))
    
    open_set = []
    heapq.heappush(open_set, (heuristic(start_idx, goal_idx), 0, start_idx, [start_idx]))
    closed = set()
    best_cost = {start_idx: 0}
    
    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        if current == goal_idx:
            return path
        if current in closed:
            continue
        closed.add(current)
        for neighbor in cuboids[current].get('neighbors', []):
            if neighbor in closed:
                continue
            tentative_g = g + heuristic(current, neighbor)
            if neighbor not in best_cost or tentative_g < best_cost[neighbor]:
                best_cost[neighbor] = tentative_g
                f_val = tentative_g + heuristic(neighbor, goal_idx)
                heapq.heappush(open_set, (f_val, tentative_g, neighbor, path + [neighbor]))
    return None

#############################################
# 7. ROS2 Publisher Node for RViz2 Visualization + Path Publishing
#############################################
class RVizPublisher(Node):
    def __init__(self, points, cuboids, path_coords, path_cuboid_indices, frame_id="map"):
        super().__init__('rviz_region_growing_publisher')
        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.pc_pub = self.create_publisher(PointCloud2, '/point_cloud', qos)
        self.marker_pub_all = self.create_publisher(MarkerArray, '/free_cuboids', qos)
        self.marker_pub_path = self.create_publisher(MarkerArray, '/path_cuboids', qos)
        self.path_pub = self.create_publisher(Path, '/planned_path', qos)
        self.timer = self.create_timer(1.0, self.publish_all)
        
        self.points = points
        self.cuboids = cuboids
        self.path_coords = path_coords  # list of 3D points (cuboid centers for path)
        self.path_cuboid_indices = path_cuboid_indices  # indices of cuboids used in the planned path
        self.frame_id = frame_id
        
        self.get_logger().info("3D Region Growing Publisher initialized.")
    
    def publish_all(self):
        now = self.get_clock().now().to_msg()
        self.publish_point_cloud(now)
        self.publish_all_cuboid_markers(now)
        self.publish_path_cuboid_markers(now)
        self.publish_path(now)
        self.get_logger().info("Published point cloud, all cuboids, path cuboids, and path.")
    
    def publish_point_cloud(self, stamp):
        header = Header()
        header.stamp = stamp
        header.frame_id = self.frame_id
        pc_list = self.points.tolist()
        pc_msg = pc2.create_cloud_xyz32(header, pc_list)
        self.pc_pub.publish(pc_msg)
    
    def publish_all_cuboid_markers(self, stamp):
        marker_array = MarkerArray()
        marker_id = 0
        for cuboid in self.cuboids:
            lower = cuboid['lower']
            upper = cuboid['upper']
            dims = cuboid['dimensions_world']
            center = (lower + upper) / 2.0
            
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = stamp
            marker.ns = "free_cuboids"
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float(center[0])
            marker.pose.position.y = float(center[1])
            marker.pose.position.z = float(center[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = float(dims[0])
            marker.scale.y = float(dims[1])
            marker.scale.z = float(dims[2])
            marker.color.r = 0.2
            marker.color.g = 0.8
            marker.color.b = 0.2
            marker.color.a = 0.3
            marker_array.markers.append(marker)
        self.marker_pub_all.publish(marker_array)
    
    def publish_path_cuboid_markers(self, stamp):
        marker_array = MarkerArray()
        marker_id = 0
        # Only publish markers for cuboids in the planned path.
        for idx in self.path_cuboid_indices:
            cuboid = self.cuboids[idx]
            lower = cuboid['lower']
            upper = cuboid['upper']
            dims = cuboid['dimensions_world']
            center = (lower + upper) / 2.0
            
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = stamp
            marker.ns = "path_cuboids"
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float(center[0])
            marker.pose.position.y = float(center[1])
            marker.pose.position.z = float(center[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = float(dims[0])
            marker.scale.y = float(dims[1])
            marker.scale.z = float(dims[2])
            marker.color.r = 0.9
            marker.color.g = 0.1
            marker.color.b = 0.1
            marker.color.a = 0.8
            marker_array.markers.append(marker)
        self.marker_pub_path.publish(marker_array)
    
    def publish_path(self, stamp):
        path_msg = Path()
        path_msg.header.stamp = stamp
        path_msg.header.frame_id = self.frame_id
        for point in self.path_coords:
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = self.frame_id
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = float(point[2])
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

#############################################
# 8. (Optional) Plot and Save Cuboid Graph
#############################################
def plot_and_save_cuboid_graph(cuboids, out_file="cuboid_graph.png"):
    # Build a graph from cuboids using their stored neighbor info.
    G = nx.Graph()
    for i, cub in enumerate(cuboids):
        G.add_node(i, cuboid=cub)
        for nb in cub.get('neighbors', []):
            if not G.has_edge(i, nb):
                center_i = (cub['lower'] + cub['upper']) / 2.0
                center_j = (cuboids[nb]['lower'] + cuboids[nb]['upper']) / 2.0
                weight = np.linalg.norm(center_i - center_j)
                G.add_edge(i, nb, weight=weight)
    pos = nx.spring_layout(G, seed=42)
    labels = {}
    for node in G.nodes(data=True):
        idx = node[0]
        cub = node[1]['cuboid']
        lower = cub['lower'].round(1)
        dims = cub['dimensions_world'].round(1)
        labels[idx] = f"({tuple(lower)}, {tuple(dims)})"
    plt.figure(figsize=(8,6))
    nx.draw(G, pos, with_labels=False, node_color="steelblue", node_size=600, edge_color="black")
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    plt.title("Cuboid Connectivity Graph")
    plt.axis('equal')
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved cuboid connectivity graph to {out_file}")

#############################################
# 9. Metrics Saving Function
#############################################
def save_metrics(metrics, filename="metrics.txt"):
    with open(filename, "w") as f:
        f.write("Attempt\tStartCuboid\tGoalCuboid\tComputeTime(s)\tPathLength(m)\tSolutionNodes\n")
        for m in metrics:
            attempt, start_idx, goal_idx, comp_time, path_length, sol_nodes = m
            f.write(f"{attempt}\t{start_idx}\t{goal_idx}\t{comp_time:.3f}\t{path_length:.3f}\t{sol_nodes}\n")
    print("Metrics saved to", filename)

#############################################
# 10. Main Routine
#############################################
def main(args=None):
    rclpy.init(args=args)
    
    # Load point cloud (Nx3 numpy array)
    pc_file = "/home/raghuram/ARPL/cuboid_decomp/cuboid_decomp-/pointcloud/pointcloud_gq/point_cloud_gq.npy"
    points = np.load(pc_file)
    if points.dtype.names is not None:
        points = np.vstack([points[name] for name in ('x', 'y', 'z')]).T
    
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
    
    # Expand obstacles (safety margin)
    safety_voxels = 2
    occupancy_expanded = expand_obstacles_3d(occupancy, safety_voxels)
    
    # Specify a fixed Z thickness for each cuboid
    max_z_thickness = 20  # adjust as needed
    free_cuboids_blocks = region_growing_3d(occupancy_expanded, max_z_thickness=max_z_thickness)
    print("Number of free cuboids found:", len(free_cuboids_blocks))
    
    # Convert blocks to world coordinates and build connectivity.
    cuboids = build_cuboids_with_connectivity(free_cuboids_blocks, global_min, resolution, tol=0.0)
    
    # (Optional) Save connectivity graph.
    # plot_and_save_cuboid_graph(cuboids, out_file="my_cuboid_graph.png")
    
    # Randomly sample start and goal cuboids until a valid path is found.
    path_nodes = None
    metrics_list = []  # To store metrics from each attempt if desired.
    max_attempts = 10
    successful_metric = None
    for attempt in range(max_attempts):
        start_idx = random.choice(range(len(cuboids)))
        goal_idx = random.choice(range(len(cuboids)))
        while goal_idx == start_idx:
            goal_idx = random.choice(range(len(cuboids)))
        print(f"Attempt {attempt+1}: Trying start_idx = {start_idx}, goal_idx = {goal_idx}")
        t0 = time.time()
        candidate_path = astar_on_cuboids(cuboids, start_idx, goal_idx)
        t1 = time.time()
        search_time = t1 - t0
        if candidate_path is not None:
            path_nodes = candidate_path
            successful_metric = (attempt+1, start_idx, goal_idx, search_time)
            break
    if path_nodes is None:
        print("Failed to find a connected path after several attempts.")
        return
    print("Planned cuboid path (node indices):", path_nodes)
    
    # Convert cuboid path to list of 3D coordinates (cuboid centers)
    path_coords = []
    for idx in path_nodes:
        cub = cuboids[idx]
        center_pt = (cub['lower'] + cub['upper']) / 2.0
        path_coords.append(center_pt)
    
    # Compute path length (in world coordinates)
    path_length = 0.0
    for i in range(len(path_coords)-1):
        path_length += np.linalg.norm(np.array(path_coords[i+1]) - np.array(path_coords[i]))
    sol_nodes = len(path_nodes)
    
    # Save metrics (only for the successful attempt)
    if successful_metric is not None:
        metric_entry = (successful_metric[0], successful_metric[1], successful_metric[2],
                        successful_metric[3], path_length, sol_nodes)
        save_metrics([metric_entry], filename="metrics_a_star.txt")
    
    # Publish to RViz: publish all cuboids and the subset used for path planning.
    node = RVizPublisher(points, cuboids, path_coords, path_cuboid_indices=path_nodes, frame_id="map")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
