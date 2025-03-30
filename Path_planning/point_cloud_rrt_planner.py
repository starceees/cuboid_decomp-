#!/usr/bin/env python3
import os
import csv
import numpy as np
import heapq
import time
import scipy.ndimage as ndi
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from visualization_msgs.msg import Marker

#############################################
# 1. Obstacle Expansion (Safety Margin)
#############################################
def expand_obstacles_3d(occupancy, safety_voxels=2):
    # Using a 6-connected structure for dilation
    structure = ndi.generate_binary_structure(3, 1)
    expanded = ndi.binary_dilation(occupancy.astype(bool), structure=structure, iterations=safety_voxels)
    return expanded.astype(np.uint8)

#############################################
# 2. Collision-Free Checker and RRT-based 3D Path Planning
#############################################
def collision_free(occupancy, p1, p2, interp_resolution=0.5):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    dist = np.linalg.norm(p2 - p1)
    n_steps = int(np.ceil(dist / interp_resolution)) + 1
    for t in np.linspace(0, 1, n_steps):
        pt = p1 + t * (p2 - p1)
        idx = tuple(np.round(pt).astype(int))
        if (idx[0] < 0 or idx[0] >= occupancy.shape[0] or
            idx[1] < 0 or idx[1] >= occupancy.shape[1] or
            idx[2] < 0 or idx[2] >= occupancy.shape[2]):
            return False
        if occupancy[idx] == 1:
            return False
    return True

def rrt_3d(occupancy, start, goal, max_iter=5000000, step_size=20, z_upper=None):
    dims = occupancy.shape
    if z_upper is None or z_upper > dims[2]:
        z_upper = dims[2]
    print(f"[INFO] Running RRT planner: max_iter={max_iter}, step_size={step_size}, z_upper={z_upper}")
    
    def sample_free():
        while True:
            ix = np.random.randint(0, dims[0])
            iy = np.random.randint(0, dims[1])
            iz = np.random.randint(0, z_upper)
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
        if d > step_size:
            new_node = np.round(np.array(nearest) + (vec / d) * step_size).astype(int)
        else:
            new_node = np.array(rand_node)
        
        if new_node[2] >= z_upper:
            continue
        
        new_node = tuple(new_node)
        if occupancy[new_node] == 1:
            continue
        if not collision_free(occupancy, nearest, new_node):
            continue
        tree[new_node] = nearest
        nodes.append(new_node)
        if np.linalg.norm(np.array(new_node) - np.array(goal)) <= step_size:
            if goal[2] < z_upper and collision_free(occupancy, new_node, goal):
                tree[goal] = new_node
                path = []
                current = goal
                while current is not None:
                    path.append(current)
                    current = tree[current]
                return path[::-1], len(nodes)
    return None, len(nodes)

#############################################
# 3. Grid-to-World Conversion
#############################################
def grid_to_world_3d(idx, global_min, resolution):
    return np.array([
        global_min[0] + (idx[0] + 0.5) * resolution,
        global_min[1] + (idx[1] + 0.5) * resolution,
        global_min[2] + (idx[2] + 0.5) * resolution
    ])

#############################################
# 4. Load Waypoints from TXT File
#############################################
def load_waypoints_txt(filename):
    """
    Expects a text file where:
      - The first line is the number of waypoints.
      - Each waypoint is represented on 7 separate lines:
          x, y, z, qx, qy, qz, qw.
    Only the (x, y, z) values are returned as numpy arrays.
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    count = int(lines[0].strip())
    waypoints = []
    idx = 1
    for _ in range(count):
        vals = [float(lines[idx + j].strip()) for j in range(7)]
        idx += 7
        waypoints.append(np.array(vals[:3]))
    return waypoints

#############################################
# 5. ROS2 Visualizer Node: Path, Obstacles & Drone Marker Animation
#############################################
class PathVisualizer(Node):
    def __init__(self, path_world, obstacles_points, drone_mesh_resource, frame_id="map"):
        super().__init__('path_visualizer')
        
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.path_pub = self.create_publisher(Path, '/planned_path', qos)
        self.pc_pub = self.create_publisher(PointCloud2, '/obstacle_cloud', qos)
        self.drone_pub = self.create_publisher(Marker, '/drone_mesh', qos)
        
        self.path_world = path_world        # List of [x,y,z] points (overall path)
        self.obstacles_points = obstacles_points  # Obstacle point cloud (list of [x,y,z])
        self.drone_mesh_resource = drone_mesh_resource
        self.frame_id = frame_id
        
        # Drone animation state
        self.current_segment_index = 0
        self.alpha = 0.0
        self.alpha_increment = 0.02
        
        self.timer = self.create_timer(0.1, self.publish_all)
        self.get_logger().info('[INFO] ROS2 Path Visualizer initialized')
    
    def publish_all(self):
        now = self.get_clock().now().to_msg()
        self.publish_path(now)
        self.publish_obstacles(now)
        self.publish_drone_marker(now)
        self.get_logger().info('[INFO] Published path, obstacles, and drone mesh')
    
    def publish_path(self, stamp):
        path_msg = Path()
        path_msg.header.stamp = stamp
        path_msg.header.frame_id = self.frame_id
        for point in self.path_world:
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = self.frame_id
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = float(point[2])
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)
    
    def publish_obstacles(self, stamp):
        header = Header()
        header.stamp = stamp
        header.frame_id = self.frame_id
        pc_msg = pc2.create_cloud_xyz32(header, self.obstacles_points)
        self.pc_pub.publish(pc_msg)
    
    def publish_drone_marker(self, stamp):
        if not self.path_world or len(self.path_world) < 2:
            return
        
        if self.current_segment_index >= len(self.path_world) - 1:
            self.current_segment_index = 0
        pt1 = np.array(self.path_world[self.current_segment_index])
        pt2 = np.array(self.path_world[self.current_segment_index + 1])
        pos = (1 - self.alpha) * pt1 + self.alpha * pt2
        self.alpha += self.alpha_increment
        if self.alpha >= 1.0:
            self.alpha = 0.0
            self.current_segment_index += 1
            if self.current_segment_index >= len(self.path_world) - 1:
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
# 6. Save Metrics Function (CSV Format)
#############################################
def save_metrics_csv(metrics, metrics_filepath):
    header = ["Segment", "StartGrid", "GoalGrid", "ComputeTime(s)", "PathLength(m)", "TreeSize", "SolutionNodes"]
    with open(metrics_filepath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for m in metrics:
            seg, start_grid, goal_grid, comp_time, path_length, tree_size, sol_nodes = m
            writer.writerow([seg, str(start_grid), str(goal_grid), f"{comp_time:.3f}", f"{path_length:.3f}", tree_size, sol_nodes])
    print(f"[INFO] Metrics saved to {metrics_filepath}")

#############################################
# 7. Main Routine: Iterative RRT Planning, Metrics & ROS2 Visualization
#############################################
def main(args=None):
    # CONFIGURATION parameters
    CONFIG = {
        "POINT_CLOUD_FILE": "/home/raghuram/ARPL/cuboid_decomp/cuboid_decomp-/pointcloud/pointcloud_gq/point_cloud_gq.npy",  # update as needed
        "RESOLUTION": 0.2,
        "SAFETY_VOXELS": 2,
        "MAX_ITER": 500,            # for each segment, reduce max iterations for demo purposes
        "STEP_SIZE": 20,
        "Z_RESTRICT": 2.0/3.0,      # restrict z-range to 2/3 of grid height
        "WAYPOINTS_FILE": "waypoints.txt",  # file containing waypoints (first line = count, then 7 lines per waypoint)
        "METRICS_DIR": "metrics_rrt",
        "METRICS_FILENAME": "metrics_rrt.csv",
        "DRONE_MESH_RESOURCE": "file:///home/raghuram/ARPL/cuboid_decomp/cuboid_decomp-/simulator/meshes/race2.stl"
    }
    
    # Load point cloud to build occupancy grid.
    point_cloud_file = CONFIG["POINT_CLOUD_FILE"]
    try:
        points = np.load(point_cloud_file)
        print(f"[INFO] Loaded point cloud with shape {points.shape}")
    except Exception as e:
        print(f"[ERROR] Error loading point cloud: {e}")
        return
    
    if points.dtype.names is not None:
        points = np.vstack([points[name] for name in ('x', 'y', 'z')]).T

    # Compute occupancy grid bounds and resolution.
    resolution = CONFIG["RESOLUTION"]
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    print(f"[INFO] Point Cloud Bounding Box:\n  Min: {min_bound}\n  Max: {max_bound}")
    
    extent = max_bound - min_bound
    nx = int(np.ceil(extent[0] / resolution))
    ny = int(np.ceil(extent[1] / resolution))
    nz = int(np.ceil(extent[2] / resolution))
    occupancy = np.zeros((nx, ny, nz), dtype=np.uint8)
    
    idxs = np.floor((points - min_bound) / resolution).astype(int)
    idxs = np.clip(idxs, 0, [nx-1, ny-1, nz-1])
    for ix, iy, iz in idxs:
        occupancy[ix, iy, iz] = 1
    print(f"[INFO] Occupancy grid shape: {occupancy.shape}")
    
    # Expand obstacles for safety.
    safety_voxels = CONFIG["SAFETY_VOXELS"]
    occupancy_safety = expand_obstacles_3d(occupancy, safety_voxels=safety_voxels)
    
    # Load waypoints from file and convert from world coordinates to grid indices.
    waypoints = []
    try:
        loaded_waypoints = load_waypoints_txt(CONFIG["WAYPOINTS_FILE"])
        print(f"[INFO] Loaded {len(loaded_waypoints)} waypoints from {CONFIG['WAYPOINTS_FILE']}")
        for wp in loaded_waypoints:
            grid_idx = np.floor((wp - min_bound) / resolution).astype(int)
            grid_idx = np.clip(grid_idx, [0,0,0], [nx-1, ny-1, nz-1])
            waypoints.append(tuple(grid_idx))
    except Exception as e:
        print(f"[ERROR] Error loading waypoints from {CONFIG['WAYPOINTS_FILE']}: {e}")
        # Fallback: sample 6 free-space waypoints randomly.
        free_cells = np.argwhere(occupancy_safety == 0)
        if free_cells.size == 0:
            print("[ERROR] No free space found!")
            return
        np.random.shuffle(free_cells)
        num_waypoints = 6
        waypoints = [tuple(cell) for cell in free_cells[:num_waypoints]]
        print("[INFO] Sampled waypoints from free space as fallback.")
    
    print("[INFO] Waypoints (grid indices):")
    for wp in waypoints:
        print(wp)
    
    # Restrict z-range for planning.
    z_upper = int(nz * CONFIG["Z_RESTRICT"])
    print(f"[INFO] Restricting z range to [0, {z_upper}) out of {nz}")
    
    overall_path = []  # Combined grid indices for overall path.
    metrics = []       # List to store metrics for each segment.
    current_start = waypoints[0]
    
    # Iteratively plan between consecutive waypoints.
    for seg in range(len(waypoints) - 1):
        current_goal = waypoints[seg + 1]
        print(f"[INFO] Planning segment {seg+1}: from {current_start} to {current_goal}")
        start_time = time.time()
        path_segment, tree_size = rrt_3d(occupancy_safety, current_start, current_goal,
                                         max_iter=CONFIG["MAX_ITER"],
                                         step_size=CONFIG["STEP_SIZE"],
                                         z_upper=z_upper)
        compute_time = time.time() - start_time
        
        if path_segment is None:
            print(f"[WARNING] Segment {seg+1}: No path found from {current_start} to {current_goal}")
            current_start = current_goal
            continue
        
        # Convert segment path (grid indices) to world coordinates.
        path_world_segment = [grid_to_world_3d(pt, min_bound, resolution) for pt in path_segment]
        # Compute segment length in world space.
        path_length = sum(np.linalg.norm(np.array(path_world_segment[i+1]) - np.array(path_world_segment[i]))
                          for i in range(len(path_world_segment)-1))
        sol_nodes = len(path_segment)
        metrics.append((seg+1, current_start, current_goal, compute_time, path_length, tree_size, sol_nodes))
        print(f"[INFO] Segment {seg+1}: time={compute_time:.3f}s, length={path_length:.3f}m, tree_size={tree_size}, solution_nodes={sol_nodes}")
        
        # Append segment to overall path (avoid duplicating connecting waypoint).
        if overall_path:
            overall_path.extend(path_segment[1:])
        else:
            overall_path.extend(path_segment)
        current_start = current_goal
    
    # Create metrics directory if it doesn't exist and define metrics file path.
    metrics_dir = CONFIG["METRICS_DIR"]
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_filepath = os.path.join(metrics_dir, CONFIG["METRICS_FILENAME"])
    save_metrics_csv(metrics, metrics_filepath)
    
    if not overall_path:
        print("[ERROR] No valid overall path was found. Exiting without visualization.")
        return
    
    # Convert overall path to world coordinates.
    overall_path_world = [grid_to_world_3d(pt, min_bound, resolution) for pt in overall_path]
    
    # Create obstacle point cloud for visualization.
    occ_indices = np.argwhere(occupancy == 1)
    obstacles_points = [grid_to_world_3d(tuple(idx), min_bound, resolution) for idx in occ_indices]
    
    # Initialize ROS2 and start the visualizer node.
    rclpy.init(args=args)
    visualizer = PathVisualizer(overall_path_world, obstacles_points, CONFIG["DRONE_MESH_RESOURCE"], frame_id="map")
    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
