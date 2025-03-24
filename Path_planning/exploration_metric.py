#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
import sensor_msgs_py.point_cloud2 as pc2
import random, os, time
from tqdm import tqdm

#############################################
# Utility: Convert grid index to world coordinate (center of voxel)
#############################################
def grid_to_world(idx, global_min, resolution):
    return global_min + (np.array(idx) + 0.5) * resolution

#############################################
# Saving Metrics to File
#############################################
def save_metrics(metrics, filename="metrics.txt"):
    with open(filename, "w") as f:
        f.write("Segment\tStartGrid\tGoalGrid\tComputeTime(s)\tPathLength(m)\tTreeSize\tSolutionNodes\tStartPoint\tGoalPoint\n")
        for seg, s, g, comp_time, path_length, tree_size, sol_nodes, sp, gp in metrics:
            sp_str = f"({sp[0]:.2f}, {sp[1]:.2f}, {sp[2]:.2f})"
            gp_str = f"({gp[0]:.2f}, {gp[1]:.2f}, {gp[2]:.2f})"
            f.write(f"{seg}\t{s}\t{g}\t{comp_time:.3f}\t{path_length:.3f}\t{tree_size}\t{sol_nodes}\t{sp_str}\t{gp_str}\n")
    print("Metrics saved to", filename)

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
    nz_dim = max(1, int(np.ceil(extent[2] / resolution)))  # Ensure at least one voxel in z
    occupancy = np.zeros((nx_dim, ny_dim, nz_dim), dtype=np.uint8)
    
    idxs = np.floor((points - global_min) / resolution).astype(int)
    idxs = np.clip(idxs, 0, [nx_dim-1, ny_dim-1, nz_dim-1])
    for (ix, iy, iz) in idxs:
        occupancy[ix, iy, iz] = 1
    return occupancy

#############################################
# 3. Collision-Free Checker and RRT Planner in Voxel Space
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

def rrt_3d(occupancy, start, goal, max_iter=100000, step_size=10, z_upper=None):
    dims = occupancy.shape
    if z_upper is None or z_upper > dims[2]:
        z_upper = dims[2]
    tree = {start: None}
    nodes = [start]
    for i in tqdm(range(max_iter), desc="RRT iterations"):
        # Sample a random free voxel
        ix = np.random.randint(0, dims[0])
        iy = np.random.randint(0, dims[1])
        iz = np.random.randint(0, z_upper)
        if occupancy[ix, iy, iz] != 0:
            continue
        rand_node = (ix, iy, iz)
        # Find nearest node in tree
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
# 4. ROS2 Publisher Node for Visualization & Drone Motion
#############################################
class RVizPublisher(Node):
    def __init__(self, points, path_coords, drone_mesh_resource, frame_id="map"):
        super().__init__('rviz_rrt_planner')
        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.pc_pub = self.create_publisher(PointCloud2, '/point_cloud', qos)
        self.line_path_pub = self.create_publisher(Marker, '/line_path', qos)
        self.path_pub = self.create_publisher(Path, '/planned_path', qos)
        self.drone_pub = self.create_publisher(Marker, '/drone_mesh', qos)
        
        self.points = points
        self.path_coords = path_coords  # Overall path in world coordinates
        self.frame_id = frame_id
        
        self.drone_mesh_resource = drone_mesh_resource
        self.current_segment_index = 0
        self.alpha = 0.0
        self.alpha_increment = 0.02
        
        self.timer = self.create_timer(0.1, self.publish_all)
        self.get_logger().info("RRT Planner Node initialized.")
    
    def publish_all(self):
        now = self.get_clock().now().to_msg()
        self.publish_point_cloud(now)
        self.publish_line_path(now)
        self.publish_path(now)
        self.publish_drone_marker(now)
        self.get_logger().info("Published point cloud, path, and drone mesh.")
    
    def publish_point_cloud(self, stamp):
        header = Header()
        header.stamp = stamp
        header.frame_id = self.frame_id
        pc_list = self.points.tolist()
        pc_msg = pc2.create_cloud_xyz32(header, pc_list)
        self.pc_pub.publish(pc_msg)
    
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
        if not self.path_coords:
            # No path available; nothing to publish.
            return
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
# 5. Main Routine: RRT Planning in Voxel Space with 5 Successive Waypoints & Metrics
#############################################
def main(args=None):
    rclpy.init(args=args)
    
    # Load point cloud (Nx3 numpy array)
    pc_file = "/home/raghuram/ARPL/cuboid_decomp/cuboid_decomp-/simulator/occupancy_point_cloud.npy"  # Update as needed
    points = np.load(pc_file)
    # If loaded as structured array, convert to Nx3
    if points.dtype.names is not None:
        points = np.vstack([points[name] for name in ('x','y','z')]).T

    # Print bounding box in world coordinates
    global_min = np.min(points, axis=0)
    global_max = np.max(points, axis=0)
    print("Point Cloud Bounding Box:")
    print("  Min:", global_min)
    print("  Max:", global_max)
    
    # Build occupancy grid and expand obstacles to add safety margin
    resolution = 0.2
    occupancy = build_occupancy_grid(points, global_min, resolution)
    print(f"Occupancy grid shape: {occupancy.shape}, Occupied voxels: {occupancy.sum()}")
    safety_voxels = 2
    occupancy_expanded = expand_obstacles_3d(occupancy, safety_voxels)
    
    # --- Sample 5 Unique Random Free-Space Waypoints from Occupancy Grid ---
    free_indices = np.argwhere(occupancy == 0)
    free_indices = np.unique(free_indices, axis=0)
    if free_indices.shape[0] < 5:
        raise ValueError("Not enough free cells available!")
    
    # Sample indices directly from free_indices array
    sample_indices = random.sample(range(len(free_indices)), 5)
    waypoint_grid_indices = [tuple(free_indices[i]) for i in sample_indices]
    waypoint_points = [grid_to_world(idx, global_min, resolution) for idx in waypoint_grid_indices]
    
    print("Random free-space waypoint coordinates (world):")
    for pt in waypoint_points:
        print(pt)
    
    # --- Plan successive subpaths between the 5 waypoints using RRT in voxel space ---
    num_segments = len(waypoint_grid_indices) - 1
    overall_path = []  # Combined grid path (list of grid indices)
    metrics = []       # (Segment, StartGrid, GoalGrid, ComputeTime, PathLength, TreeSize, SolutionNodes, StartPoint, GoalPoint)
    current_start = waypoint_grid_indices[0]
    for seg in range(num_segments):
        current_goal = waypoint_grid_indices[seg+1]
        # Check for degenerate case
        if current_start == current_goal:
            print(f"Segment {seg+1}: Start and goal are identical; skipping planning.")
            continue
        start_time = time.time()
        path_segment, tree_size = rrt_3d(occupancy_expanded, current_start, current_goal, max_iter=100000, step_size=10)
        compute_time = time.time() - start_time
        if path_segment is None:
            print(f"Segment {seg+1}: No path found from {current_start} to {current_goal}")
            continue
        # Convert grid path to world coordinates to compute path length
        path_world = [grid_to_world(pt, global_min, resolution) for pt in path_segment]
        path_length = sum(np.linalg.norm(np.array(path_world[i+1]) - np.array(path_world[i]))
                          for i in range(len(path_world)-1))
        sol_nodes = len(path_segment)
        metrics.append((seg+1, current_start, current_goal, compute_time, path_length, tree_size, sol_nodes,
                        grid_to_world(current_start, global_min, resolution),
                        grid_to_world(current_goal, global_min, resolution)))
        overall_path.extend(path_segment if seg == 0 else path_segment[1:])
        current_start = current_goal
        print(f"Segment {seg+1}: from {current_start} to {current_goal}, time: {compute_time:.3f}s, length: {path_length:.3f}m, TreeSize: {tree_size}, SolutionNodes: {sol_nodes}")
    
    save_metrics(metrics, "metrics.txt")
    
    # If no overall path was found, exit before creating the publisher node.
    if not overall_path:
        print("No valid overall path was found. Exiting without visualization.")
        rclpy.shutdown()
        return

    # Build overall path in world coordinates using the single conversion function
    overall_path_coords = [grid_to_world(pt, global_min, resolution) for pt in overall_path]
    
    # Define drone mesh resource (using an STL file)
    drone_mesh_resource = "file:///home/raghuram/ARPL/cuboid_decomp/cuboid_decomp-/simulator/meshes/race2.stl"
    
    # Create ROS2 publisher node for visualization using the overall path
    publisher_node = RVizPublisher(points, overall_path_coords, drone_mesh_resource, frame_id="map")
    publisher_node.current_segment_index = 0
    publisher_node.alpha = 0.0
    publisher_node.alpha_increment = 0.02
    
    try:
        rclpy.spin(publisher_node)
    except KeyboardInterrupt:
        pass
    finally:
        publisher_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
