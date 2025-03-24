#!/usr/bin/env python3
import numpy as np
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
import heapq

#############################################
# 1. Obstacle Expansion (Safety Margin)
#############################################
def expand_obstacles_3d(occupancy, safety_voxels=2):
    structure = ndi.generate_binary_structure(3, 1)  # 6-connected structure
    expanded = ndi.binary_dilation(occupancy.astype(bool), structure=structure, iterations=safety_voxels)
    return expanded.astype(np.uint8)

#############################################
# 2. Collision-Free Checker (for interpolation if needed)
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
# 4. A* Search in 3D (with adaptive z-range)
#############################################
def a_star_3d(occupancy, start, goal, z_upper=None):
    dims = occupancy.shape
    # If no z_upper is provided, use the full z-range.
    if z_upper is None:
        z_upper = dims[2]
    # Ensure that both start and goal are within the allowed z range.
    if start[2] >= z_upper or goal[2] >= z_upper:
        # Print a warning and adjust z_upper to cover both points.
        print("Adjusting z_upper from", z_upper, "to", max(start[2], goal[2]) + 1)
        z_upper = max(start[2], goal[2]) + 1
        if z_upper > dims[2]:
            z_upper = dims[2]
    
    # If start or goal is an obstacle, immediately fail.
    if occupancy[start] == 1 or occupancy[goal] == 1:
        return None, 0
    
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    expanded_nodes = 0

    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    # Use full 26-connected neighbors.
    neighbor_offsets = [(dx, dy, dz) for dx in [-1, 0, 1]
                                     for dy in [-1, 0, 1]
                                     for dz in [-1, 0, 1]
                                     if not (dx == 0 and dy == 0 and dz == 0)]
    
    while open_set:
        current_priority, current = heapq.heappop(open_set)
        expanded_nodes += 1
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, expanded_nodes
        for offset in neighbor_offsets:
            neighbor = (current[0] + offset[0],
                        current[1] + offset[1],
                        current[2] + offset[2])
            # Check grid bounds and enforce z-range.
            if (neighbor[0] < 0 or neighbor[0] >= dims[0] or 
                neighbor[1] < 0 or neighbor[1] >= dims[1] or 
                neighbor[2] < 0 or neighbor[2] >= z_upper):
                continue
            if occupancy[neighbor] == 1:
                continue
            tentative_g = g_score[current] + np.linalg.norm(np.array(offset))
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                priority = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (priority, neighbor))
                came_from[neighbor] = current
    return None, expanded_nodes

#############################################
# 5. ROS2 Visualizer Node: Path, Obstacles & Drone Animation
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
        
        self.path_world = path_world            # List of [x, y, z] points (overall path)
        self.obstacles_points = obstacles_points  # Obstacle point cloud (list of [x, y, z])
        self.drone_mesh_resource = drone_mesh_resource
        self.frame_id = frame_id
        
        # Drone animation state
        self.current_segment_index = 0
        self.alpha = 0.0
        self.alpha_increment = 0.02
        
        self.timer = self.create_timer(0.1, self.publish_all)
        self.get_logger().info('ROS2 Path Visualizer initialized')
    
    def publish_all(self):
        now = self.get_clock().now().to_msg()
        self.publish_path(now)
        self.publish_obstacles(now)
        self.publish_drone_marker(now)
    
    def publish_path(self, stamp):
        path_msg = Path()
        path_msg.header.stamp = stamp
        path_msg.header.frame_id = self.frame_id
        for pt in self.path_world:
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = self.frame_id
            pose.pose.position.x = float(pt[0])
            pose.pose.position.y = float(pt[1])
            pose.pose.position.z = float(pt[2])
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
# 6. Metrics Saving Function
#############################################
def save_metrics(metrics, filename="metrics.txt"):
    with open(filename, "w") as f:
        f.write("Segment\tStartGrid\tGoalGrid\tComputeTime(s)\tPathLength(m)\tExpandedNodes\tSolutionNodes\n")
        for m in metrics:
            seg, start_grid, goal_grid, comp_time, path_length, expanded_nodes, sol_nodes = m
            f.write(f"{seg}\t{start_grid}\t{goal_grid}\t{comp_time:.3f}\t{path_length:.3f}\t{expanded_nodes}\t{sol_nodes}\n")
    print("Metrics saved to", filename)

#############################################
# 7. Main Routine: Iterative A* Planning, Metrics & ROS2 Visualization
#############################################
def main(args=None):
    # Load point cloud to build occupancy grid.
    point_cloud_file = "/home/raghuram/ARPL/cuboid_decomp/cuboid_decomp-/pointcloud/pointcloud_gq/point_cloud_gq.npy"  # Update path as needed
    try:
        points = np.load(point_cloud_file)
        print(f"Loaded point cloud with shape {points.shape}")
    except Exception as e:
        print("Error loading point cloud:", e)
        return
    if points.dtype.names is not None:
        points = np.vstack([points[name] for name in ('x', 'y', 'z')]).T

    # Compute occupancy grid.
    resolution = 0.2
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    print("Point Cloud Bounding Box:\n  Min:", min_bound, "\n  Max:", max_bound)
    extent = max_bound - min_bound
    nx = int(np.ceil(extent[0] / resolution))
    ny = int(np.ceil(extent[1] / resolution))
    nz = int(np.ceil(extent[2] / resolution))
    occupancy = np.zeros((nx, ny, nz), dtype=np.uint8)
    idxs = np.floor((points - min_bound) / resolution).astype(int)
    idxs = np.clip(idxs, 0, [nx-1, ny-1, nz-1])
    for ix, iy, iz in idxs:
        occupancy[ix, iy, iz] = 1
    print(f"Occupancy grid shape: {occupancy.shape}")
    
    # Expand obstacles.
    safety_voxels = 2
    occupancy_safety = expand_obstacles_3d(occupancy, safety_voxels=safety_voxels)
    
    # Sample 6 free-space waypoints (first is global start, then 5 successive queries).
    free_cells = np.argwhere(occupancy_safety == 0)
    if free_cells.size == 0:
        print("No free space found!")
        return
    np.random.shuffle(free_cells)
    num_waypoints = 6
    waypoints = [tuple(cell) for cell in free_cells[:num_waypoints]]
    print("Sampled Waypoints (grid indices):")
    for wp in waypoints:
        print(wp)
    
    # Restrict z-range for planning.
    z_upper = int(nz * (2.0/3.0))
    print(f"Restricting z range to [0, {z_upper}) out of {nz}")
    
    overall_path = []     # Accumulate overall path (in grid indices)
    metrics_data = []     # Store metrics for each segment
    current_start = waypoints[0]
    
    # For each successive waypoint, plan a path using A*.
    for seg in range(1, len(waypoints)):
        current_goal = waypoints[seg]
        print(f"Planning segment {seg}: from {current_start} to {current_goal}")
        start_time = time.time()
        path_segment, expanded_nodes = a_star_3d(occupancy_safety, current_start, current_goal, z_upper=z_upper)
        compute_time = time.time() - start_time
        if path_segment is None:
            print(f"Segment {seg}: No path found from {current_start} to {current_goal}")
            current_start = current_goal
            continue
        path_world_segment = [grid_to_world_3d(pt, min_bound, resolution) for pt in path_segment]
        path_length = sum(np.linalg.norm(np.array(path_world_segment[i+1]) - np.array(path_world_segment[i]))
                          for i in range(len(path_world_segment)-1))
        sol_nodes = len(path_segment)
        metrics_data.append((seg, current_start, current_goal, compute_time, path_length, expanded_nodes, sol_nodes))
        print(f"Segment {seg}: time={compute_time:.3f}s, length={path_length:.3f}m, expanded_nodes={expanded_nodes}, solution_nodes={sol_nodes}")
        if overall_path:
            overall_path.extend(path_segment[1:])  # avoid duplicating connecting node
        else:
            overall_path.extend(path_segment)
        current_start = current_goal
    
    save_metrics(metrics_data, "metrics.txt")
    
    if not overall_path:
        print("No valid overall path was found. Exiting without visualization.")
        return
    
    overall_path_world = [grid_to_world_3d(pt, min_bound, resolution) for pt in overall_path]
    # Build an obstacle point cloud from all occupied voxels.
    occ_indices = np.argwhere(occupancy == 1)
    obstacles_points = [grid_to_world_3d(tuple(idx), min_bound, resolution) for idx in occ_indices]
    
    # Define drone mesh resource (update path as needed)
    drone_mesh_resource = "file:///home/raghuram/ARPL/cuboid_decomp/cuboid_decomp-/simulator/meshes/race2.stl"
    
    # Initialize ROS2 and launch the visualizer node.
    rclpy.init(args=args)
    visualizer = PathVisualizer(overall_path_world, obstacles_points, drone_mesh_resource, frame_id="map")
    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
