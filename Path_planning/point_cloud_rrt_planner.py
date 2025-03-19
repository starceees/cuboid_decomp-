#!/usr/bin/env python3

import numpy as np
import heapq
from tqdm import tqdm
import time
import scipy.ndimage as ndi
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
import tf2_ros
import tf2_geometry_msgs
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

#############################################
# 1. Obstacle Expansion (Safety Margin)
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
# 2. RRT-based 3D Path Planning (with Z-limit)
#############################################

def collision_free(occupancy, p1, p2, interp_resolution=0.5):
    """
    Checks if the straight-line path between grid points p1 and p2 is collision free.
    Interpolates along the line with steps of size 'interp_resolution'.
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    dist = np.linalg.norm(p2 - p1)
    n_steps = int(np.ceil(dist / interp_resolution)) + 1
    for t in np.linspace(0, 1, n_steps):
        pt = p1 + t*(p2 - p1)
        idx = tuple(np.round(pt).astype(int))
        # Check bounds
        if (idx[0] < 0 or idx[0] >= occupancy.shape[0] or
            idx[1] < 0 or idx[1] >= occupancy.shape[1] or
            idx[2] < 0 or idx[2] >= occupancy.shape[2]):
            return False
        if occupancy[idx] == 1:
            return False
    return True

def rrt_3d(occupancy, start, goal, max_iter=5000000, step_size=20, z_upper=None):
    """
    A simple RRT planner in grid space, restricting z to be <= z_upper.
    occupancy: 3D numpy array (1=obstacle, 0=free).
    start, goal: grid indices (tuple of ints).
    max_iter: maximum iterations.
    step_size: maximum extension (in grid cells) per iteration.
    z_upper: maximum z index to allow (if None, use full range).
    Returns a list of grid indices representing the path, or None if planning fails.
    """
    dims = occupancy.shape
    if z_upper is None or z_upper > dims[2]:
        z_upper = dims[2]
    print("running rrt planner: max_iter =", max_iter, 
          "step_size =", step_size, "z_upper =", z_upper)
    
    def sample_free():
        # Sample random free cell, but limit z <= z_upper-1.
        while True:
            ix = np.random.randint(0, dims[0])
            iy = np.random.randint(0, dims[1])
            iz = np.random.randint(0, z_upper)  # restricted z range
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
        # Extend from nearest toward rand_node by step_size.
        if d > step_size:
            new_node = np.round(np.array(nearest) + (vec / d) * step_size).astype(int)
        else:
            new_node = np.array(rand_node)
        
        # If new_node's z is beyond z_upper, skip it.
        if new_node[2] >= z_upper:
            continue
        
        new_node = tuple(new_node)
        if occupancy[new_node] == 1:
            continue
        if not collision_free(occupancy, nearest, new_node):
            continue
        tree[new_node] = nearest
        nodes.append(new_node)
        # If new_node is close enough to the goal, try connecting directly.
        if np.linalg.norm(np.array(new_node) - np.array(goal)) <= step_size:
            # Also clamp goal's z if needed.
            if goal[2] < z_upper and collision_free(occupancy, new_node, goal):
                tree[goal] = new_node
                path = []
                current = goal
                while current is not None:
                    path.append(current)
                    current = tree[current]
                return path[::-1]
    return None

#############################################
# 3. Grid-to-World Conversion
#############################################

def grid_to_world_3d(idx, global_min, resolution):
    """
    Converts grid indices (ix, iy, iz) to world coordinates (center of voxel).
    """
    return np.array([
        global_min[0] + (idx[0] + 0.5) * resolution,
        global_min[1] + (idx[1] + 0.5) * resolution,
        global_min[2] + (idx[2] + 0.5) * resolution
    ])

#############################################
# 4. ROS2 Path Visualization
#############################################

class PathVisualizer(Node):
    def __init__(self, points, path_world, global_min, resolution, start_idx, goal_idx):
        super().__init__('path_visualizer')
        
        # Create publishers
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.pointcloud_publisher = self.create_publisher(
            PointCloud2, 
            '/point_cloud', 
            qos
        )
        
        self.path_publisher = self.create_publisher(
            Path,
            '/planned_path',
            qos
        )
        
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            '/path_markers',
            qos
        )
        
        # Store data
        self.points = points
        self.path_world = path_world
        self.global_min = global_min
        self.resolution = resolution
        self.start_idx = start_idx
        self.goal_idx = goal_idx
        
        # Set up a timer for publishing
        self.timer = self.create_timer(1.0, self.publish_all)
        self.get_logger().info('Path Visualizer node initialized')
        
    def publish_all(self):
        """Publish point cloud, path, and markers at regular intervals"""
        now = self.get_clock().now().to_msg()
        frame_id = "map"
        
        # Publish point cloud
        self.publish_point_cloud(self.points, now, frame_id)
        
        # Publish path
        self.publish_path(self.path_world, now, frame_id)
        
        # Publish start and goal markers
        self.publish_markers(now, frame_id)
        
        self.get_logger().info('Published visualization data')
        
    def publish_point_cloud(self, points, stamp, frame_id):
        """Convert numpy points to ROS PointCloud2 message and publish"""
        # Create header
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        
        # Create PointCloud2 message
        pc_msg = pc2.create_cloud_xyz32(header, points)
        
        # Publish
        self.pointcloud_publisher.publish(pc_msg)
        
    def publish_path(self, path_world, stamp, frame_id):
        """Convert world points to ROS Path message and publish"""
        path_msg = Path()
        path_msg.header.stamp = stamp
        path_msg.header.frame_id = frame_id
        
        for point in path_world:
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = frame_id
            
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = float(point[2])
            
            # Set orientation to identity quaternion
            pose.pose.orientation.w = 1.0
            
            path_msg.poses.append(pose)
            
        self.path_publisher.publish(path_msg)
        
    def publish_markers(self, stamp, frame_id):
        """Publish markers for start and goal positions"""
        marker_array = MarkerArray()
        
        # Start marker (blue sphere)
        start_marker = Marker()
        start_marker.header.frame_id = frame_id
        start_marker.header.stamp = stamp
        start_marker.ns = "path_points"
        start_marker.id = 0
        start_marker.type = Marker.SPHERE
        start_marker.action = Marker.ADD
        
        # Set position
        start_world = grid_to_world_3d(self.start_idx, self.global_min, self.resolution)
        start_marker.pose.position.x = float(start_world[0])
        start_marker.pose.position.y = float(start_world[1])
        start_marker.pose.position.z = float(start_world[2])
        
        # Set orientation (identity quaternion)
        start_marker.pose.orientation.w = 1.0
        
        # Set scale
        start_marker.scale.x = 0.5
        start_marker.scale.y = 0.5
        start_marker.scale.z = 0.5
        
        # Set color (blue)
        start_marker.color.r = 0.0
        start_marker.color.g = 0.0
        start_marker.color.b = 1.0
        start_marker.color.a = 1.0
        
        # Set lifetime
        start_marker.lifetime.sec = 0
        
        # Goal marker (green sphere)
        goal_marker = Marker()
        goal_marker.header.frame_id = frame_id
        goal_marker.header.stamp = stamp
        goal_marker.ns = "path_points"
        goal_marker.id = 1
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        
        # Set position
        goal_world = grid_to_world_3d(self.goal_idx, self.global_min, self.resolution)
        goal_marker.pose.position.x = float(goal_world[0])
        goal_marker.pose.position.y = float(goal_world[1])
        goal_marker.pose.position.z = float(goal_world[2])
        
        # Set orientation (identity quaternion)
        goal_marker.pose.orientation.w = 1.0
        
        # Set scale
        goal_marker.scale.x = 0.5
        goal_marker.scale.y = 0.5
        goal_marker.scale.z = 0.5
        
        # Set color (green)
        goal_marker.color.r = 0.0
        goal_marker.color.g = 1.0
        goal_marker.color.b = 0.0
        goal_marker.color.a = 1.0
        
        # Set lifetime
        goal_marker.lifetime.sec = 0
        
        # Add markers to array
        marker_array.markers.append(start_marker)
        marker_array.markers.append(goal_marker)
        
        # Publish marker array
        self.marker_publisher.publish(marker_array)

#############################################
# 5. Main Routine
#############################################

def main(args=None):
    # Load the raw point cloud from file to build the occupancy grid.
    point_cloud_file = "/home/raghuram/ARPL/cuboid_decomp/cuboid_decomp-/pointcloud/pointcloud_gq/point_cloud_gq.npy"  # Update with your saved point cloud path
    try:
        points = np.load(point_cloud_file)
        print(f"Loaded point cloud from {point_cloud_file} with shape {points.shape}")
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        return
    
    # If the point cloud is structured (with field names), convert to simple XYZ array
    if points.dtype.names is not None:
        points = np.vstack([points[name] for name in ('x', 'y', 'z')]).T
    
    # Compute occupancy grid from the point cloud
    resolution = 0.2
    
    # Compute bounding box
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    print("Point Cloud Bounding Box:")
    print("  Min:", min_bound)
    print("  Max:", max_bound)
    
    extent = max_bound - min_bound
    nx = int(np.ceil(extent[0] / resolution))
    ny = int(np.ceil(extent[1] / resolution))
    nz = int(np.ceil(extent[2] / resolution))
    occupancy = np.zeros((nx, ny, nz), dtype=np.uint8)
    
    # Fill occupancy grid
    idxs = np.floor((points - min_bound) / resolution).astype(int)
    idxs = np.clip(idxs, 0, [nx-1, ny-1, nz-1])
    for ix, iy, iz in idxs:
        occupancy[ix, iy, iz] = 1
    global_min = min_bound
    print(f"Occupancy grid shape {occupancy.shape}. global_min: {global_min}")
    
    # Expand obstacles for planning
    safety_voxels = 2
    occupancy_safety = expand_obstacles_3d(occupancy, safety_voxels=safety_voxels)
    
    # Find free cells in the expanded grid to use as start and goal
    free_cells = np.argwhere(occupancy_safety == 0)
    if free_cells.size == 0:
        print("No free space found in the safety occupancy grid!")
        return
    
    np.random.shuffle(free_cells)
    start_idx = tuple(free_cells[0])
    goal_idx = tuple(free_cells[1])
    print("Random Grid start:", start_idx, "Random Grid goal:", goal_idx)
    
    # Restrict the z-range to 2/3 of the total height (in grid cells)
    z_upper = int(nz * (2.0/3.0))
    print(f"Restricting z range to [0, {z_upper}) out of {nz}.")
    
    # RRT planning on the safety occupancy grid with the z-limit
    start_time = time.time()
    path_indices = rrt_3d(occupancy_safety, start_idx, goal_idx,
                          max_iter=5000000, step_size=20, z_upper=z_upper)
    end_time = time.time()
    print("RRT search took {:.2f} seconds.".format(end_time - start_time))
    
    if path_indices is None:
        print("No path found between start and goal.")
        return
    
    print("Path found with", len(path_indices), "steps.")
    
    # Convert path grid indices to world coordinates
    path_world = [grid_to_world_3d(idx, global_min, resolution) for idx in path_indices]
    
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Create and run the path visualizer node
    visualizer = PathVisualizer(points, path_world, global_min, resolution, start_idx, goal_idx)
    
    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        visualizer.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
