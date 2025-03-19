#!/usr/bin/env python3

import numpy as np
import heapq
import time
import scipy.ndimage as ndi
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from builtin_interfaces.msg import Time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

#############################################
# 1. Obstacle Expansion (Safety Margin)
#############################################

def expand_obstacles_3d(occupancy, safety_voxels=2):
    structure = ndi.generate_binary_structure(3, 1)  # 6-connected structure
    expanded = ndi.binary_dilation(occupancy.astype(bool), structure=structure, iterations=safety_voxels)
    return expanded.astype(np.uint8)

#############################################
# 2. RRT-based 3D Path Planning (with Z-limit)
#############################################

def collision_free(occupancy, p1, p2, interp_resolution=0.5):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    dist = np.linalg.norm(p2 - p1)
    n_steps = int(np.ceil(dist / interp_resolution)) + 1
    for t in np.linspace(0, 1, n_steps):
        pt = p1 + t*(p2 - p1)
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
    print("running rrt planner: max_iter =", max_iter, 
          "step_size =", step_size, "z_upper =", z_upper)
    
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
                return path[::-1]
    return None

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
# 4. ROS2 Path Visualization Node for RViz2
#############################################

class PathVisualizer(Node):
    def __init__(self, path_world, obstacles_points, frame_id="map"):
        super().__init__('path_visualizer')
        
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.path_pub = self.create_publisher(Path, '/planned_path', qos)
        self.pc_pub = self.create_publisher(PointCloud2, '/obstacle_cloud', qos)
        
        self.path_world = path_world
        self.obstacles_points = obstacles_points
        self.frame_id = frame_id
        
        self.timer = self.create_timer(1.0, self.publish_all)
        self.get_logger().info('ROS2 Path Visualizer initialized')
    
    def publish_all(self):
        now = self.get_clock().now().to_msg()
        self.publish_path(now)
        self.publish_obstacles(now)
        self.get_logger().info('Published path and obstacle cloud')
    
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

#############################################
# 5. Main Routine: Compute and Publish for RViz2
#############################################

def main(args=None):
    # Load point cloud from file to build occupancy grid.
    point_cloud_file = "/home/raghuram/ARPL/cuboid_decomp/cuboid_decomp-/pointcloud/pointcloud_gq/point_cloud_gq.npy"  # Update with your point cloud path
    try:
        points = np.load(point_cloud_file)
        print(f"Loaded point cloud with shape {points.shape}")
    except Exception as e:
        print("Error loading point cloud:", e)
        return
    
    if points.dtype.names is not None:
        points = np.vstack([points[name] for name in ('x', 'y', 'z')]).T
    
    # Compute occupancy grid
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
    
    global_min = min_bound
    print(f"Occupancy grid shape: {occupancy.shape}")
    
    # Expand obstacles for safety.
    safety_voxels = 2
    occupancy_safety = expand_obstacles_3d(occupancy, safety_voxels=safety_voxels)
    
    # Select random free cells as start and goal.
    free_cells = np.argwhere(occupancy_safety == 0)
    if free_cells.size == 0:
        print("No free space found!")
        return
    np.random.shuffle(free_cells)
    start_idx = tuple(free_cells[0])
    goal_idx = tuple(free_cells[1])
    print("Start:", start_idx, "Goal:", goal_idx)
    
    # Restrict z-range for planning.
    z_upper = int(nz * (2.0/3.0))
    print(f"Restricting z range to [0, {z_upper}) out of {nz}")
    
    # Run RRT planning.
    start_time = time.time()
    path_indices = rrt_3d(occupancy_safety, start_idx, goal_idx, max_iter=5000000, step_size=20, z_upper=z_upper)
    end_time = time.time()
    print("RRT search took {:.2f} seconds".format(end_time - start_time))
    
    if path_indices is None:
        print("No path found!")
        return
    print("Path found with", len(path_indices), "steps")
    
    # Convert grid indices to world coordinates.
    path_world = [grid_to_world_3d(idx, global_min, resolution) for idx in path_indices]
    
    # Create an obstacle point cloud for visualization (from original occupancy grid).
    occ_indices = np.argwhere(occupancy == 1)
    occ_points = np.array([grid_to_world_3d(tuple(idx), global_min, resolution) for idx in occ_indices])
    
    # Initialize ROS2 and start the visualizer node.
    rclpy.init(args=args)
    visualizer = PathVisualizer(path_world, occ_points)
    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
