#!/usr/bin/env python3
import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
import random

#############################################
# 1. Obstacle Expansion (3D Safety Margin)
#############################################
def expand_obstacles_3d(occupancy, safety_voxels=2):
    """
    Expands obstacles in a 3D occupancy grid by 'safety_voxels' using morphological dilation.
    occupancy: 3D numpy array (1=obstacle, 0=free)
    Returns a new 3D array with obstacles expanded.
    """
    from scipy.ndimage import generate_binary_structure, binary_dilation
    structure = generate_binary_structure(rank=3, connectivity=1)  # 6-connected 3D structure
    expanded = binary_dilation(occupancy.astype(bool), structure=structure, iterations=safety_voxels)
    return expanded.astype(np.uint8)

#############################################
# 2. Build 3D Occupancy Grid from Point Cloud
#############################################
def build_occupancy_grid(points, global_min, resolution):
    """
    Given a Nx3 point cloud and a resolution (meters per voxel), build a 3D occupancy grid.
    Occupied voxels are marked as 1, free voxels as 0.
    global_min: minimum bound of the point cloud (3-vector)
    """
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
#############################################
def region_growing_3d(occupancy):
    """
    Greedily grows maximal free cuboids from unvisited free voxels.
    occupancy: 3D numpy array with 1=obstacle, 0=free.
    Returns a list of cuboids, each as a dictionary:
      {
         'min_idx': np.array([ix, iy, iz]),
         'dimensions': np.array([dx, dy, dz])
      }
    """
    nx, ny, nz = occupancy.shape
    visited = np.zeros_like(occupancy, dtype=bool)
    cuboids = []
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if occupancy[i, j, k] == 0 and not visited[i, j, k]:
                    # Initialize boundaries for region growing.
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
                        # Expand in +z
                        if k_max < nz - 1:
                            candidate = occupancy[i_min:i_max+1, j_min:j_max+1, k_max+1]
                            if np.all(candidate == 0):
                                k_max += 1
                                changed = True
                    # Mark all voxels in the region as visited.
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
    """
    Given a block (cuboid) with keys:
      'min_idx': [i, j, k]
      'dimensions': [dx, dy, dz]
    Convert to world coordinates:
      lower = global_min + min_idx * resolution
      upper = global_min + (min_idx + dimensions) * resolution
    Returns a dictionary with:
      'lower': np.array([x, y, z]),
      'upper': np.array([x, y, z]),
      'dimensions_world': upper - lower
    """
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
# 5. ROS2 Publisher Node for RViz2 Visualization
#############################################
class RVizPublisher(Node):
    def __init__(self, points, cuboids, frame_id="map"):
        """
        :param points: Nx3 numpy array (original point cloud)
        :param cuboids: list of dictionaries with keys 'lower', 'upper', 'dimensions_world'
        """
        super().__init__('rviz_region_growing_publisher')
        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.pc_pub = self.create_publisher(PointCloud2, '/point_cloud', qos)
        self.marker_pub = self.create_publisher(MarkerArray, '/free_cuboids', qos)
        self.timer = self.create_timer(1.0, self.publish_all)
        
        self.points = points
        self.cuboids = cuboids
        self.frame_id = frame_id
        
        self.get_logger().info("3D Region Growing Publisher initialized.")
    
    def publish_all(self):
        now = self.get_clock().now().to_msg()
        self.publish_point_cloud(now)
        self.publish_cuboid_markers(now)
        self.get_logger().info("Published point cloud and cuboid markers.")
    
    def publish_point_cloud(self, stamp):
        header = Header()
        header.stamp = stamp
        header.frame_id = self.frame_id
        pc_list = self.points.tolist()
        pc_msg = pc2.create_cloud_xyz32(header, pc_list)
        self.pc_pub.publish(pc_msg)
    
    def publish_cuboid_markers(self, stamp):
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
            marker.color.r = random.random()
            marker.color.g = random.random()
            marker.color.b = random.random()
            marker.color.a = 0.3
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)

#############################################
# 6. Main Routine
#############################################
def main(args=None):
    rclpy.init(args=args)
    
    # Load point cloud (Nx3 numpy array)
    # Update the path to your actual point cloud file.
    pc_file = "/home/raghuram/ARPL/cuboid_decomp/cuboid_decomp-/pointcloud/pointcloud_gq/point_cloud_gq.npy"
    points = np.load(pc_file)
    if points.dtype.names is not None:
        points = np.vstack([points[name] for name in ('x', 'y', 'z')]).T
    
    # Compute bounding box of the point cloud
    global_min = np.min(points, axis=0)
    global_max = np.max(points, axis=0)
    print("Point Cloud Bounding Box:")
    print("  Min:", global_min)
    print("  Max:", global_max)
    
    # Set voxel resolution (meters per voxel)
    resolution = 0.2
    
    # Build occupancy grid
    occupancy = build_occupancy_grid(points, global_min, resolution)
    print(f"Occupancy grid shape: {occupancy.shape}, Occupied voxels: {occupancy.sum()}")
    
    # Expand obstacles in 3D (for safety margin)
    safety_voxels = 2
    occupancy_expanded = expand_obstacles_3d(occupancy, safety_voxels)
    
    # Perform region growing on the 3D occupancy grid to obtain free cuboids.
    free_cuboids_blocks = region_growing_3d(occupancy_expanded)
    print("Number of free cuboids found (in grid indices):", len(free_cuboids_blocks))
    
    # Convert each free cuboid from grid indices to world coordinates.
    cuboids = []
    for block in free_cuboids_blocks:
        cub = block_to_world_cuboid(block, global_min, resolution)
        cuboids.append(cub)
    
    print("Publishing point cloud and free cuboids in RViz2...")
    
    # Create and spin the ROS2 publisher node.
    node = RVizPublisher(points, cuboids, frame_id="map")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
