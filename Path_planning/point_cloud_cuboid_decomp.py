#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import random
import scipy.ndimage as ndi
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2

#############################################
# 1. Obstacle Expansion (3D Safety Margin)
#############################################
def expand_obstacles_3d(occupancy, safety_voxels=2):
    """
    Expands obstacles in a 3D occupancy grid by 'safety_voxels' using morphological dilation.
    occupancy: 3D numpy array (1=obstacle, 0=free)
    Returns a new 3D array with obstacles expanded.
    """
    structure = ndi.generate_binary_structure(rank=3, connectivity=1)  # 6-connected 3D structure
    expanded = ndi.binary_dilation(occupancy.astype(bool),
                                   structure=structure,
                                   iterations=safety_voxels)
    return expanded.astype(np.uint8)

#############################################
# 2. 3D Uniform Free-Space Decomposition
#############################################
def uniform_free_decomposition_3d(occupancy, block_size, free_thresh=0.9):
    """
    Partitions the occupancy grid into fixed-size 3D blocks.
    - occupancy: 3D numpy array (1=obstacle, 0=free)
    - block_size: (Bx, By, Bz) in voxels
    - free_thresh: fraction of free voxels needed to mark block as free
    Returns a list of dicts, each describing a free cuboid:
      {
        'min_idx': np.array([ix, iy, iz]),
        'dimensions': np.array([Bx, By, Bz]) (or clipped if at boundary)
      }
    """
    nx, ny, nz = occupancy.shape
    Bx, By, Bz = block_size
    free_blocks = []
    
    for ix in range(0, nx, Bx):
        for iy in range(0, ny, By):
            for iz in range(0, nz, Bz):
                i_end = min(ix + Bx, nx)
                j_end = min(iy + By, ny)
                k_end = min(iz + Bz, nz)
                sub_block = occupancy[ix:i_end, iy:j_end, iz:k_end]
                
                total_voxels = sub_block.size
                free_count = np.count_nonzero(sub_block == 0)
                if total_voxels > 0:
                    free_ratio = free_count / total_voxels
                else:
                    free_ratio = 0.0
                
                if free_ratio >= free_thresh:
                    dims = np.array([i_end - ix, j_end - iy, k_end - iz])
                    free_blocks.append({
                        'min_idx': np.array([ix, iy, iz]),
                        'dimensions': dims
                    })
    return free_blocks

#############################################
# 3. Grid-to-World Conversion
#############################################
def block_to_world_cuboid(block, global_min, resolution):
    """
    Given a block dict with 'min_idx' and 'dimensions' in voxel units,
    return a dict describing the same region in world coordinates:
      {
        'lower': [x, y, z],
        'upper': [x, y, z],
        'dimensions_world': [dx, dy, dz]
      }
    """
    min_idx = block['min_idx']
    dims = block['dimensions']
    
    # Lower corner in world coordinates (voxel center of min_idx)
    lower = global_min + (min_idx * resolution)
    # Upper corner in world coordinates (voxel center of (min_idx + dims) ) minus a small epsilon
    # But often we just consider the block's full extent:
    upper = global_min + ((min_idx + dims) * resolution)
    
    return {
        'lower': lower,
        'upper': upper,
        'dimensions_world': upper - lower
    }

#############################################
# 4. ROS2 Publisher Node
#############################################
class RVizPublisher(Node):
    def __init__(self, points, cuboids, frame_id="map"):
        """
        :param points: Nx3 numpy array of the original point cloud
        :param cuboids: list of dicts with keys 'lower', 'upper', 'dimensions_world'
        :param frame_id: The ROS frame in which to visualize
        """
        super().__init__('rviz_publisher_3d_decomposition')
        
        # QoS for reliability
        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Publishers
        self.pc_pub = self.create_publisher(PointCloud2, '/point_cloud', qos)
        self.marker_pub = self.create_publisher(MarkerArray, '/free_cuboids', qos)
        
        # Data
        self.points = points
        self.cuboids = cuboids
        self.frame_id = frame_id
        
        # Timer to publish periodically
        self.timer = self.create_timer(1.0, self.publish_all)
        self.get_logger().info('3D Decomposition Publisher node initialized')
    
    def publish_all(self):
        now = self.get_clock().now().to_msg()
        self.publish_point_cloud(now)
        self.publish_cuboid_markers(now)
        self.get_logger().info('Published point cloud and free cuboid markers.')
    
    def publish_point_cloud(self, stamp):
        header = Header()
        header.stamp = stamp
        header.frame_id = self.frame_id
        
        # Convert Nx3 numpy array to list of [x,y,z]
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
            
            # Center of the cuboid
            center = (lower + upper) / 2.0
            
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = stamp
            marker.ns = "free_cuboids"
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Position = center
            marker.pose.position.x = float(center[0])
            marker.pose.position.y = float(center[1])
            marker.pose.position.z = float(center[2])
            marker.pose.orientation.w = 1.0
            
            # Scale = block dimensions
            marker.scale.x = float(dims[0])
            marker.scale.y = float(dims[1])
            marker.scale.z = float(dims[2])
            
            # Color
            marker.color.r = random.random()
            marker.color.g = random.random()
            marker.color.b = random.random()
            marker.color.a = 0.3
            
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)

#############################################
# 5. Main: Build 3D Occupancy, Decompose, Publish
#############################################
def main(args=None):
    rclpy.init(args=args)
    
    # Load your 3D point cloud
    pc_file = "/home/raghuram/ARPL/cuboid_decomp/cuboid_decomp-/pointcloud/pointcloud_gq/point_cloud_gq.npy"  # <-- Update path as needed
    points = np.load(pc_file)
    # If the .npy has named fields, convert to Nx3
    if points.dtype.names is not None:
        points = np.vstack([points[name] for name in ('x', 'y', 'z')]).T
    
    # Compute bounding box
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    print("Point Cloud Bounding Box:")
    print("  Min:", min_bound)
    print("  Max:", max_bound)
    
    # Build occupancy grid
    resolution = 0.2  # meters per voxel (adjust as needed)
    extent = max_bound - min_bound
    nx = int(np.ceil(extent[0] / resolution))
    ny = int(np.ceil(extent[1] / resolution))
    nz = int(np.ceil(extent[2] / resolution))
    occupancy = np.zeros((nx, ny, nz), dtype=np.uint8)
    
    # Fill occupancy: mark a voxel as occupied if any point falls in it
    idxs = np.floor((points - min_bound) / resolution).astype(int)
    idxs = np.clip(idxs, 0, [nx-1, ny-1, nz-1])
    for (ix, iy, iz) in idxs:
        occupancy[ix, iy, iz] = 1
    
    print(f"Occupancy grid shape: {occupancy.shape}. Occupied voxels: {occupancy.sum()}")
    
    # Expand obstacles in 3D
    safety_voxels = 2
    occupancy_expanded = expand_obstacles_3d(occupancy, safety_voxels)
    
    # 3D uniform decomposition
    # e.g. blocks of size 5x5x3 voxels, free if >= 90% free
    block_size = (5, 5, 3)
    free_thresh = 0.9
    free_blocks = uniform_free_decomposition_3d(occupancy_expanded, block_size, free_thresh)
    print("Number of free blocks found:", len(free_blocks))
    
    # Convert blocks to world cuboids
    cuboids = []
    for block in free_blocks:
        cub = block_to_world_cuboid(block, min_bound, resolution)
        cuboids.append(cub)
    
    # Create and spin the publisher node
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
