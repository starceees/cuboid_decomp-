#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2

class PointCloudExtractor(Node):
    def __init__(self):
        super().__init__('point_cloud_extractor')
        # Adjust the topic name if needed.
        self.subscription = self.create_subscription(
            PointCloud2,
            '/quadrotor/nvblox_node/static_occupancy',
            self.listener_callback,
            10)
    
    def listener_callback(self, msg):
        points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        
        if not points:
            self.get_logger().info("Received an empty point cloud.")
            return
        
        pc_array = np.array(points)
        self.get_logger().info(f"Received point cloud with shape: {pc_array.shape}")
        
        np.save('point_cloud_gq.npy', pc_array)
        self.get_logger().info("Point cloud saved to point_cloud.npy")
        
        # Destroy the node and shut down the ROS context
        self.destroy_node()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    extractor = PointCloudExtractor()
    rclpy.spin(extractor)
    extractor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
