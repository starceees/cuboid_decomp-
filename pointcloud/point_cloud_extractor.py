#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2

import argparse

class PointCloudExtractor(Node):
    def __init__(self, topic_name=None, output_file="point_cloud.npy"):
        super().__init__('point_cloud_extractor')
        self.output_file = output_file

        # If user did not specify a topic, auto-discover one publishing PointCloud2
        if topic_name is None:
            topic_name = self.auto_discover_topic()
            if topic_name is None:
                self.get_logger().error("No PointCloud2 topic found. Exiting.")
                # Destroy this node so we can shut down
                self.destroy_node()
                rclpy.shutdown()
                return

        self.get_logger().info(f"Subscribing to topic: {topic_name}")
        self.subscription = self.create_subscription(
            PointCloud2,
            topic_name,
            self.listener_callback,
            10
        )

    def auto_discover_topic(self):
        """
        Look for any topic whose type is sensor_msgs/msg/PointCloud2 and return the first match.
        Return None if no PointCloud2 topics are found.
        """
        topics_and_types = self.get_topic_names_and_types()
        for (name, types) in topics_and_types:
            if 'sensor_msgs/msg/PointCloud2' in types:
                return name
        return None

    def listener_callback(self, msg):
        points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        
        if not points:
            self.get_logger().info("Received an empty point cloud.")
            return
        
        pc_array = np.array(points)
        self.get_logger().info(f"Received point cloud with shape: {pc_array.shape}")
        
        np.save(self.output_file, pc_array)
        self.get_logger().info(f"Point cloud saved to {self.output_file}")
        
        # Destroy the node and shut down the ROS context
        self.destroy_node()
        rclpy.shutdown()


def main(args=None):
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="PointCloudExtractor Node")
    parser.add_argument(
        "--topic",
        help="PointCloud2 topic to subscribe to. If omitted, the node will auto-discover one.",
        default=None
    )
    parser.add_argument(
        "--output",
        help="Output filename for the saved NumPy array.",
        default="point_cloud.npy"
    )

    # Extract known arguments for local usage; pass the rest to rclpy
    known_args, ros_unknown_args = parser.parse_known_args()

    rclpy.init(args=ros_unknown_args)
    extractor = PointCloudExtractor(
        topic_name=known_args.topic,
        output_file=known_args.output
    )
    rclpy.spin(extractor)


if __name__ == '__main__':
    main()
