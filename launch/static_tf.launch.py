from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # This Node publishes a static transform with zero translation and rotation
    # between 'world' (parent) and 'odom' (child).
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_world_to_odom',
        arguments=[
            '0',    # x
            '0',    # y
            '0',    # z
            '0',    # roll
            '0',    # pitch
            '0',    # yaw
            'world',  # parent frame
            'odom'    # child frame
        ]
    )

    return LaunchDescription([static_tf_node])

