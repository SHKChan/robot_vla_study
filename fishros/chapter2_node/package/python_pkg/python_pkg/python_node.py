import rclpy
from rclpy.node import Node


def main(args=None):
    rclpy.init(args=args)
    node = Node('python_node')
    node.get_logger().info('Hello from Python Node!')
    rclpy.spin(node)
    rclpy.shutdown()

# Place next to __init__.py
# colcon build
# source install/setup.bash
