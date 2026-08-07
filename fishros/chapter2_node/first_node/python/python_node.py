import rclpy
from rclpy.node import Node


def main(args=None):
    rclpy.init(args=args)
    node = Node('python_node')
    node.get_logger().info('Hello from Python Node!')
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

# Format for console output
# export RCUTILS_CONSOLE_OUTPUT_FORMAT="[{function_name}:{line_number}]:{message}"
# export RCUTILS_CONSOLE_OUTPUT_FORMAT="[{severity}] [{time}] [{name}] [{function_name}:{line_number}]: {message}"

# python3 python_node.py