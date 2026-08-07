import rclpy
from chapter2_nodes.oop_node.python.person_node import PersonNode



class DeveloperNode(PersonNode):
    def __init__(self, name: str, age: str, stack: str):
        super().__init__(name, age)
        self.get_logger().info(f'Hello from {self.get_name()}!')
        self.stack = stack

    def introduce(self):
        print(f'I am {self.name}, {self.age}, and a {self.stack} developer.')


def main(args=None):
    rclpy.init(args=args)
    d1 = DeveloperNode('Felix', '33', 'CPP')
    d1.introduce()
    rclpy.spin(d1)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
