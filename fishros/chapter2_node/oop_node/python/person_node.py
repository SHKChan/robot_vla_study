import rclpy
from rclpy.node import Node


class PersonNode(Node):
    def __init__(self, name: str, age: str):
        super().__init__(self.__class__.__name__)
        self.get_logger().info(f'Hello from {self.get_name()}!')
        self.name = name
        self.age = age

    def eat(self, food: str):
        print(f'I am {self.name}, {self.age}, and I am eatting {food}.')


def main(args=None):
    rclpy.init(args=args)
    p1 = PersonNode('Bob', '22')
    p1.eat('hot dog')
    rclpy.spin(p1)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
