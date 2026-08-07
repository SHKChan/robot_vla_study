import threading
import time

import rclpy
from rclpy.node import Node, Subscription

from queue import Queue
from example_interfaces.msg import String

from espeakng import Speaker

class NovelSubNode(Node):
    _subscriber:Subscription
    _queues: Queue
    _speak_thread: threading.Thread

    def __init__(self) -> None:
        super().__init__(self.__class__.__name__)
        self.get_logger().info(f'Hello from {self.get_name()}!')

        self._queues = Queue()
        self._subscriber = self.create_subscription(String, 'Novel', self.novel_callback, 10)

    def novel_callback(self, msg: String) -> None:
        self._queues.put(msg.data)

    def speak(self) -> None:
        speaker: Speaker = Speaker('en')

        while rclpy.ok():
            if self._queues.qsize() > 0:
                text: str = self._queues.get()
                self.get_logger().info(f'Speaking:\n {self._queues.get()}')
                speaker.say(text)
                speaker.wait() # wait for speech to finish
            else:
                # Put the thread to sleep for 1 second
                time.sleep(1)

    def start_speaking(self) -> None:
        self._speak_thread = threading.Thread(target=self.speak)
        self._speak_thread.start()

def main():
    rclpy.init()
    node: NovelSubNode = NovelSubNode()
    node.start_speaking()
    rclpy.spin(node)
    rclpy.shutdown()