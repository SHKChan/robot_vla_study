from queue import Queue
import random
import threading
import time

from requests import Response
import requests

import rclpy
from rclpy.node import Node, Publisher

from example_interfaces.msg import String

class Downloader:
    def __init__(self):
        pass

    def download(self, url, callback):
        print(f"Threading: {threading.get_ident()}, starting download of {url}")
        response = requests.get(url)
        response.encoding = "utf-8"
        # Sleep random time
        random_seconds = random.uniform(5, 10)
        time.sleep(random_seconds)

        if(200 == response.status_code):
            if(callback is not None):
                callback(url, response.text)
            return response
        else:
            # Print error reason
            print(f"Error downloading {url}: {response.reason}")
            return None

    def start_download(self, url, callback):
        # Create and start a thread
        thread = threading.Thread(target=self.download, args=(url, callback))
        thread.start()

class NovelPubNode(Node):
    downloader: Downloader
    _queues: Queue
    _publisher: Publisher
    _index: int

    def __init__(self) -> None:
        super().__init__(self.__class__.__name__)
        self.get_logger().info(f'Hello from {self.get_name()}!')

        self.downloader = Downloader()
        self._queues = Queue()
        self._index = 0

        self._publisher = self.create_publisher(String, 'Novel', 10)
        self.create_timer(5, self.timer_callback)

    def timer_callback(self) -> None:
        if self._queues.qsize() > 0:
            line = self._queues.get()
            msg: String = String()
            msg.data = f'Line {self._index} : "{line}"'
            self._publisher.publish(msg)
            self.get_logger().info(f'Published:\n Line {self._index} : "{msg.data}"')
            self._index += 1


    def start_download(self, url: str) -> None:
        response: Response = self.downloader.start_download(url, self.download_callback)

    def download_callback(self, url: str, text: str) -> None:
        if(text is not None):
            self.get_logger().info('Downloaded:\n "%s"' % text)
            for line in text.splitlines():
                self._queues.put(line)

def main():
    rclpy.init()
    node: NovelPubNode = NovelPubNode()
    node.start_download('http://0.0.0.0:8000/novel1.txt')
    node.start_download('http://0.0.0.0:8000/novel2.txt')
    node.start_download('http://0.0.0.0:8000/novel3.txt')
    rclpy.spin(node)
    rclpy.shutdown()