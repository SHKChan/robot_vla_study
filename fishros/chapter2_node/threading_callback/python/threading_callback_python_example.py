import random
import threading
import requests
import time

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

# Callback function
def world_count(url, result):
    print(f"Url: {url}: Word count: {len(result)} -> Preview: {result[:20]}...")

def main():
    downloader = Downloader()
    downloader.start_download("http://0.0.0.0:8000/novel1.txt", world_count)
    downloader.start_download("http://0.0.0.0:8000/novel2.txt", world_count)
    downloader.start_download("http://0.0.0.0:8000/novel3.txt", world_count)

if __name__ == "__main__":
    main()


# Install python3-requests
# sudo apt update
# sudo apt install python3-requests

# Create txt files
# echo "Chapter 1: "Make America Great Again" (And Every Other Ally Miserable)" > novel1.txt
# echo "Chapter 2: Russia Is Our Friend (Because True Friends Share Classified Secrets)" > novel2.txt
# echo "Chapter 3: I Love China (Until the Tariffs Kick In)" > novel3.txt

# Serve files for download
# python3 -m http.server