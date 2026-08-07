#include "./include/cpp-httplib/httplib.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <functional>
#include <random>
#include <cstdlib>
#include <vector>
#include <mutex>

std::mutex cout_mutex;

class Downloader
{
private:
    std::list<std::thread> workers;

public:
    Downloader() = default;

    ~Downloader()
    {
        joinAll();
    }

    void download(const std::string &host, const std::string &path,
                  std::function<void(const std::string &, const std::string &, const std::string &)> callback)
    {
        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "Thread ID: " << std::this_thread::get_id()
                      << ", start downloading of " << host << path << std::endl;
        }

        httplib::Client client(host);

        // thread_local std::mt19937 rng(std::random_device{}());
        // std::uniform_int_distribution<> dist(5, 10);
        // std::this_thread::sleep_for(std::chrono::seconds(dist(rng)));
        std::this_thread::sleep_for(std::chrono::seconds(std::rand() % 5 + 5));

        auto response = client.Get(path.c_str());
        if (response && response->status == 200)
        {
            callback(host, path, response->body);
        }
        else
        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "Thread ID: " << std::this_thread::get_id()
                      << ", failed to download " << host << path << std::endl;
        }
    }

    void startDownload(const std::string &host, const std::string &path,
                       std::function<void(const std::string &, const std::string &, const std::string &)> callback)
    {
        auto download_fun = std::bind(&Downloader::download, this, host, path, callback);
        std::thread t(download_fun);
        workers.push_back(std::move(t));
    }

    void joinAll()
    {
        for (auto &t : workers)
        {
            if (t.joinable())
            {
                t.join();
            }
        }
        workers.clear();
    }
};

int main(int argc, char *argv[])
{
    auto downloader = Downloader();

    auto callback = [](const std::string &host, const std::string &path, const std::string &result)
    {
        std::lock_guard<std::mutex> lock(cout_mutex); // Lock before printing
        std::cout << "Url: " << host << path
                  << " -> Word count: " << result.length()
                  << " -> Preview: " << result.substr(0, 20)
                  << "..." << std::endl;
    };

    downloader.startDownload("http://0.0.0.0:8000", "/novel1.txt", callback);
    downloader.startDownload("http://0.0.0.0:8000", "/novel2.txt", callback);
    downloader.startDownload("http://0.0.0.0:8000", "/novel3.txt", callback);

    downloader.joinAll();

    return 0;
}