#include "rclcpp/rclcpp.hpp"
#include "developer_node.hpp"
#include "person_node.hpp"

using namespace rclcpp;

int main(int argc, char **argv)
{
    init(argc, argv);

    auto perNode = std::make_shared<PersonNode>("Bob", "22");
    auto devNode = std::make_shared<DeveloperNode>("Felix", "33", "CPP");

    perNode->eat("hot dog");
    devNode->introduce();

    // use MultiThreadedExecutor to spin both nodes simultaneously
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(perNode);
    executor.add_node(devNode);
    executor.spin(); // blocks here, both nodes running

    shutdown();
    return 0;
}