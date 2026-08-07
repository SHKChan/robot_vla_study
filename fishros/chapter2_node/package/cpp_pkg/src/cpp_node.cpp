#include "rclcpp/rclcpp.hpp"

int main(int argc, char **argv)
{
    std::cout << "Args Count: " << argc << std::endl;
    std::cout << "First Arg: " << argv[0] << std::endl;

    rclcpp::init(argc, argv);
    std::shared_ptr<rclcpp::Node> node = std::make_shared<rclcpp::Node>("cpp_node");
    RCLCPP_INFO(node->get_logger(), "Hello from C++ Node!");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

// Check if hpp and libraries has linked
// ldd cpp_node