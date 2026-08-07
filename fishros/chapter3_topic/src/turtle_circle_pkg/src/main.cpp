#include "rclcpp/rclcpp.hpp"
#include "turtle_circle_pub.hpp"

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    std::shared_ptr<TurtleCirclePubNode> node = std::make_shared<TurtleCirclePubNode>(3.0);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}