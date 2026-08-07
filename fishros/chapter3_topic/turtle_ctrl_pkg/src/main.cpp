#include "rclcpp/rclcpp.hpp"
#include "turtle_ctrl.hpp"

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    std::shared_ptr<TurtleCtrlNode> node = std::make_shared<TurtleCtrlNode>();
    node->go_to(2.0, 2.0, 0.0);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}