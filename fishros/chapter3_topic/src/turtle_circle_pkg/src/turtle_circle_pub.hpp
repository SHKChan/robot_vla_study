#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include <chrono>

using namespace std;
using namespace chrono_literals;

class TurtleCirclePubNode : public rclcpp::Node
{
private:
    static constexpr const char *NODE_NAME = "TurtleCirclePubNode";

    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;

    double radius_ = 0.0;

public:
    TurtleCirclePubNode(double radius = 1.0)
        : Node(NODE_NAME), radius_(radius)
    {
        RCLCPP_INFO(get_logger(), "Hello from %s!", this->get_name());

        this->publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("turtle1/cmd_vel", 10);
        this->timer_ = this->create_wall_timer(1000ms, bind(&TurtleCirclePubNode::timer_callback, this));
    }

    void timer_callback()
    {
        auto msg = geometry_msgs::msg::Twist();
        // radius = linear / angular
        msg.angular.z = 1.0;
        msg.linear.x = this->radius_ * msg.angular.z;
        this->publisher_->publish(msg);

        RCLCPP_INFO(get_logger(), "Publishing: radius: %f, linear: %f, angular: %f", this->radius_, msg.linear.x, msg.angular.z);
    }
};

// ros2 run turtlesim turtlesim_node

// ros2 topic list -t // List curent topic
// /parameter_events [rcl_interfaces/msg/ParameterEvent]
// /rosout [rcl_interfaces/msg/Log]
// /turtle1/cmd_vel [geometry_msgs/msg/Twist] // Command for turtle
// /turtle1/color_sensor [turtlesim/msg/Color]
// /turtle1/pose [turtlesim/msg/Pose] // Position of turtle