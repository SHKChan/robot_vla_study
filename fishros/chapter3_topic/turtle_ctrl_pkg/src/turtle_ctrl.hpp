#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "turtlesim/msg/pose.hpp"
#include <chrono>

using namespace std;
using namespace chrono_literals;

class TurtleCtrlNode : public rclcpp::Node
{
private:
    static constexpr const char *NODE_NAME = "TurtleCtrlNode";
    static constexpr const double MAX_LINEAR_VEL = 3.0;

    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_;
    rclcpp::Subscription<turtlesim::msg::Pose>::SharedPtr subscriber_;

    bool target_set_ = false;
    turtlesim::msg::Pose target_;
    double kp_ = 1.5, ki_ = 0.0, kd_ = 0.0; // tune these — start with ki_=kd_=0
    double integral_distance_ = 0.0, integral_angle_ = 0.0;
    double pre_error_distance_ = 0.0, pre_error_angle_ = 0.0;

public:
    TurtleCtrlNode(double radius = 1.0)
        : Node(NODE_NAME)
    {
        RCLCPP_INFO(get_logger(), "Hello from %s!", this->get_name());

        this->publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("turtle1/cmd_vel", 10);
        this->subscriber_ = this->create_subscription<turtlesim::msg::Pose>("turtle1/pose", 10, bind(&TurtleCtrlNode::pose_callback, this, placeholders::_1));
    }

    void pose_callback(const turtlesim::msg::Pose::SharedPtr pose)
    {
        if (!this->target_set_)
            return;

        // Current robot pose from /turtle1/pose
        RCLCPP_INFO(get_logger(), "Current position: %f, %f", pose->x, pose->y);

        // Position error in world frame (target - current)
        double error_x = this->target_.x - pose->x;
        double error_y = this->target_.y - pose->y;

        // --- Convert world-frame x/y error into robot-actionable error ---
        // distance_error: straight-line distance to target -> drives linear velocity
        double distance_error = std::sqrt(error_x * error_x + error_y * error_y);
        // angle_error: heading difference between "direction to target" and current heading -> drives angular velocity
        double angle_error = std::atan2(error_y, error_x) - pose->theta;
        // Normalize angle_error to [-pi, pi] so the turtle always turns the short way
        // (without this, e.g. 350 deg error would spin almost a full circle instead of turning -10 deg)
        while (angle_error > M_PI)
            angle_error -= 2 * M_PI;
        while (angle_error < -M_PI)
            angle_error += 2 * M_PI;

        RCLCPP_INFO(get_logger(), "distance_err: %f, angle_err: %f", distance_error, angle_error);

        // --- Integral term: accumulate error over time to eliminate steady-state offset ---
        // Caution: no anti-windup here — if the turtle is blocked/stalled, these grow unbounded
        // and cause overshoot once it's freed. Fine for this learning exercise, but flag for later.
        this->integral_distance_ += distance_error;
        this->integral_angle_ += angle_error;

        // --- Derivative term: rate of change of error, used to dampen oscillation ---
        double derivative_distance = distance_error - this->pre_error_distance_;
        double derivative_angle = angle_error - this->pre_error_angle_;

        // --- PID output: P (current error) + I (accumulated error) + D (error trend) ---
        double linear_vel = kp_ * distance_error + ki_ * integral_distance_ + kd_ * derivative_distance;
        double angular_vel = kp_ * angle_error + ki_ * integral_angle_ + kd_ * derivative_angle;

        // Save current error as "previous" for next callback's derivative calc
        this->pre_error_distance_ = distance_error;
        this->pre_error_angle_ = angle_error;

        // Snap to zero once within tolerance — prevents jitter/creep from tiny residual PID output near the goal
        if (distance_error < 0.05)
        {
            linear_vel = 0.0;
            angular_vel = 0.0;

            this->integral_distance_ = 0.0;
            this->integral_angle_ = 0.0;
            this->pre_error_distance_ = 0.0;
            this->pre_error_angle_ = 0.0;

            this->target_set_ = false;
        }

        auto msg = geometry_msgs::msg::Twist();
        msg.linear.x = linear_vel;
        msg.angular.z = angular_vel;
        this->publisher_->publish(msg);
    }

    void go_to(double x, double y, double theta)
    {
        this->target_.x = x;
        this->target_.y = y;
        this->target_.theta = theta;
        this->target_set_ = true;
    }
};

// ros2 run turtlesim turtlesim_node

// ros2 topic list -t // List curent topic
// /parameter_events [rcl_interfaces/msg/ParameterEvent]
// /rosout [rcl_interfaces/msg/Log]
// /turtle1/cmd_vel [geometry_msgs/msg/Twist] // Command for turtle
// /turtle1/color_sensor [turtlesim/msg/Color]
// /turtle1/pose [turtlesim/msg/Pose] // Position of turtle