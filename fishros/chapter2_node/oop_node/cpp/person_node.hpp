#pragma once
#include "rclcpp/rclcpp.hpp"
#include <iostream>

using namespace rclcpp;

class PersonNode : public Node
{
private:
    static constexpr const char *NODE_NAME = "PersonNode";

public:
    std::string name;
    std::string age;

    PersonNode(const std::string &name, const std::string &age)
        : Node(NODE_NAME)
    {
        RCLCPP_INFO(get_logger(), "Hello from %s!", this->get_name());
        this->name = name;
        this->age = age;
    }

    void eat(const std::string food)
    {
        RCLCPP_INFO(get_logger(), "I am %s, %s, and %s developer.",
                    this->name.c_str(), this->age.c_str(), food.c_str());
    }
};