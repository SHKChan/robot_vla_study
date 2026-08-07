#pragma once
#include <iostream>
#include "person_node.hpp"

using namespace rclcpp;

class DeveloperNode : public PersonNode
{
private:
    static constexpr const char *NODE_NAME = "DeveloperNode";

public:
    std::string role;

    DeveloperNode(const std::string &name, const std::string &age, const std::string &role)
        : PersonNode(name, age)
    {
        this->role = role;
    }

    void introduce()
    {
        RCLCPP_INFO(get_logger(), "I am %s, %s, eating %s.",
                    this->name.c_str(), this->age.c_str(), this->role.c_str());
    }
};