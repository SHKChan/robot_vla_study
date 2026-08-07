cmake_minimum_required(VERSION 3.8)
project(oop_node_cpp)

find_package(rclcpp REQUIRED)

add_executable(main_node main.cpp)
target_include_directories(main_node PUBLIC
    ${rclcpp_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(main_node ${rclcpp_LIBRARIES})