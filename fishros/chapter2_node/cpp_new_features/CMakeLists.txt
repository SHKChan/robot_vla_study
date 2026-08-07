cmake_minimum_required(VERSION 3.8)
project(cpp_new_features)

# auto
add_executable(auto_example auto_example.cpp)
# target_include_directories(person_node PUBLIC ${rclcpp_INCLUDE_DIRS})
# target_link_libraries(person_node ${rclcpp_LIBRARIES})

# share pointer
add_executable(share_pointer_example share_pointer_example.cpp)

# lambda
add_executable(lambda_example lambda_example.cpp)

# function wrapper
add_executable(function_wrapper_example function_wrapper_example.cpp)