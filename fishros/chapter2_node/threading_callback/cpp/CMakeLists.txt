cmake_minimum_required(VERSION 3.16)
project(threading_callback_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

add_executable(threading_callback_example threading_callback_example.cpp)

target_include_directories(threading_callback_example PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(threading_callback_example PRIVATE
    Threads::Threads
)

target_compile_options(threading_callback_example PRIVATE -Wall -Wextra)