# Topic Introduction

- Show useful commands

```bash
# Run the turtlesim node
ros2 run turtlesim turtlesim_node

# Output information about a node
ros2 node info /turtlesim

# Output messages from a topic
ros2 topic echo /turtle/pose

# Print information about a topic
ros2 topic info /turtle1/cmd_vel

# Output the interface definition
ros2 interface show geometry_msgs/msg/Twist

# Publish a message to a topic
ros2 topic pub /turtle1/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5, y: 0.5}, angular: {z: 0.5}}"

```

- ROS Coordinate

```plaintext
               +X (Up)
                  ^
                  |   -Z (Forward)
                  |  /
                  | /
                  |/
   +Y (Right) <---+---> -Y (Left)
                 /|
                / |
               /  |
              v   v
    +Z (Backward) -X (Down)

```
