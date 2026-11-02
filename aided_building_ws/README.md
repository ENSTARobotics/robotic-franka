# aided_building

Simplified project to simulate the Franka FR3 arm in Gazebo, controlled through `ros2_control`.

## Overview
- Only the FR3 arm (no gripper/tool)
- Simple ROS 2 communication through `aided_building_comm`
- Simulation in Gazebo Sim with `gz_ros2_control`
- Trajectory control with `JointTrajectoryController` (`fr3_arm_controller`)
- Configurable **SAFE_ZONE** and **WORK_ZONE**
- Operator's hand/PCB simulated in Gazebo
- Automatic move to **SAFE_ZONE** at startup

## Packages
- `aided_building_description`: FR3 URDF/xacro for Gazebo + `ros2_control` configuration, zone models/configurations
- `aided_building_bringup`: simulation launch (Gazebo, robot spawn, controllers, RViz, operator's hand)
- `aided_building_comm`: node that sends trajectories, auto-move to SAFE_ZONE, services to go to SAFE/WORK_ZONE

## Interfaces
| Interface | Type | Description |
| --- | --- | --- |
| `/franka/joint_targets` | `std_msgs/msg/Float64MultiArray` | Sends 7 joint positions (fr3_joint1..7) to the command node |
| `/franka/go_safe_zone` | `std_srvs/srv/Trigger` | Moves the robot to the SAFE_ZONE position |
| `/franka/go_work_zone` | `std_srvs/srv/Trigger` | Moves the robot to the WORK_ZONE position |
| `/franka/home` | `std_srvs/srv/Trigger` | Alias for `/franka/go_safe_zone` (kept for compatibility) |
| `/franka/stop` | `std_srvs/srv/Trigger` | Holds the current pose |
| `/joint_states` | `sensor_msgs/msg/JointState` | Joint states published by `joint_state_broadcaster` |
| `/fr3_arm_controller/joint_trajectory` | `trajectory_msgs/msg/JointTrajectory` | Trajectories sent by `franka_comm` to the controller |

## Zone configuration

The zones are set with **ROS parameters** (passed by the launch). There is also a reference file in `aided_building_description/config/zones.yaml` with profiles per scenario.

- **SAFE_ZONE**: initial/default safe position (away from the operator's hand). The robot goes to this position automatically at startup (configurable).
- **WORK_ZONE**: work position close to the operator's hand/PCB, where the robot moves after being in the SAFE_ZONE.

## How to run (step by step)

### 1) Set up the environment
```bash
source /opt/ros/jazzy/setup.bash
cd /path/to/aided_building_ws  # e.g. /work/aided_building_ws
source install/setup.bash
```

### 2) Build (if there are changes)
```bash
colcon build --symlink-install
source install/setup.bash
```

### 3) Start the simulation
```bash
ros2 launch aided_building_bringup sim.launch.py
```

Optional customization:
```bash
ros2 launch aided_building_bringup sim.launch.py \
  scenario:=scenario_1 \
  spawn_operator_hand:=true \
  hand_x:=0.4 hand_y:=0.0 hand_z:=0.25 \
  auto_move_to_safe_zone:=true startup_delay:=10.0 move_duration:=3.0 \
  safe_zone_joints:="[0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398]" \
  work_zone_joints:="[0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5]"
```

### 4) Check the controllers
```bash
ros2 control list_controllers
```
`joint_state_broadcaster` and `fr3_arm_controller` must be `active`.

### 5) Control the robot
```bash
# Go to SAFE_ZONE
ros2 service call /franka/go_safe_zone std_srvs/srv/Trigger {}

# Go to WORK_ZONE
ros2 service call /franka/go_work_zone std_srvs/srv/Trigger {}

# Send a custom position
ros2 topic pub --once /franka/joint_targets std_msgs/msg/Float64MultiArray "{data: [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5]}"

# Stop/hold
ros2 service call /franka/stop std_srvs/srv/Trigger {}
```

## Simulating the operator's hand
The hand/PCB is spawned from the SDF model in `aided_building_description/models/operator_hand.sdf`. Adjust it with `hand_x,y,z,hand_roll,hand_pitch,hand_yaw` or turn it off with `spawn_operator_hand:=false`. To calibrate `work_zone_joints`, move the robot to the desired position and read `/joint_states`.

## Checking/debugging
```bash
ros2 control list_controllers
ros2 topic echo /joint_states --once
ros2 node list
ros2 service list | grep franka
```

## Build
```bash
colcon build --symlink-install --packages-select aided_building_description aided_building_bringup aided_building_comm
source install/setup.bash
```

## Notes
- Only 7 joints (FR3), no gripper/tool
- Gazebo simulation only
- Only send commands after the controllers are `active`
- Environment: ROS 2 Jazzy, Gazebo Sim, `ros_gz`, `ros2_control`, `gz_ros2_control`, `franka_description`
