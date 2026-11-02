# aided_building_gripper

Control of the Franka Hand with **the same interface in Gazebo and on the real robot**.

| Mode | Who provides the interface |
| --- | --- |
| Gazebo (`use_sim:=true`) | `sim_gripper_server` (this package) |
| Real (`use_sim:=false`) | `franka_gripper_node` (official, `franka_ros2`) |

The team's code always calls the same actions. Only the launch changes.

## Interface (namespace `/franka_gripper`)

| Name | Type | Use |
| --- | --- | --- |
| `homing` | `franka_msgs/action/Homing` | Opens fully (on the real one, it also calibrates) |
| `move` | `franka_msgs/action/Move` | `width` [m], `speed` [m/s] |
| `grasp` | `franka_msgs/action/Grasp` | `width` [m], `force` [N], `speed`, `epsilon.inner/outer` [m] |
| `gripper_action` | `control_msgs/action/GripperCommand` | For MoveIt: `position` = width/2 |
| `stop` | `std_srvs/srv/Trigger` | Stops the current motion |
| `joint_states` | `sensor_msgs/msg/JointState` | State of the fingers |

`grasp` rule (same as `libfranka`): the gripper closes until it finds the object.
The grasp succeeds if the measured width `d` satisfies
`width - epsilon.inner <= d <= width + epsilon.outer`.
For objects of variable size, use a larger `epsilon`.

Example with a soldering iron handle about 20 mm in diameter:

```bash
ros2 run aided_building_gripper gripper_cli grasp 0.020 30 0.05 0.005
```

## Running

```bash
# Full simulation (arm + gripper)
ros2 launch aided_building_bringup sim.launch.py

# Only the gripper, on the real robot
ros2 launch aided_building_gripper gripper.launch.py use_sim:=false robot_ip:=<IP>
```

If the real arm is launched with `franka_bringup` and `load_gripper:=true`, it already starts
`franka_gripper`. In that case, do **not** run the launch above (the node would be duplicated).

## Quick test

```bash
ros2 run aided_building_gripper gripper_cli homing
ros2 run aided_building_gripper gripper_cli move 0.03
ros2 run aided_building_gripper gripper_cli grasp 0.02 20
ros2 run aided_building_gripper gripper_cli open
ros2 run aided_building_gripper gripper_cli stop
```

### With an object

`object:=true` places a 20 mm cylinder (50 g) on a thin stand, between the open fingers at
the arm's initial pose (SAFE_ZONE). The default is `false`, because the stand would block
the arm.

```bash
ros2 launch aided_building_bringup sim.launch.py object:=true
ros2 run aided_building_gripper gripper_cli grasp 0.02 20   # success, fingers at ~0.010 m
ros2 run aided_building_gripper gripper_cli grasp 0.04 20   # failed: 20 mm is outside 40 ± 5 mm
ros2 run aided_building_gripper gripper_cli move 0.0        # failed: blocked by the object
# Lift 10 cm with the object held:
ros2 topic pub --once /franka/joint_targets std_msgs/msg/Float64MultiArray \
  "{data: [0.0, -0.6861, 0.0, -2.0279, 0.0, 1.3418, 0.7854]}"
```

Without a graphical interface (server, CI): `gz_args:="-r -s empty.sdf" use_rviz:=false`.
In Docker with `--net=host`, export `GZ_IP=127.0.0.1` before the launch; without it,
`ros_gz_sim create` may not find the world (it stays at "Requesting list of world names").

## Build

`franka_msgs` is needed in both modes. `franka_gripper` (with `libfranka`) is only
needed in real mode.

```bash
vcs import src < dependencies.repos
# Simulation only (does not build libfranka):
colcon build --symlink-install --packages-up-to aided_building_bringup
```

## Limitations of the simulation

- **The force is not applied.** Gazebo's position control does not regulate force. The
  `force` field is validated (0 to 70 N), but ignored. After contact, the fingers are held
  `grasp_squeeze` (2 mm) inside the object, and the resulting force depends on Gazebo's physics.
- The speed uses the wall clock. If Gazebo runs slower than real time, the motion will look
  faster in simulated time.
- To hold objects in Gazebo, they need consistent collision and mass. The fingers already
  have the pads' friction (`mu = 1.13`, the value from the official `franka_gazebo`).

## Implementation (sim)

- The URDF turns on the Franka Hand (`hand=true`). Only `fr3_finger_joint1` gets a position
  command. `fr3_finger_joint2` is a mimic and is in the `ros2_control` block with state
  interfaces only; without that, `gz_ros2_control` does not move it. Gazebo's warning
  `Attempting to create a mimic constraint ... does not support mimic constraints` is
  harmless: `gz_ros2_control` itself makes finger 2 follow finger 1.
- Gazebo (DART) sticks a prismatic joint that stops exactly at its limit. That is why the
  fingers start at 0.0395 (and not 0.04) and the fully open setpoint is `limit_margin`
  (1 mm of width) below 0.08 m. `homing`/`open` still succeed (2 mm tolerance).
- The controller is `fr3_gripper_controller` (`JointGroupPositionController`). The server
  generates a ramp of setpoints at the requested speed and detects stalling (contact).
- Parameters: `config/sim_gripper.yaml`.
