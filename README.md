# Aided Building

**Human-Robot Collaborative Manual Soldering & Dynamic Trust Assessment.** HRI course project (Prof. Adriana Tapus).

A Franka Emika Panda performs a mock soldering task (a pen on a cardboard PCB, never a hot iron)
next to a human operator. The system estimates the operator's stress from their face and voice,
fuses it into a trust index `T(t)`, adapts the robot's speed or stops it, and explains on a screen
and by voice what it is doing and why.

The code that runs on the lab's robot is in [`real_robot/`](real_robot/). To connect and install,
follow [`real_robot/SETUP.md`](real_robot/SETUP.md).

## Status (November 2, 2026)

| Part | Status | Code |
|---|---|---|
| Robot control over UDP (speed, pause, vertical retreat, stop on lost link) | Works on the real robot | `real_robot/robot/` |
| Soldering routine over a recorded PCB (grid of points, FK/IK) | Works on the real robot | `real_robot/robot/plan_placa.py` |
| Keyboard supervisor with CSV logging | Works on the real robot | `real_robot/supervisor/supervisor_teclado.py` |
| Facial expression to stress (GPU) | Works live with a webcam, not yet driving the real robot | `real_robot/percepcion/rostro.py` |
| Voice emotion to stress (GPU), English model | Works live with a microphone, not yet driving the real robot | `real_robot/percepcion/voz.py` |
| Trust index `T(t)` and NORMAL / CAUTIOUS / PAUSED state machine | Works live with a simulated robot, not yet on the real robot | `real_robot/supervisor/` |
| Operator screen | Works live with a simulated robot, not yet on the real robot | `real_robot/pantalla/` |
| Spoken dialogue, task narration and voice commands ("stop", "slow down") | Works live with a simulated robot, not yet on the real robot | `real_robot/dialogo/` |
| Hands near the PCB: stop, move away, wait for "continue" (YOLO hand detector) | Tested with recorded commands and a simulated robot; not yet with the lab camera or the real robot | `real_robot/percepcion/manos.py` |
| Hand detection (MediaPipe) | ROS 2 package, not integrated with the real robot | `aided_building_ws/src/aided_building_hand_perception/` |
| Gazebo simulation of an FR3 | Simulation only | `aided_building_ws/` |

Not built yet: hand distance as a trust input (hands only stop the robot), smooth braking (`stop_skill()` stops abruptly),
per-participant calibration, and the user study.

## Who did what

| Member | Contribution | State |
|---|---|---|
| Jair Vasquez | Real-robot control with frankapy, UDP bridge, soldering routine and kinematics, trust index and state machine, operator screen, spoken dialogue, face and voice real-time integration, RAVDESS retraining of the voice model, setup guide | `real_robot/`: see the table above |
| Lucas Emanuel | Hand detection with MediaPipe and ArUco calibration; voice CNN design and EmoDB training ([voice-emotion-recognition](https://github.com/JairVasquezT/voice-emotion-recognition)) | Hand detection needs ROS 2 and the robot's TF, so it is not integrated yet |
| Gustavo | Initial ROS 2 and Gazebo base, `aided_building_comm`, description and bringup packages | Simulation of an FR3; does not drive the lab's Panda |
| Edilberto | Gripper interface (`aided_building_gripper`) and `real.launch.py` | Gripper simulated in Gazebo; `real.launch.py` needs `franka_ros2`, which does not work with the lab's Panda |

Joab built the facial expression recognition (mini-Xception on FER-2013, ROB311 lab); it was converted
from Keras to PyTorch here.

## Why two software stacks

The lab's robot is a **Panda (FER), not an FR3**, and its libfranka version is incompatible with
`franka_ros2`. So:

- **`real_robot/`** runs on the robot: frankapy (ROS 1 Noetic) in the lab's Docker image, plus plain
  Python programs on the host that talk over UDP.
- **`aided_building_ws/`** is the original ROS 2 Jazzy workspace for Gazebo, with FR3 joint names. It
  does not control the lab's robot. See [its README](aided_building_ws/README.md).

Moving the supervisor to ROS 2 nodes, to join both stacks, is a possible later phase.

## Repository layout

```text
real_robot/              code for the lab's robot (see its README)
  robot/                   controller, kinematics, pose recording (runs in Docker)
  supervisor/              trust index, state machine, keyboard control
  percepcion/              face and voice models
  pantalla/                operator screen
  dialogo/                 robot speech and voice commands
  datos/                   recorded poses, PCB corners, ArUco positions
aided_building_ws/       ROS 2 + Gazebo workspace (simulation)
proj-idea.md             original project concept
```

## Safety

This is a course prototype, not a certified system. Someone must hold the emergency stop whenever the
robot is active. Voice "stop" is an additional channel, never the main one. Hand detection
(`real_robot/percepcion/manos.py`) stops the robot and moves it away when hands come near the PCB, but it has
not been validated in the lab and only watches the PCB zone, not the whole arm. No validation against ISO/TS 15066
has been done.
