# Setup: the lab's Panda (U2IS)

Based on the *Franka Panda Tutorial - U2IS* (Caio Conti, Oct. 2026) and what we found while using it.
**Desk credentials are not here** because this repository is public: they are in that tutorial, or ask
the lab's supervisor.

Only tested on Ubuntu 24.04 with Python 3.12.

## 1. Network

Run an Ethernet cable **directly** from your computer to the robot's **control box**, not to the arm's
base. Set a static IP on that interface:

| Field | Value |
|---|---|
| Address | `147.250.35.100` |
| Netmask | `255.255.255.0` |
| Gateway | none, so Wi-Fi keeps providing internet. In Ubuntu: "Use this connection only for resources on its network" |

Check it:

```bash
ping 147.250.35.11
ip route get 147.250.35.11               # shows the interface, after "dev"
cat /sys/class/net/<interface>/speed     # must print 1000
```

**The link must run at 1000 Mb/s, with a stable ping below 1 ms.** With a card that negotiated 100 Mb/s
(ping peaks above 1 ms), the robot aborted every motion with `communication_constraints_violation`. A
gigabit USB-Ethernet adapter fixed it.

## 2. Desk (the robot's web interface)

1. Open http://147.250.35.11 in **Chrome**. Firefox did not work.
2. The robot starts locked (yellow light): unlock the joints under *Joints*, and the light turns **blue**.
3. **Activate FCI**, which is required to control it from a computer.

When you finish: deactivate FCI and lock the joints.

## 3. Code and Docker with frankapy (ROS 1)

Get the code. The English version with the hand detection is in this repository (default branch
`feat/manos-cerca`); Jair's fork, https://github.com/JairVasquezT/robotic-franka (branch `feat/percepcion`),
still has the earlier version with Spanish names.

```bash
sudo apt install git docker.io
git clone https://github.com/joabzicg/robotic-franka.git ~/robotic-franka
```

The lab's Panda is a **FER** and does not work with `franka_ros2`. Use the lab's image with frankapy:

```bash
sudo docker pull jgarcia10/u2is-franka-ros1:latest
```

Create the container **sharing your copy of this repository**. Without `-v` it cannot see `real_robot/`:

```bash
sudo docker run -it --name franka_container \
  -v ~/robotic-franka:/root/robotic-franka \
  --net=host --ipc=host --ulimit rtprio=99 --ulimit memlock=-1 --cap-add=sys_nice \
  --security-opt seccomp=unconfined jgarcia10/u2is-franka-ros1:latest
```

`--net=host` is required: the programs talk over `127.0.0.1`. If you already have a `franka_container`
created without `-v`, create it again (`sudo docker rm franka_container` and repeat the `run`).

After that:

```bash
sudo docker start franka_container
sudo docker exec -it franka_container bash      # once per terminal
```

## 4. Host programs (face, voice, hands, screen and dialogue)

The keyboard supervisor needs none of this, and the controller runs in the container. Without an NVIDIA
GPU, run `face.py` and `voice.py` with `--cpu`.

```bash
sudo apt install python3.12-venv libportaudio2
cd ~/robotic-franka/real_robot
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python dialogue/download_models.py
.venv/bin/python -c "import cv2, torch, librosa, sounddevice, PIL, piper, vosk, ultralytics; print('CUDA:', torch.cuda.is_available())"
```

Choose your speakers in *Settings → Sound*. Cameras and microphones are chosen by name:

- `face.py` looks for a Logitech C920. On another computer use `--camera <name or number>`; if it cannot
  find one, the error lists those available. Each webcam shows up as two `/dev/video*` devices: use the first.
- `voice.py` and `dialogue.py` look for a Logitech C930e and fall back to the system microphone, with a warning.
  Both print the microphone in use. Choose another with `--device <name>`, or list them with
  `voice.py --list`. Prefer a microphone close to the speaker: a noisy one detects speech poorly.

## 5. Start the stack (inside the container)

From the second time on, `./launch.sh`, run from the Ubuntu Terminal, does this for you: see
[Quick start](README.md#quick-start). The first time, do it by hand so you can see each step.

In **each** container terminal:

```bash
cd /root/frankapy/catkin_ws && source devel/setup.bash
```

One terminal each, **in this order**:

```bash
roscore
~/franka-interface/build/franka_interface --robot_ip 147.250.35.11
roslaunch franka_ros_interface franka_ros_interface.launch robot_ip:=147.250.35.11
roslaunch franka_gripper franka_gripper.launch robot_ip:=147.250.35.11 __ns:=/franka_gripper_1
```

To shut down: Ctrl+C in reverse order (gripper, ros_interface, franka_interface, roscore).

These messages are normal: `autolab_core not installed as catkin package`, and
`Will try to acquire lock while setting franka_interface status`.

## 6. Test

In a fifth container terminal, with your hand on the emergency stop:

```bash
cd /root/robotic-franka/real_robot/robot
python3 communication_test.py 1      # turns the wrist 20° and back
```

If that works, continue with the [README](README.md).

The poses and corners in `data/` belong to our table. If the table or the PCB moved, or the pen changed,
record them again (`robot/record_poses.py`) and check the plan (`python3 robot/pcb_plan.py`).

Notes:

- `fa.get_gripper_width()` fails in this setup, including in the tutorial's `move_robot.py`: comment it out.
- frankapy has virtual walls. `Target joints in collision with virtual walls!` means the target is outside
  its allowed space; it is not a communication failure.
- If the robot does not move, guide mode stays stiff, or you see `done_skill_id error` or
  `Can't parse message of type "RobotStateMessage"`, look at the `franka_interface` terminal. If it says
  `communication_constraints_violation`, check the network (section 1).
