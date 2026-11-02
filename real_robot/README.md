# real_robot: the lab's Panda with frankapy

The lab's robot is a **Franka Emika Panda (FER), not an FR3**, and its libfranka version does not work
with `franka_ros2`. So this part does not use the ROS 2 workspace: it runs on **frankapy (ROS 1 Noetic)**
in the lab's Docker image `jgarcia10/u2is-franka-ros1`, and talks over UDP to programs on the host.

**First time with the robot (network, Desk, Docker, installation): see [SETUP.md](SETUP.md).**
What has been tested on the real robot is listed in the [main README](../README.md#status-november-2-2026).

```text
camera -> perception/face.py ---\
                                 +-> supervisor/supervisor.py -> screen/screen.py (screen)
mic    -> perception/voice.py --/      |  ^   ^   ^
                                       |  |   |   '-- dialogue/dialogue.py (speech, voice commands)
table camera -> perception/hands.py ---|--|---'
                                       v  |
                     robot/cycle_controller.py (Docker) -> Panda
```

## Contents

```text
robot/                    runs in the frankapy container
  cycle_controller.py     runs the cycle (--cycles N) or the routine over the PCB (--pcb)
  pcb_plan.py             Panda FK/IK in pure Python and grid plan (does not move anything)
  record_poses.py         records cycle poses or PCB corners by guiding the arm by hand
  communication_test.py   diagnoses communication_constraints_violation
  start_stack.sh          starts the frankapy stack in one terminal (used by launch.sh)
  examples/move_robot.py  frankapy example
supervisor/               runs on the host
  keyboard_supervisor.py  keys -> state; shows the robot's state and writes a CSV
  supervisor.py           perception -> T(t) -> state (automatic mode) + manual keys + hands lock
  trust.py                sensor fusion into T(t) and state machine (thresholds are here)
perception/               runs on the host (needs .venv)
  face.py                 webcam -> YuNet + mini-Xception -> stress and confidence -> UDP 5010
  voice.py                microphone -> log-mel CNN (RAVDESS) -> stress and confidence -> UDP 5010
  microphone.py           picks the microphone by name (default: C930e); voice.py and dialogue.py share it
  mini_xception.py        Joab's ROB311 network, converted from Keras to PyTorch
  convert_weights.py      verified Keras -> PyTorch conversion (only to redo it; needs TensorFlow)
  pcb.py                  checks that a non-white rectangle (the PCB) is in place
  hands.py                table camera -> YOLO hand detector -> hands near the PCB -> UDP 5016 (also the PCB check)
  detect_aruco.py         finds the dictionary and ids of the ArUco markers in a photo
  models/                 YuNet, mini-Xception, voice CNN, hand YOLO and their licenses
screen/screen.py          operator screen: camera, face and voice expression, state, robot
dialogue/                 runs on the host
  dialogue.py             the robot speaks English (Piper) and understands keywords (Vosk)
  download_models.py      downloads the Piper voice and Vosk model (not in git)
data/                     poses.json (rad), pose_info.json (tool tip and PCB corners),
                          table.json (ArUco markers, robot base frame)
launch.sh                 opens everything in terminal tabs (see Quick start)
logs/                     one CSV per session (not in git)
```

Everything is in English: program names, command-line options, code comments, the operator screen and the robot's speech.

## Quick start

Start from the **Ubuntu Terminal (Ctrl+Alt+T), not from the VS Code terminal**: `launch.sh` opens its programs in
`gnome-terminal` tabs, and those do not open from VS Code, so the script stops and tells you.

Before starting: in Desk, joints unlocked (blue light) and FCI active; emergency stop in your hand.

```bash
cd ~/robotic-franka/real_robot      # or wherever you cloned it
./launch.sh
```

1. **Stack tab:** the script opens it after you press Enter. Type your password and wait for `stack ready`, then go
   back and press Enter.
2. **Controller tab:** type your password, check what it prints and press Enter there. Then press Enter in the
   first terminal. The other tabs open: Supervisor, Face, Voice, Screen, Hands and Dialogue.
3. **Calibrate:** stay quiet for a few seconds. Then press `k` in the Supervisor tab, look at the camera and talk
   normally for 20 s. Repeat this for each person.
4. **Start:** put the PCB in place, keep your hands off it and answer *yes* when the robot asks. The `a` key in the
   Supervisor tab also starts it, without checking the PCB.
5. **Stop:** say *stop*, press `p` in the Supervisor tab, or use the emergency stop. Hands near the PCB also stop
   it and move it away; say *continue* once they are out.
6. **Shut down:** `q` in Supervisor and Ctrl+C in Stack (or `./launch.sh --stop`). Then, in Desk, deactivate FCI
   and lock the joints.

| Option | What it does |
|---|---|
| `--normal` | normal speed (0.6) instead of the slow one (0.3) |
| `--cycle` | cycle between recorded poses instead of the routine over the PCB |
| `--simulated` | no robot and no Docker |
| `--restart` | closes everything, robot programs included, and starts again; use it if a port is in use or something was left open |
| `--stop` | only closes everything |
| `--dry-run` | prints the commands without opening anything |

**Other terminals (not tested).** We only use the Ubuntu Terminal. Terminator should also work, because `launch.sh`
still opens its own `gnome-terminal` window. To see everything split on one screen, for example with Terminator,
tmux or vim's `:terminal`, get the commands from `./launch.sh --dry-run` and paste each one into its own pane,
in the order printed. None of these three is installed on the lab laptop.

## Manual control

```bash
# host
python3 supervisor/keyboard_supervisor.py --v-normal 0.6 --v-cautious 0.3

# container
cd /root/robotic-franka/real_robot/robot
python3 cycle_controller.py --pcb --max-scale 0.6     # or --cycles 1: cycle between recorded poses
```

Press Enter in the controller, then a key in the supervisor window:

| Key | State | Robot |
|---|---|---|
| `n` | NORMAL | speed `--v-normal` |
| `c` | CAUTIOUS | speed `--v-cautious`, from the next segment |
| `p` | PAUSED | `stop_skill()` and holds; `n` or `c` resumes the segment |
| `r` | RETRACT | stops, slowly lifts the tool 5 cm vertically, and waits |
| `q` | (quit) | sends PAUSED and closes |

If the controller gets no valid command for 0.5 s, it stops; it never carries on. The controller's
`--max-scale` caps whatever speed the supervisor asks for (default 0.3, for first tests). It must be at
least `--v-normal`, or the modes end up at the same speed.

## Automatic mode: the operator's state decides

From `real_robot/`, one terminal each:

```bash
.venv/bin/python perception/face.py                 # camera -> stress (--camera N, --show, --cpu)
.venv/bin/python perception/voice.py                    # microphone -> stress (--device, --list, --cpu)
python3 supervisor/supervisor.py --v-normal 0.6 --v-cautious 0.3
.venv/bin/python screen/screen.py                 # --full on the operator's monitor
.venv/bin/python dialogue/dialogue.py                   # robot speech
```

The supervisor starts in MANUAL and PAUSED, so the robot does not move until you allow it: `a` (or
answering *yes*) enables automatic mode; `n`, `c`, `p`, `r` take manual control at once.

**Calibration.** Press `k` in the supervisor with the operator calm, looking at the camera and talking normally.
For 20 s, with the robot paused and voice orders ignored, it measures each sensor's normal stress level for this
person; afterwards only stress above that level counts. Repeat it for each person: it is not saved between sessions.

**How `T(t)` is computed.** Each sensor sends `{"source": "face"|"audio", "stress": 0..1, "confidence": 0..1}`.

- Stress weights, the same for face and voice: fear, anger, disgust 1; sadness 0.6; surprise 0.5;
  happiness, neutral 0.
- Confidence is 1 minus the network's normalized entropy; for voice, also scaled by the share of the
  window that contains speech.
- In silence, `voice.py` repeats its last result while its confidence fades to 0 over 5 s, so it stays visible
  but stops counting. The background noise level is re-estimated continuously from the last 15 s.
- The supervisor takes the mean stress weighted by confidence and by sensor (face 70 %, voice 30 %), and
  `T(t) = 1 - stress`, smoothed. A small prior toward "no stress" keeps one unsure sensor from pausing the robot.

**State machine.** It goes down at once (`T < 0.6` CAUTIOUS, `T < 0.35` PAUSED) and up only after 2-3 s of
calm. With no valid perception for more than 2 s, it goes to CAUTIOUS at most. Thresholds are a starting
point, to be tuned in a pilot.

**Models.** Both are trained on acted datasets, so spontaneous faces and voices will score lower.

| Signal | Model | Dataset | Score |
|---|---|---|---|
| Face | mini-Xception (Joab's ROB311 lab) | FER-2013 | 61 % accuracy; confuses neutral with fear on sample faces |
| Voice | Log-mel CNN from [voice-emotion-recognition](https://github.com/JairVasquezT/voice-emotion-recognition), retrained with background noise | RAVDESS, English | About 66 % UAR on unseen actors at -60 to -20 dB noise (one fold of six). Without noise training it scored 70.4 % ± 5.3 on clean audio but called any room noise "disgust" |

## Robot speech (`dialogue.py`)

The robot greets and asks whether to begin ("Welcome! ... Shall we begin?"): *yes* or *ready* starts
automatic mode, *no* waits. During the task:

- *stop*, *wait* or *pause* stop at once ("Stopping. Sorry if I was going too fast." then "Shall I continue?").
- *slow down* goes slower for 20 s.
- It announces what the trust index decides ("I noticed some tension, so I'm slowing down.", "Back to
  normal speed.").

- It narrates the task ("Moving to point two of six.", "Working on point two.", "Returning to home.");
  `--no-narration` turns this off.

While it speaks, the microphone is ignored, so it does not hear itself. The exception is narration, when
only "stop" is accepted, so that the operator can still stop it. Narration phrases were checked against the
recognizer with repetitions: words like "start" and "soldering" can be misheard as "stop". **Voice "stop" is
an additional channel: the emergency stop remains the main one.**

## UDP ports (all on 127.0.0.1)

| Port | From | To |
|---|---|---|
| 5005 | supervisor | robot |
| 5006 | robot | supervisor |
| 5010 | face and voice | supervisor |
| 5011 | supervisor | screen |
| 5012 | camera image | screen |
| 5013 | dialogue | supervisor |
| 5014 | supervisor | dialogue |
| 5015 | PCB check | dialogue |
| 5016 | hand check | supervisor |

## PCB check (`pcb.py`, optional)

The camera that sees the table finds the 4 ArUco markers, which locate the recorded PCB corners in the image.
Inside that zone the program looks for a rectangular shape of a different color from the white base, so it needs
no training or reference photo. A hand does not count, because it is not rectangular; a white board would not
be detected.

```bash
.venv/bin/python perception/pcb.py --show     # shows the zone, how much it is covered and how rectangular
```

With it running, the robot asks for the PCB until it is there ("Please place the PCB on the table.", then "PCB
detected. Shall we begin?"). Without it, the robot greets as usual. Until the routine starts it keeps checking: if
the PCB stops being visible, a hand over it included, the robot says so and *yes* does not start it. During the
routine it does not check, because the arm itself covers the PCB.

## Hands near the PCB (`hands.py`)

The table camera (the C930e, which also sees the ArUco markers) runs a YOLOv8n hand detector
(`perception/models/hands_yolov8n.pt`, from the team's YOLO_hands; see `HANDS_YOLO.txt`). The markers place the
recorded PCB corners in the image, and the watched zone is the PCB plus a margin (5 cm, `--margin-cm`). A hand
counts when its box touches that zone. It is "near" with hands in 2 of the last 3 images (10 Hz) and clears after
1 s without them.

```bash
.venv/bin/python perception/hands.py --show     # zone in green (red with hands), detected hands in orange
```

A camera can only be opened by one program, so `hands.py` also does the PCB check and sends what `pcb.py`
sends; `launch.sh` opens `hands.py` instead of `pcb.py`. Use `--no-pcb` if `pcb.py` runs on another camera.

What happens, in automatic and in manual mode:

1. Hands near the PCB while the robot is allowed to move (NORMAL or CAUTIOUS): the supervisor sends RETRACT, the
   controller stops and lifts the tool 5 cm, and the robot says "Careful, your hands are near the board. I'm
   pausing and moving away."
2. It stays there while there are hands, and also after they leave: "Thank you, the area is clear. Say continue
   when you are ready."
3. *Continue* (or *yes*, *ready*) resumes slowly (CAUTIOUS for 20 s), like after a *stop*. With hands still there
   it answers "Please move your hands away from the board first." and does not move. In the Supervisor tab,
   `a`, `n` or `c` also resume, with the same check.

Before the routine starts (robot paused), hands do not trigger anything: the operator is placing the PCB. If the
robot is paused for another reason, the block starts as soon as something tries to resume with hands there.
Without `hands.py` running nothing is watched, and the screen and the supervisor show "hands --". The detector
is trained on generic photos of hands: check it with `--show` on the lab table before trusting it, and adjust
`--conf` (0.4) or `--margin-cm` if the gripper or the pen are taken for a hand. **It is an additional channel:
the emergency stop remains the main one.**

## Without the robot

```bash
python3 robot/pcb_plan.py                                  # check the PCB plan
python3 robot/cycle_controller.py --simulated --pcb        # fake arm, same UDP ports
.venv/bin/python dialogue/dialogue.py --keyboard --mute         # type words; speech shown as text
```

The poses and corners in `data/` belong to our table. If the table or the PCB moves, record them again
with `robot/record_poses.py` (`--corners` for the PCB).

## Known issues

- `communication_constraints_violation`, the robot does not move, or guide mode stays stiff: almost always
  the network. The link to the control box must run at 1000 Mb/s with a stable ping below 1 ms. Our laptop's
  built-in Realtek card (100 Mb/s, peaks above 1 ms) failed; a gigabit USB-Ethernet adapter works.
  Check with `python3 robot/communication_test.py 1`.
- `stop_skill()` stops abruptly, with a visible vibration. Smooth braking needs frankapy's dynamic mode
  (`dynamic=True`), not implemented yet.
