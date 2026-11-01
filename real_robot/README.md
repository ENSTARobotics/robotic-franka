# real_robot: the lab's Panda with frankapy

The lab's robot is a **Franka Emika Panda (FER), not an FR3**, and its libfranka version does not work
with `franka_ros2`. So this part does not use the ROS 2 workspace: it runs on **frankapy (ROS 1 Noetic)**
in the lab's Docker image `jgarcia10/u2is-franka-ros1`, and talks over UDP to programs on the host.

**First time with the robot (network, Desk, Docker, installation): see [SETUP.md](SETUP.md).**
What has been tested on the real robot is listed in the [main README](../README.md#status-november-1-2026).

```text
camera  -> percepcion/rostro.py -\
                                  +-> supervisor/supervisor.py -> pantalla/pantalla.py (screen)
mic     -> percepcion/voz.py ----/       |  ^      ^
                                         |  |      '-- dialogo/dialogo.py (speech, voice commands)
                                         v  |
                          robot/controlador_ciclo.py (Docker) -> Panda
```

## Contents

```text
robot/                    runs in the frankapy container
  controlador_ciclo.py      runs the cycle (--ciclos N) or the routine over the PCB (--placa)
  plan_placa.py             Panda FK/IK in pure Python and grid plan (does not move anything)
  grabar_poses.py           records cycle poses or PCB corners by guiding the arm by hand
  prueba_comunicacion.py    diagnoses communication_constraints_violation
  ejemplos/move_robot.py    frankapy example
supervisor/               runs on the host
  supervisor_teclado.py     keys -> state; shows the robot's state and writes a CSV
  supervisor.py             perception -> T(t) -> state (automatic mode) + manual keys
  confianza.py              sensor fusion into T(t) and state machine (thresholds are here)
percepcion/               runs on the host (needs .venv)
  rostro.py                 webcam -> YuNet + mini-Xception -> stress and confidence -> UDP 5010
  voz.py                    microphone -> log-mel CNN (RAVDESS) -> stress and confidence -> UDP 5010
  mini_xception.py          Joab's ROB311 network, converted from Keras to PyTorch
  convertir_pesos.py        verified Keras -> PyTorch conversion (only to redo it; needs TensorFlow)
  detectar_aruco.py         finds the dictionary and ids of the ArUco markers in a photo
  modelos/                  YuNet, mini-Xception, voice CNN and their licenses
pantalla/pantalla.py      operator screen: camera, face and voice expression, state, robot
dialogo/                  runs on the host
  dialogo.py                the robot speaks English (Piper) and understands keywords (Vosk)
  descargar_modelos.py      downloads the Piper voice and Vosk model (not in git)
datos/                    poses.json (rad), poses_info.json (tool tip and PCB corners),
                          mesa.json (ArUco markers, robot base frame)
logs/                     one CSV per session (not in git)
```

Program names, command-line options and code comments are in Spanish; the operator screen too.

## Manual control

```bash
# host
python3 supervisor/supervisor_teclado.py --v-normal 0.6 --v-cautious 0.3

# container
cd /root/robotic-franka/real_robot/robot
python3 controlador_ciclo.py --placa --escala-max 0.6     # or --ciclos 1: cycle between recorded poses
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
`--escala-max` caps whatever speed the supervisor asks for (default 0.3, for first tests). It must be at
least `--v-normal`, or the modes end up at the same speed.

## Automatic mode: the operator's state decides

From `real_robot/`, one terminal each:

```bash
.venv/bin/python percepcion/rostro.py                 # camera -> stress (--camara N, --ver, --cpu)
.venv/bin/python percepcion/voz.py                    # microphone -> stress (--dispositivo C920, --cpu)
python3 supervisor/supervisor.py --v-normal 0.6 --v-cautious 0.3
.venv/bin/python pantalla/pantalla.py                 # --completa on the operator's monitor
.venv/bin/python dialogo/dialogo.py                   # robot speech
```

The supervisor starts in MANUAL and PAUSED, so the robot does not move until you allow it: `a` (or
answering *yes*) enables automatic mode; `n`, `c`, `p`, `r` take manual control at once.

**How `T(t)` is computed.** Each sensor sends `{"fuente": "rostro"|"audio", "estres": 0..1, "confianza": 0..1}`.

- Stress weights, the same for face and voice: fear, anger, disgust 1; sadness 0.6; surprise 0.5;
  happiness, neutral 0.
- Confidence is 1 minus the network's normalized entropy; for voice, also scaled by the share of the
  window that contains speech.
- The supervisor takes the confidence-weighted mean stress, and `T(t) = 1 - stress`, smoothed.

**State machine.** It goes down at once (`T < 0.6` CAUTIOUS, `T < 0.35` PAUSED) and up only after 2-3 s of
calm. With no valid perception for more than 2 s, it goes to CAUTIOUS at most. Thresholds are a starting
point, to be tuned in a pilot.

**Models.** Both are trained on acted datasets, so spontaneous faces and voices will score lower.

| Signal | Model | Dataset | Score |
|---|---|---|---|
| Face | mini-Xception (Joab's ROB311 lab) | FER-2013 | 61 % accuracy; confuses neutral with fear on sample faces |
| Voice | Log-mel CNN from [voice-emotion-recognition](https://github.com/JairVasquezT/voice-emotion-recognition) | RAVDESS, English | 70.4 % ± 5.3 UAR on unseen actors (neutral 83 %, anger 85 %, fear 73 %) |

## Robot speech (`dialogo.py`)

The robot greets and asks whether to begin ("Welcome! ... Shall we begin?"): *yes* or *ready* starts
automatic mode, *no* waits. During the task:

- *stop*, *wait* or *pause* stop at once ("Stopping. Sorry if I was going too fast." then "Shall I continue?").
- *slow down* goes slower for 20 s.
- It announces what the trust index decides ("I noticed some tension, so I'm slowing down.", "Back to
  normal speed.").

While it speaks, the microphone is ignored, so it does not hear itself. **Voice "stop" is an additional
channel: the emergency stop remains the main one.**

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

## Without the robot

```bash
python3 robot/plan_placa.py                                  # check the PCB plan
python3 robot/controlador_ciclo.py --simulado --placa        # fake arm, same UDP ports
.venv/bin/python dialogo/dialogo.py --teclado --mudo         # type words; speech shown as text
```

The poses and corners in `datos/` belong to our table. If the table or the PCB moves, record them again
with `robot/grabar_poses.py` (`--esquinas` for the PCB).

## Known issues

- `communication_constraints_violation`, the robot does not move, or guide mode stays stiff: almost always
  the network. The link to the control box must run at 1000 Mb/s with a stable ping below 1 ms. Our laptop's
  built-in Realtek card (100 Mb/s, peaks above 1 ms) failed; a gigabit USB-Ethernet adapter works.
  Check with `python3 robot/prueba_comunicacion.py 1`.
- `stop_skill()` stops abruptly, with a visible vibration. Smooth braking needs frankapy's dynamic mode
  (`dynamic=True`), not implemented yet.
