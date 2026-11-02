#!/usr/bin/env python3
"""Control panel: one window to start, run and stop everything, without typing commands.

    python3 panel/panel.py              # serves the panel on http://127.0.0.1:8765
    panel/open_panel.sh                 # starts it if needed and opens it as a Chrome app window
    panel/open_panel.sh --install       # adds "Aided Building" to the Ubuntu app menu

Only the Python standard library. It starts and watches the same programs as launch.sh: the frankapy
stack and the controller in Docker (or the simulated controller), supervisor.py without a terminal
(keys go over UDP 5017), and face, voice, hands, screen and dialogue. It reads the supervisor's summary
on UDP 5018 to show the state.

Docker: if your user cannot use docker, each Docker step asks for your password in a system window
(pkexec). To avoid that, see SETUP.md (docker group).

Safety: it only listens on 127.0.0.1, and every action needs the token embedded in the page, so other
web pages cannot drive the robot. The emergency stop is still the main stop.
"""
import argparse
import collections
import glob
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "robot"))
import pcb_plan

PANEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(PANEL_DIR, os.pardir))           # real_robot/
PY = os.path.join(ROOT, ".venv", "bin", "python")
CONTAINER = "franka_container"
CONTAINER_PATH = "/root/" + os.path.relpath(ROOT, os.path.expanduser("~"))
ROS_SETUP = "source /root/frankapy/catkin_ws/devel/setup.bash"
ROBOT_IP = "147.250.35.11"
IP = "127.0.0.1"
PORT = 8765
CONTROL_PORT = 5017         # keys for supervisor.py
HANDS_PORT = 5016           # hands messages for supervisor.py (virtual hands, simulated only)
VIRTUAL_HAND_R = 0.04       # m, radius of a virtual hand on the table
VIRTUAL_POINT_R = 0.03      # m around each soldering point, like hands.py --point-radius-cm 3
GUI_PORT = 5018             # supervisor.py's summary
LOG_LINES = 400
STACK_TIMEOUT = 60.0        # s to wait for "stack ready"
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

MODULES = ["face", "voice", "hands", "screen", "dialogue"]
SIM_WEBCAM = "C920"          # simulated: this webcam (camera and microphone) if it is plugged in, else the laptop's


def sim_devices():
    """(camera, microphone) for the simulation: the C920 if connected, otherwise the laptop's own."""
    camera, mic = "0", "system"
    for path in glob.glob("/sys/class/video4linux/video*"):
        try:
            with open(os.path.join(path, "name")) as f_name, open(os.path.join(path, "index")) as f_index:
                if SIM_WEBCAM in f_name.read() and f_index.read().strip() == "0":
                    camera = SIM_WEBCAM
        except OSError:
            pass
    r = subprocess.run(["pactl", "list", "short", "sources"], capture_output=True, text=True)
    if SIM_WEBCAM in r.stdout:
        mic = SIM_WEBCAM
    return camera, mic


def sim_args(name):
    """Simulated: a camera can only be opened by one program, so face.py lends its images to hands.py, which
    watches the whole image (there are no table markers)."""
    camera, mic = sim_devices()
    return {"face": ["--camera", camera, "--share-frames"], "voice": ["--device", mic],
            "hands": ["--frames-from-face", "--no-markers"], "dialogue": ["--device", mic]}.get(name, [])
TITLES = {"stack": "frankapy stack", "controller": "Controller", "supervisor": "Supervisor", "face": "Face",
          "voice": "Voice", "hands": "Hands + PCB", "screen": "Operator screen", "dialogue": "Dialogue",
          "comm_test": "Communication test", "pcb_plan": "PCB plan check"}
KEYS = {"a": "automatic", "n": "manual normal", "c": "manual slower", "p": "pause", "r": "move away",
        "k": "calibrate"}


def docker_ok():
    """True if this user can talk to Docker without a password."""
    try:
        return subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=5).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def docker(*args):
    """Docker command line: direct, or through pkexec (password window) without permission."""
    return (["docker"] if docker_ok() else ["pkexec", "docker"]) + list(args)


class Process(object):
    """A program started by the panel, with its output kept in memory."""

    def __init__(self, name):
        self.name = name
        self.popen = None
        self.lines = collections.deque(maxlen=LOG_LINES)
        self.partial = ""
        self.started = None
        self.lock = threading.Lock()

    def running(self):
        return self.popen is not None and self.popen.poll() is None

    def exit_code(self):
        return None if self.popen is None else self.popen.poll()

    def log(self, text):
        with self.lock:
            self.lines.append(time.strftime("%H:%M:%S ") + text)

    def start(self, cmd, cwd=ROOT, stdin=False):
        if self.running():
            return False
        self.lines.clear()
        self.log("$ " + " ".join(cmd))
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        # Started in the background (app menu, nohup), SIGINT arrives ignored and the children inherit it: they
        # would not stop with SIGINT, and the supervisor would not send PAUSED on closing. Restore it.
        self.popen = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
                                      start_new_session=True,
                                      preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
        self.started = time.time()
        threading.Thread(target=self._read, args=(self.popen,), daemon=True).start()
        return True

    def _read(self, popen):
        # Programs redraw their status line with \r: keep only complete lines, and the last redraw
        while True:
            data = popen.stdout.read1(4096) if hasattr(popen.stdout, "read1") else popen.stdout.read(1)
            if not data:
                break
            text = ANSI.sub("", (self.partial + data.decode("utf-8", "replace")))
            parts = re.split(r"[\r\n]", text)
            self.partial = parts.pop()
            with self.lock:
                for part in parts:
                    if part.strip():
                        self.lines.append(time.strftime("%H:%M:%S ") + part)
        code = popen.wait()
        self.log("[finished, code %s]" % code)

    def write(self, text):
        if self.running() and self.popen.stdin:
            self.popen.stdin.write(text.encode())
            self.popen.stdin.flush()

    def interrupt(self):
        if self.running():
            try:
                os.killpg(self.popen.pid, signal.SIGINT)
            except ProcessLookupError:
                pass

    def wait(self, timeout):
        t0 = time.time()
        while self.running() and time.time() - t0 < timeout:
            time.sleep(0.1)
        return not self.running()

    def kill(self):
        if self.running():
            try:
                os.killpg(self.popen.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def tail(self, n=LOG_LINES):
        with self.lock:
            lines = list(self.lines)[-n:]
        if self.partial.strip():
            lines.append("  " + ANSI.sub("", self.partial).strip())
        return lines

    def last(self):
        lines = self.tail(1)
        return lines[-1][9:] if lines else ""


class Panel(object):
    def __init__(self):
        self.processes = {n: Process(n) for n in TITLES}
        self.settings = {"simulated": False, "routine": "pcb", "speed": "slow",
                         "modules": {"face": True, "voice": True, "hands": True, "screen": True, "dialogue": True}}
        self.summary, self.t_summary = None, 0.0
        self.network = {}
        self.docker = {"ok": None, "container": None}
        self.busy = ""              # what the panel is doing right now (start-up, stop...)
        self.events = collections.deque(maxlen=200)
        self.lock = threading.Lock()
        # Virtual hands (simulated): two points on the table that the page drags; at rest by the operator's side
        self.virtual = {"enabled": False, "hands": [[0.78, -0.16], [0.78, 0.16]], "near": [False, False]}
        self.virtual_points = None
        threading.Thread(target=self._send_virtual_hands, daemon=True).start()
        threading.Thread(target=self._listen_summary, daemon=True).start()
        threading.Thread(target=self._watch, daemon=True).start()

    # ---------- background ----------

    def _listen_summary(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((IP, GUI_PORT))
        while True:
            data, _ = sock.recvfrom(65536)
            try:
                self.summary, self.t_summary = json.loads(data.decode("utf-8")), time.time()
            except ValueError:
                pass

    def _watch(self):
        while True:
            self.network = self._check_network()
            ok = docker_ok()
            container = None
            if ok:
                r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER],
                                   capture_output=True, text=True)
                container = r.stdout.strip() == "true" if r.returncode == 0 else "missing"
            self.docker = {"ok": ok, "container": container, "stack": self._stack_running()}
            time.sleep(2.0)

    @staticmethod
    def _check_network():
        out = {"iface": None, "speed": None, "ping_ms": None}
        r = subprocess.run(["ip", "route", "get", ROBOT_IP], capture_output=True, text=True)
        m = re.search(r"dev (\S+)", r.stdout)
        if m:
            out["iface"] = m.group(1)
            try:
                with open("/sys/class/net/%s/speed" % out["iface"]) as f:
                    out["speed"] = int(f.read().strip())
            except (OSError, ValueError):
                pass
        r = subprocess.run(["ping", "-c", "1", "-W", "1", ROBOT_IP], capture_output=True, text=True)
        m = re.search(r"time=([\d.]+) ms", r.stdout)
        if m:
            out["ping_ms"] = float(m.group(1))
        return out

    @staticmethod
    def _stack_running():
        r = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True)
        return "build/franka_interface" in r.stdout

    def event(self, text):
        self.events.append(time.strftime("%H:%M:%S ") + text)

    def _send_virtual_hands(self):
        """While enabled, plays the role of hands.py: a hand counts as near when it touches the PCB plus margin."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        while True:
            time.sleep(0.1)
            if not self.virtual["enabled"]:
                continue
            if self.virtual_points is None:
                self.virtual_points = scene()["points"]
            # The operator holds the PCB: only a hand reaching a soldering point counts, not one on the edges
            near = [any(((h[0] - p[0]) ** 2 + (h[1] - p[1]) ** 2) ** 0.5 < VIRTUAL_POINT_R + VIRTUAL_HAND_R
                        for p in self.virtual_points) for h in self.virtual["hands"]]
            self.virtual["near"] = near
            sock.sendto(json.dumps({"t": time.time(), "near": any(near), "hands": sum(near),
                                    "confidence": 1.0, "virtual": True}).encode("utf-8"), (IP, HANDS_PORT))

    def set_virtual_hands(self, enabled=None, hands=None):
        if enabled is not None:
            if enabled and not self.settings["simulated"]:
                return "virtual hands only work in simulated mode"
            if enabled and self.processes["hands"].running():
                self.processes["hands"].interrupt()     # two senders on 5016 would contradict each other
                self.event("Hands program stopped: the virtual hands replace it")
            self.virtual["enabled"] = bool(enabled)
            self.event("virtual hands " + ("on: drag them in the 3D view" if enabled else "off"))
        if hands is not None:
            self.virtual["hands"] = [[float(h[0]), float(h[1])] for h in hands[:2]]
        return "ok"

    # ---------- commands ----------

    def scale(self):
        return (0.6, 0.6, 0.3) if self.settings["speed"] == "normal" else (0.3, 0.3, 0.15)

    def start_controller(self):
        max_scale = self.scale()[0]
        routine = "--pcb" if self.settings["routine"] == "pcb" else "--cycles 1"
        p = self.processes["controller"]
        if self.settings["simulated"]:
            return p.start(["python3", "robot/cycle_controller.py", "--simulated"] + routine.split()
                           + ["--max-scale", str(max_scale), "--yes"])
        return p.start(docker("exec", "-i", CONTAINER, "bash", "-c",
                              "%s && cd '%s/robot' && python3 -u cycle_controller.py %s --max-scale %s --yes"
                              % (ROS_SETUP, CONTAINER_PATH, routine, max_scale)))

    def start_module(self, name):
        if name == "hands" and self.virtual["enabled"]:
            self.event("Hands program not started: the virtual hands are on")
            return False
        script = {"face": "perception/face.py", "voice": "perception/voice.py", "hands": "perception/hands.py",
                  "screen": "screen/screen.py", "dialogue": "dialogue/dialogue.py"}[name]
        extra = sim_args(name) if self.settings["simulated"] else []
        return self.processes[name].start([PY, script] + extra)

    def start_supervisor(self):
        _, v_normal, v_cautious = self.scale()
        return self.processes["supervisor"].start(["python3", "supervisor/supervisor.py",
                                                   "--v-normal", str(v_normal), "--v-cautious", str(v_cautious)])

    def key(self, k):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(json.dumps({"key": k}).encode("utf-8"), (IP, CONTROL_PORT))
        self.event("key %s (%s)" % (k, KEYS.get(k, k)))

    def _container_exec(self, script):
        """Runs a short command in the container (to stop the programs started with docker exec)."""
        subprocess.run(docker("exec", CONTAINER, "bash", "-c", script), stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=60)

    def full_simulation(self):
        """Everything as in the real test, with the fake arm and the laptop's camera and microphone. The hands are
        the virtual ones dragged in the 3D view: the laptop camera sees too little of the table, and hand detection
        on it gave many false alarms."""
        if self.busy:
            return "busy: " + self.busy
        if any(self.processes[n].running() for n in ["controller", "supervisor"] + MODULES):
            return "something is already running: press Stop all first"
        self.settings["simulated"] = True
        self.settings["modules"] = {m: m != "hands" for m in MODULES}
        self.virtual["enabled"] = True
        self.virtual["hands"] = [[0.78, -0.16], [0.78, 0.16]]
        self.event("full simulation: fake arm + face, voice, screen and speech; drag the virtual hands in the 3D view")
        return self.start_system()

    def start_system(self):
        """Stack -> controller -> supervisor -> perception, screen and dialogue."""
        if self.busy:
            return "busy: " + self.busy
        threading.Thread(target=self._start_system, daemon=True).start()
        return "starting"

    def _start_system(self):
        try:
            if not self.settings["simulated"]:
                self.busy = "starting the Docker container"
                self.event(self.busy)
                r = subprocess.run(docker("start", CONTAINER), capture_output=True, text=True)
                if r.returncode != 0:
                    self.event("[error] docker start: " + (r.stderr.strip() or "failed"))
                    return
                if self._stack_running():
                    self.event("the stack was already running")
                else:
                    self.busy = "starting the frankapy stack"
                    self.event(self.busy)
                    stack = self.processes["stack"]
                    # -t: start_stack.sh uses job control (set -m); it is stopped with pkill, not with stdin
                    stack.start(docker("exec", "-t", CONTAINER, "bash", "%s/robot/start_stack.sh" % CONTAINER_PATH))
                    t0 = time.time()
                    while not any("stack ready" in l for l in stack.tail(50)):
                        if not stack.running() or time.time() - t0 > STACK_TIMEOUT:
                            self.event("[error] the stack did not start: look at its log")
                            return
                        time.sleep(0.5)
                    self.event("stack ready")
            self.busy = "starting the controller and the supervisor"
            self.event(self.busy)
            self.start_controller()
            time.sleep(2.0)
            if not self.processes["controller"].running():
                self.event("[error] the controller stopped: look at its log")
                return
            self.start_supervisor()
            time.sleep(1.0)
            self.busy = "starting perception, screen and dialogue"
            for name in ("face", "voice", "hands", "screen"):
                if self.settings["modules"].get(name):
                    self.start_module(name)
                    time.sleep(0.5)
            if self.settings["modules"].get("dialogue"):
                time.sleep(3.0)         # let the voice measure the background noise before the robot greets
                self.start_module("dialogue")
            self.event("system ready: the robot waits PAUSED. Calibrate (k), then answer the robot \"yes\" "
                       "or press Start routine.")
        finally:
            self.busy = ""

    def stop_controller(self):
        p = self.processes["controller"]
        if self.settings["simulated"] or not p.running():
            p.interrupt()
        else:
            self._container_exec("pkill -INT -f '[c]ycle_controller'")
        if not p.wait(5.0):
            p.kill()

    def restart_routine(self):
        """Pauses, starts the controller again (it goes back to home first) and the dialogue."""
        if self.busy:
            return "busy: " + self.busy

        def run():
            self.busy = "restarting the routine"
            try:
                self.key("p")
                self.stop_controller()
                time.sleep(1.0)
                self.start_controller()
                if self.processes["dialogue"].running():
                    self.processes["dialogue"].interrupt()
                    self.processes["dialogue"].wait(3.0)
                    self.start_module("dialogue")
                self.event("routine restarted: PAUSED at the start. Press Start routine.")
            finally:
                self.busy = ""
        threading.Thread(target=run, daemon=True).start()
        return "restarting"

    def stop_all(self, then_exit=False):
        def run():
            self.busy = "stopping everything"
            self.event(self.busy)
            try:
                # The supervisor first: on closing it sends PAUSED, and without it the controller stops in 0.5 s
                sup = self.processes["supervisor"]
                sup.interrupt()
                sup.wait(3.0)
                for name in MODULES + ["pcb_plan"]:
                    self.processes[name].interrupt()
                for name in MODULES:
                    if not self.processes[name].wait(3.0):
                        self.processes[name].kill()
                ctrl = self.processes["controller"]
                if self.settings["simulated"] or not (ctrl.running() or self._stack_running()):
                    ctrl.interrupt()
                    ctrl.wait(3.0)
                else:
                    self._container_exec("pkill -INT -f '[c]ycle_controller'; pkill -INT -f '[c]ommunication_test'; "
                                         "sleep 2; pkill -INT -f '[s]tart_stack.sh'; sleep 10; "
                                         "pkill -INT -f '[f]ranka_gripper.launch'; sleep 2; "
                                         "pkill -INT -f '[f]ranka_ros_interface.launch'; sleep 2; "
                                         "pkill -INT -f 'build/[f]ranka_interface'; sleep 2; pkill -INT -x roscore")
                for p in self.processes.values():
                    if not p.wait(3.0):
                        p.kill()
                self.event("everything stopped. In Desk: deactivate FCI and lock the joints.")
            finally:
                self.busy = ""
                if then_exit:
                    time.sleep(0.5)     # let the page get its answer first
                    os._exit(0)
        threading.Thread(target=run, daemon=True).start()
        return "stopping"

    def comm_test(self):
        if self.settings["simulated"]:
            return "only with the real robot"
        if not self._stack_running():
            return "start the system first (the stack must be running)"
        if self.processes["controller"].running():
            return "stop the controller first: only one program can drive the robot"
        p = self.processes["comm_test"]
        p.start(docker("exec", "-i", CONTAINER, "bash", "-c", "%s && cd '%s/robot' && python3 -u communication_test.py 1"
                       % (ROS_SETUP, CONTAINER_PATH)), stdin=True)

        def press_enter():
            t0 = time.time()
            while p.running() and time.time() - t0 < 30:
                if any("Enter to run it" in l for l in p.tail(5)):
                    p.write("\n")
                    return
                time.sleep(0.3)
        threading.Thread(target=press_enter, daemon=True).start()
        return "running: the wrist turns 20 degrees and back"

    def pcb_plan(self):
        self.processes["pcb_plan"].start(["python3", "robot/pcb_plan.py"])
        return "checking"

    # ---------- state for the page ----------

    def status(self):
        now = time.time()
        summary = self.summary if self.summary and now - self.t_summary <= 1.5 else None
        return {
            "processes": {n: {"title": TITLES[n], "running": p.running(), "code": p.exit_code(), "last": p.last()}
                          for n, p in self.processes.items()},
            "settings": self.settings, "summary": summary, "network": self.network, "docker": self.docker,
            "busy": self.busy, "events": list(self.events)[-12:], "has_venv": os.path.exists(PY),
            "virtual": self.virtual, "sim_devices": sim_devices() if self.settings["simulated"] else None,
        }


def scene():
    """What the 3D view draws besides the arm: table markers, PCB corners and soldering points (robot frame, m)."""
    out = {"dh": pcb_plan.DH, "flange": 0.107, "tcp": pcb_plan.TCP_FRANKAPY, "pen": 0.0,
           "arucos": [], "table_z": 0.0, "corners": [], "points": []}
    try:
        with open(os.path.join(pcb_plan.DATA_DIR, "table.json")) as f:
            table = json.load(f)
        out["arucos"] = [table["arucos"][k] for k in table.get("contour_order", sorted(table["arucos"]))]
        out["table_z"], out["aruco_side"] = table.get("table_z", 0.0), table.get("aruco_side_m", 0.048)
        _, info = pcb_plan.load()
        E = [info["corner_%d" % i]["tip_xyz"] for i in range(1, 5)]
        out["corners"], out["pen"] = E, info["corner_1"]["pen_length_cm"] / 100.0
        out["points"] = pcb_plan.grid_points(E, 2, 3, pcb_plan.MARGIN)
        out["point_r"] = VIRTUAL_POINT_R
        out["hand_r"] = VIRTUAL_HAND_R
    except (IOError, KeyError, ValueError) as e:
        out["error"] = str(e)
    return out


def make_handler(panel, token):
    page = open(os.path.join(PANEL_DIR, "index.html"), encoding="utf-8").read().replace("{{TOKEN}}", token)
    icon = open(os.path.join(PANEL_DIR, "icon.svg"), encoding="utf-8").read()
    allowed_hosts = {"127.0.0.1:%d" % PORT, "localhost:%d" % PORT}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, kind="application/json"):
            data = body.encode("utf-8") if isinstance(body, str) else json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", kind + "; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _allowed(self):
            # Host check stops DNS rebinding; the token stops other pages from sending actions
            return self.headers.get("Host") in allowed_hosts

        def do_GET(self):
            if not self._allowed():
                return self._send(403, {"error": "forbidden"})
            url = urlparse(self.path)
            if url.path == "/":
                return self._send(200, page, "text/html")
            if url.path == "/icon.svg":
                return self._send(200, icon, "image/svg+xml")
            if self.headers.get("X-Token") != token:
                return self._send(403, {"error": "bad token"})
            if url.path == "/api/status":
                return self._send(200, panel.status())
            if url.path == "/api/scene":
                return self._send(200, scene())
            if url.path == "/api/live":           # small and frequent: the 3D view polls it at 10 Hz
                fresh = panel.summary if panel.summary and time.time() - panel.t_summary <= 1.5 else None
                return self._send(200, {"summary": fresh, "virtual": panel.virtual})
            if url.path == "/api/log":
                name = parse_qs(url.query).get("name", [""])[0]
                if name == "events":
                    return self._send(200, {"lines": list(panel.events)})
                if name in panel.processes:
                    return self._send(200, {"lines": panel.processes[name].tail()})
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._allowed() or self.headers.get("X-Token") != token:
                return self._send(403, {"error": "forbidden"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except ValueError:
                return self._send(400, {"error": "bad json"})
            action = body.get("action")
            result = "ok"
            if action == "settings":
                for k in ("simulated", "routine", "speed"):
                    if k in body:
                        panel.settings[k] = body[k]
                if not panel.settings["simulated"]:
                    panel.virtual["enabled"] = False    # never fake hands next to the real robot
                for k, v in (body.get("modules") or {}).items():
                    if k in MODULES:
                        panel.settings["modules"][k] = bool(v)
            elif action == "full_simulation":
                result = panel.full_simulation()
            elif action == "start_system":
                result = panel.start_system()
            elif action == "stop_all":
                result = panel.stop_all()
            elif action == "quit":
                result = panel.stop_all(then_exit=True)
            elif action == "key" and body.get("key") in KEYS:
                if not panel.processes["supervisor"].running():
                    result = "the supervisor is not running: start the system first"
                else:
                    panel.key(body["key"])
            elif action == "restart_routine":
                result = panel.restart_routine()
            elif action == "module" and body.get("name") in MODULES:
                name = body["name"]
                if body.get("on"):
                    panel.start_module(name)
                else:
                    panel.processes[name].interrupt()
            elif action == "virtual_hands":
                result = panel.set_virtual_hands(body.get("enabled"), body.get("hands"))
            elif action == "comm_test":
                result = panel.comm_test()
            elif action == "pcb_plan":
                result = panel.pcb_plan()
            elif action == "open_desk":
                browser = shutil.which("google-chrome") or shutil.which("xdg-open")
                subprocess.Popen([browser, "http://%s" % ROBOT_IP], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True)
            else:
                return self._send(400, {"error": "unknown action"})
            return self._send(200, {"result": result})

    return Handler


def main():
    global PORT
    p = argparse.ArgumentParser(description="Control panel for the soldering robot")
    p.add_argument("--port", type=int, default=PORT)
    args = p.parse_args()
    PORT = args.port
    signal.signal(signal.SIGINT, signal.default_int_handler)     # Ctrl+C works even when started in the background
    token = secrets.token_urlsafe(24)
    panel = Panel()
    try:
        server = ThreadingHTTPServer((IP, PORT), make_handler(panel, token))
    except OSError as e:
        sys.exit("[error] cannot listen on %s:%d (%s): is the panel already open?" % (IP, PORT, e))
    print("Control panel on http://%s:%d (Ctrl+C stops everything and closes it)" % (IP, PORT), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("closing: stopping everything...", flush=True)
        panel.stop_all(then_exit=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
