#!/bin/bash
# Opens everything needed in tabs. Runs on the host, in the Ubuntu Terminal (not in the VS Code one).
#   ./launch.sh                 # real robot, slow speed, routine over the PCB
#   ./launch.sh --normal        # normal speed (0.6)
#   ./launch.sh --cycle         # cycle between poses instead of the PCB
#   ./launch.sh --simulated     # no robot and no Docker; the C920 webcam, or the laptop's camera and mic
#   ./launch.sh --dry-run       # shows the commands without opening anything
#   ./launch.sh --restart       # closes everything, robot included, and starts from scratch (combinable)
#   ./launch.sh --stop          # only closes everything
# The robot does not move until you press Enter in the controller and give the go with "yes" or the a key.
DIR=$(dirname "$(realpath "$0")")
CONTAINER=franka_container
# With -v ~/folder:/root/folder, the path in the container is the same with /root instead of $HOME
CONTAINER_PATH=/root/${DIR#"$HOME"/}

SCALE=0.3; V_NORMAL=0.3; V_CAUTIOUS=0.15; ROUTINE=--pcb; SIMULATED=0; DRY_RUN=0; STOP=0; GO_ON=1
for arg in "$@"; do
    case $arg in
        --normal) SCALE=0.6; V_NORMAL=0.6; V_CAUTIOUS=0.3 ;;
        --cycle) ROUTINE="--cycles 1" ;;
        --simulated) SIMULATED=1 ;;
        --dry-run) DRY_RUN=1 ;;
        --restart) STOP=1 ;;
        --stop) STOP=1; GO_ON=0 ;;
        *) sed -n '2,9p' "$0"; exit 1 ;;
    esac
done

# The VS Code terminal (installed as a snap) carries variables that keep gnome-terminal from opening
if [ "$DRY_RUN" = 0 ] && [ "$GO_ON" = 1 ] && { [ "$TERM_PROGRAM" = vscode ] || [ -n "$VSCODE_PID" ] || [ -n "$SNAP" ]; }; then
    echo "[error] Run it from the Ubuntu Terminal (Ctrl+Alt+T): the tabs do not open from VS Code."
    echo "        To see the commands and open them by hand: ./launch.sh --dry-run"
    exit 1
fi

PY=$DIR/.venv/bin/python
[ -x "$PY" ] || PY=$HOME/aided_building/.venv/bin/python
[ -x "$PY" ] || { echo "[error] .venv not found: see SETUP.md"; exit 1; }
[ "$DRY_RUN" = 1 ] || command -v gnome-terminal >/dev/null || { echo "[error] gnome-terminal is needed"; exit 1; }

stop_all() {
    # The supervisor first: when it closes it sends PAUSED, and without it the controller stops the robot in 0.5 s
    if [ "$DRY_RUN" = 1 ]; then
        echo "(would close supervisor, controller, perception, screen, dialogue and the stack, in that order)"
        return
    fi
    echo "Closing everything..."
    pkill -INT -f 'supervisor/([s]upervisor|[k]eyboard_supervisor)\.py'
    sleep 1
    pkill -INT -f 'dialogue/[d]ialogue\.py|screen/[s]creen\.py|perception/([f]ace|[v]oice|[p]cb|[h]ands)\.py|robot/[c]ycle_controller\.py'
    if ps -eo args | grep -qE '[b]uild/franka_interface|[r]oscore|python3 [c]ycle_controller'; then
        echo "Your password is needed to close the robot programs:"
        sudo docker exec "$CONTAINER" bash -c "pkill -INT -f '[c]ycle_controller'; sleep 2
            pkill -INT -f '[f]ranka_gripper.launch'; sleep 2
            pkill -INT -f '[f]ranka_ros_interface.launch'; sleep 2
            pkill -INT -f 'build/[f]ranka_interface'; sleep 2
            pkill -INT -x roscore; sleep 2"
        if ps -eo args | grep -qE '[b]uild/franka_interface|bin/[r]oscore'; then
            echo "Some are still open: restarting the container."
            sudo docker restart "$CONTAINER" >/dev/null
        fi
    fi
    echo "Everything closed. You can close the old tabs: they no longer run anything."
}
[ "$STOP" = 1 ] && stop_all
[ "$GO_ON" = 1 ] || exit 0

first=1
tab() {                                 # tab <title> <command>; stays open so errors can be read
    if [ "$DRY_RUN" = 1 ]; then
        printf '[%s]\n  %s\n' "$1" "$2"
        return
    fi
    local where=--tab
    [ "$first" = 1 ] && where=--window && first=0
    gnome-terminal "$where" --title="$1" -- bash -c "$2; echo; echo '[$1] finished'; exec bash"
    sleep 0.7
}
wait_for() {
    [ "$DRY_RUN" = 1 ] && { echo "(pause: $1)"; return; }
    read -r -p "$1 " _
}

if [ "$SIMULATED" = 1 ]; then
    tab Controller "python3 '$DIR/robot/cycle_controller.py' --simulated $ROUTINE --max-scale $SCALE"
else
    if [ "$DRY_RUN" = 0 ]; then
        interface=$(ip route get 147.250.35.11 2>/dev/null | grep -oP 'dev \K\S+')
        speed=$(cat "/sys/class/net/$interface/speed" 2>/dev/null)
        [ "$speed" = 1000 ] || echo "[warning] link to the robot: ${speed:-not connected} Mb/s; it must be 1000"
    fi
    echo "Before going on: in Desk, joints unlocked (blue light) and FCI active."
    wait_for "Emergency stop at hand and hands out of the arm's reach. Enter to open the stack..."
    if ps -eo args | grep -q '[b]uild/franka_interface'; then
        echo "The stack is already running: not opening it again."
    else
        tab Stack "sudo docker start $CONTAINER >/dev/null && sudo docker exec -it $CONTAINER bash '$CONTAINER_PATH/robot/start_stack.sh'"
        echo "Type your password in the Stack tab and wait until it says 'stack ready'."
        wait_for "Enter when it says so..."
    fi
    tab Controller "sudo docker exec -it $CONTAINER bash -c 'source /root/frankapy/catkin_ws/devel/setup.bash && cd \"$CONTAINER_PATH/robot\" && python3 cycle_controller.py $ROUTINE --max-scale $SCALE'"
    echo "In the Controller tab: password, check what it prints and press Enter."
    wait_for "Enter when it is waiting for the supervisor..."
fi

# Simulated: the laptop's camera and microphone. One camera can only be opened by one program, so face.py
# lends its images to hands.py, which watches the whole image (there are no table markers)
FACE=; VOICE=; HANDS=; DIALOGUE=
if [ "$SIMULATED" = 1 ]; then
    CAM=0; MIC=system                   # the C920 webcam if it is plugged in, else the laptop's
    grep -qs C920 /sys/class/video4linux/video*/name && CAM=C920
    pactl list short sources 2>/dev/null | grep -q C920 && MIC=C920
    FACE="--camera $CAM --share-frames"; VOICE="--device $MIC"; HANDS="--frames-from-face --no-markers"
    DIALOGUE="--device $MIC"
fi
tab Supervisor "python3 '$DIR/supervisor/supervisor.py' --v-normal $V_NORMAL --v-cautious $V_CAUTIOUS"
tab Face "'$PY' '$DIR/perception/face.py' $FACE"
tab Voice "'$PY' '$DIR/perception/voice.py' $VOICE"
tab Screen "'$PY' '$DIR/screen/screen.py'"
tab Hands "'$PY' '$DIR/perception/hands.py' $HANDS"    # hands next to the PCB + the PCB check
sleep 3                                 # let the voice measure the background noise before the robot greets
tab Dialogue "'$PY' '$DIR/dialogue/dialogue.py' $DIALOGUE"

echo
echo "Ready. Stay silent for a few seconds."
echo "To calibrate this person: k in the Supervisor tab, before starting."
echo "Answer the robot: 'yes' to start."
echo "To stop: p in the Supervisor tab, or the emergency stop."
echo "To shut down: q in Supervisor and Ctrl+C in Stack; then, in Desk, deactivate FCI and lock the joints."
