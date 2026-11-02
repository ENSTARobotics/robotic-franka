#!/bin/bash
# Starts the frankapy stack in a single terminal, in order. Runs INSIDE the container.
# Does not move the robot. Ctrl+C shuts it down in reverse order (gripper, ros_interface, franka_interface, roscore).
# Each line gets its label and is cut short: the RobotStateMessage error fills whole screens.
ROBOT_IP=147.250.35.11
set -m -o pipefail                      # each process in its own group: Ctrl+C only reaches this script
source /root/frankapy/catkin_ws/devel/setup.bash

groups=()
start() {                               # start <label> <seconds to wait> <command...>
    local label=$1 wait_s=$2
    shift 2
    echo "=== starting $label ==="
    # The filter ignores Ctrl+C to keep reading while the process shuts down: if it died first, that
    # process would lose its output halfway through closing
    "$@" 2>&1 | (trap '' INT; exec sed -u "s/^\(.\{0,300\}\).*/[$label] \1/") &
    groups+=("$(jobs -p %%)")
    sleep "$wait_s"
}

shut_down() {
    echo "=== shutting down in reverse order ==="
    for ((i = ${#groups[@]} - 1; i >= 0; i--)); do
        kill -INT -- "-${groups[i]}" 2>/dev/null
        sleep 2
    done
    wait 2>/dev/null
    echo "=== stack shut down. In Desk: deactivate FCI and lock the joints ==="
    exit 0
}
trap shut_down INT TERM

# franka_interface sets the impedances before its own error handling: if the robot is still in Reflex (e.g. after
# a communication_constraints_violation), it dies right away. Clear the error first; it does not move the robot.
echo "=== clearing robot errors ==="
out=$(bash "$(dirname "$(realpath "$0")")/recover_errors.sh" 2>&1); ok=$?
echo "$out" | sed "s/^/[recover] /"
if [ $ok != 0 ]; then
    if echo "$out" | grep -q "manual error recovery"; then
        # e.g. a collision (joint_reflex, cartesian_reflex): only Desk or the stop button can clear it
        echo "[error] the robot needs a manual error recovery: in Desk, acknowledge the error (or press and release the stop button), unlock the joints if asked (blue light), then Start system again"
    else
        echo "[error] cannot talk to the robot through FCI: in Desk, unlock the joints and activate FCI (blue light)"
    fi
    exit 1
fi
start roscore 5 roscore
start franka 4 ~/franka-interface/build/franka_interface --robot_ip "$ROBOT_IP"
start ros_if 5 roslaunch franka_ros_interface franka_ros_interface.launch robot_ip:="$ROBOT_IP"
start gripper 3 roslaunch franka_gripper franka_gripper.launch robot_ip:="$ROBOT_IP" __ns:=/franka_gripper_1
echo "=== stack ready. Ctrl+C here to shut it down ==="
wait
