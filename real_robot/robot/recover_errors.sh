#!/bin/bash
# Clears a robot error through FCI (libfranka's automaticErrorRecovery), e.g. the "Reflex" mode that makes
# franka_interface refuse every command. Runs INSIDE the container, with the stack stopped (FCI takes only
# one program). Does not move the robot. Builds a tiny program the first time.
ROBOT_IP=147.250.35.11
BIN=/root/.cache/aided_building/recover_errors_v2   # new name when the program below changes

if [ ! -x "$BIN" ]; then
    header=$(find /root /opt /usr/local -path '*include/franka/robot.h' 2>/dev/null | head -1)
    library=$(find /root /opt /usr/local -name 'libfranka.so*' 2>/dev/null | head -1)
    [ -n "$header" ] && [ -n "$library" ] || { echo "[error] libfranka not found in the container"; exit 1; }
    echo "Building the recovery program with $library ..."
    mkdir -p "$(dirname "$BIN")"
    g++ -std=c++17 -o "$BIN" -I"${header%/franka/robot.h}" -x c++ - -x none "$library" -Wl,-rpath,"$(dirname "$library")" <<'EOF' || exit 1
#include <iostream>
#include <franka/exception.h>
#include <franka/robot.h>
const char* MODES[] = {"other", "idle (ready)", "move", "guiding", "reflex (error)", "user stopped",
                       "automatic error recovery"};
const char* mode(franka::Robot& robot) { return MODES[static_cast<int>(robot.readOnce().robot_mode)]; }
int main(int argc, char** argv) {
  try {
    franka::Robot robot(argv[1]);
    franka::RobotState state = robot.readOnce();
    // Why the last motion stopped (e.g. cartesian_reflex = it pushed against something)
    std::cout << "Errors now: " << std::string(state.current_errors) << std::endl;
    std::cout << "Errors of the last motion: " << std::string(state.last_motion_errors) << std::endl;
    std::cout << "Robot mode before: " << mode(robot) << std::endl;
    robot.automaticErrorRecovery();
    std::cout << "Robot mode after: " << mode(robot) << std::endl;
  } catch (const franka::Exception& e) {
    std::cout << "[error] " << e.what() << std::endl;
    return 1;
  }
  std::cout << "Error cleared." << std::endl;
  return 0;
}
EOF
fi
"$BIN" "$ROBOT_IP"
