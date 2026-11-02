#!/bin/bash
# Arranca el stack de frankapy en una sola terminal, en orden. Se ejecuta DENTRO del contenedor.
# No mueve el robot. Ctrl+C lo apaga en orden inverso (garra, ros_interface, franka_interface, roscore).
# Cada línea va con su etiqueta y recortada: el error de RobotStateMessage ocupa pantallas enteras.
IP_ROBOT=147.250.35.11
set -m                                  # cada proceso, en su grupo: Ctrl+C sólo llega a este script
source /root/frankapy/catkin_ws/devel/setup.bash

grupos=()
arrancar() {                            # arrancar <etiqueta> <segundos de espera> <comando...>
    local etiqueta=$1 espera=$2
    shift 2
    echo "=== arrancando $etiqueta ==="
    # El filtro ignora Ctrl+C para seguir leyendo mientras el proceso se apaga: si muriera antes, ese
    # proceso se quedaría sin salida a mitad de su cierre
    "$@" 2>&1 | (trap '' INT; exec sed -u "s/^\(.\{0,300\}\).*/[$etiqueta] \1/") &
    grupos+=("$(jobs -p %%)")
    sleep "$espera"
}

apagar() {
    echo "=== apagando en orden inverso ==="
    for ((i = ${#grupos[@]} - 1; i >= 0; i--)); do
        kill -INT -- "-${grupos[i]}" 2>/dev/null
        sleep 2
    done
    wait 2>/dev/null
    echo "=== stack apagado. En Desk: desactiva la FCI y bloquea las articulaciones ==="
    exit 0
}
trap apagar INT TERM

arrancar roscore 5 roscore
arrancar franka 4 ~/franka-interface/build/franka_interface --robot_ip "$IP_ROBOT"
arrancar ros_if 5 roslaunch franka_ros_interface franka_ros_interface.launch robot_ip:="$IP_ROBOT"
arrancar garra 3 roslaunch franka_gripper franka_gripper.launch robot_ip:="$IP_ROBOT" __ns:=/franka_gripper_1
echo "=== stack listo. Ctrl+C aquí para apagarlo ==="
wait
