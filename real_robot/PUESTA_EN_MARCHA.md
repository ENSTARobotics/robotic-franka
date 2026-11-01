# Puesta en marcha con el Panda del laboratorio (U2IS)

Basado en el *Franka Panda Tutorial - U2IS* (Caio Conti, oct. 2026) y en lo que encontramos
al usarlo. Las **credenciales de Desk no están aquí** (el repo es público): están en ese
tutorial o pídelas al responsable del laboratorio.

## 1. Red

Cable Ethernet **directo** del ordenador a la **caja de control** del robot (no a la base del brazo).
IP estática en esa interfaz:

| Campo | Valor |
|---|---|
| Dirección | `147.250.35.100` |
| Máscara | `255.255.255.0` |
| Puerta de enlace | ninguna (así el WiFi sigue dando internet). En Ubuntu: "Use this connection only for resources on its network" |

Comprobar:

```bash
ping 147.250.35.11
cat /sys/class/net/<interfaz>/speed      # tiene que dar 1000
```

**Importante:** el enlace tiene que ser de **1000 Mb/s** y el ping estable por debajo de 1 ms.
Con una tarjeta que negociaba 100 Mb/s (con picos de ping >1 ms) el robot abortaba todos los
movimientos con `communication_constraints_violation`; con un adaptador USB-Ethernet gigabit
funciona. Para ver por qué interfaz sale el tráfico: `ip route get 147.250.35.11`.

## 2. Desk (interfaz web del robot)

1. Abrir http://147.250.35.11 en **Chrome** (Firefox no funcionó).
2. El robot arranca bloqueado (luz amarilla): desbloquear las articulaciones en la sección
   *Joints* -> luz **azul**.
3. **Activar la FCI** (necesaria para controlarlo desde el ordenador).

Al terminar: desactivar la FCI y bloquear las articulaciones.

## 3. Código y Docker con frankapy (ROS 1)

Baja el código en tu carpeta personal. Hasta que se acepten los PR, está en este fork:

```bash
sudo apt install git docker.io
git clone -b feat/percepcion https://github.com/JairVasquezT/robotic-franka.git ~/robotic-franka
```

El Panda del laboratorio es un **FER**: no funciona con `franka_ros2`. Se usa la imagen del
laboratorio con frankapy:

```bash
sudo docker pull jgarcia10/u2is-franka-ros1:latest
```

Crear el contenedor **compartiendo tu copia de este repo** (cambia la ruta por la tuya). Sin
`-v` el contenedor no ve los scripts de `real_robot/`:

```bash
sudo docker run -it --name franka_container \
  -v ~/robotic-franka:/root/robotic-franka \
  --net=host --ipc=host --ulimit rtprio=99 --ulimit memlock=-1 --cap-add=sys_nice \
  --security-opt seccomp=unconfined jgarcia10/u2is-franka-ros1:latest
```

`--net=host` es obligatorio: el puente UDP usa `127.0.0.1` entre el contenedor y el host.
Si ya tenías un `franka_container` creado sin `-v`, hay que crearlo de nuevo
(`sudo docker rm franka_container` y repetir el `run`).

Las siguientes veces:

```bash
sudo docker start franka_container
sudo docker exec -it franka_container bash      # una vez por cada terminal
```

## 4. Programas del host (cara, voz, pantalla y diálogo)

Sólo probado en Ubuntu 24.04 con Python 3.12. El supervisor no necesita nada de esto y el
controlador corre en el contenedor. Sin GPU NVIDIA, lanza `rostro.py` y `voz.py` con `--cpu`.

```bash
sudo apt install python3.12-venv libportaudio2
cd ~/robotic-franka/real_robot
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python dialogo/descargar_modelos.py
.venv/bin/python -c "import cv2, torch, librosa, sounddevice, PIL, piper, vosk; print('CUDA:', torch.cuda.is_available())"
```

## 5. Arrancar el stack (dentro del contenedor)

En **cada** terminal del contenedor:

```bash
cd /root/frankapy/catkin_ws && source devel/setup.bash
```

Una terminal para cada uno, **en este orden**:

```bash
roscore
~/franka-interface/build/franka_interface --robot_ip 147.250.35.11
roslaunch franka_ros_interface franka_ros_interface.launch robot_ip:=147.250.35.11
roslaunch franka_gripper franka_gripper.launch robot_ip:=147.250.35.11 __ns:=/franka_gripper_1
```

Apagado: Ctrl+C en orden inverso (garra -> ros_interface -> franka_interface -> roscore).

Mensajes normales: `autolab_core not installed as catkin package`,
`Will try to acquire lock while setting franka_interface status`.

## 6. Probar

En una quinta terminal del contenedor, con la mano en el paro de emergencia:

```bash
cd /root/robotic-franka/real_robot/robot
python3 prueba_comunicacion.py 1      # gira la muñeca 20° y vuelve
```

Si funciona, seguir con el `README.md` de esta carpeta (supervisor + controlador).

Las poses y las esquinas de `datos/` son de nuestra mesa. Si la mesa o la placa se movieron, o si
el bolígrafo cambió, vuelve a grabarlas (`robot/grabar_poses.py`) y revisa el plan
(`python3 robot/plan_placa.py`).

Notas:
- `fa.get_gripper_width()` falla en este setup (también en `move_robot.py` del tutorial: hay
  que comentar esa parte).
- frankapy tiene paredes virtuales: si un movimiento se rechaza con
  `Target joints in collision with virtual walls!`, el destino está fuera de su espacio
  permitido (no es un fallo de comunicación).
- Si el robot no se mueve, el modo guía queda rígido o salen `done_skill_id error` /
  `Can't parse message of type "RobotStateMessage"`, mira la terminal de `franka_interface`:
  si dice `communication_constraints_violation`, revisa la red (sección 1).
