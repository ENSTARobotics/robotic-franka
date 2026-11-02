#!/bin/bash
# Abre en pestañas todo lo necesario. Se ejecuta en el host, en la Terminal de Ubuntu (no en la de VS Code).
#   ./lanzar.sh                 # robot real, velocidad lenta, rutina sobre la placa
#   ./lanzar.sh --normal        # velocidad normal (0.6)
#   ./lanzar.sh --ciclo         # ciclo entre poses en vez de la placa
#   ./lanzar.sh --simulado      # sin robot ni Docker
#   ./lanzar.sh --solo-mostrar  # enseña los comandos sin abrir nada
#   ./lanzar.sh --reiniciar     # cierra todo, robot incluido, y arranca desde cero (combinable)
#   ./lanzar.sh --parar         # sólo cierra todo
# El robot no se mueve hasta pulsar Enter en el controlador y dar paso con "yes" o con la tecla a.
DIR=$(dirname "$(realpath "$0")")
CONTENEDOR=franka_container
# Con -v ~/carpeta:/root/carpeta, la ruta del contenedor es la misma con /root en vez de $HOME
RUTA_CONTENEDOR=/root/${DIR#"$HOME"/}

ESCALA=0.3; V_NORMAL=0.3; V_CAUTIOUS=0.15; RUTINA=--placa; SIMULADO=0; MOSTRAR=0; PARAR=0; SEGUIR=1
for arg in "$@"; do
    case $arg in
        --normal) ESCALA=0.6; V_NORMAL=0.6; V_CAUTIOUS=0.3 ;;
        --ciclo) RUTINA="--ciclos 1" ;;
        --simulado) SIMULADO=1 ;;
        --solo-mostrar) MOSTRAR=1 ;;
        --reiniciar) PARAR=1 ;;
        --parar) PARAR=1; SEGUIR=0 ;;
        *) sed -n '2,9p' "$0"; exit 1 ;;
    esac
done

# La terminal de VS Code (instalado como snap) arrastra variables que impiden abrir gnome-terminal
if [ "$MOSTRAR" = 0 ] && [ "$SEGUIR" = 1 ] && { [ "$TERM_PROGRAM" = vscode ] || [ -n "$VSCODE_PID" ] || [ -n "$SNAP" ]; }; then
    echo "[error] Lánzalo desde la Terminal de Ubuntu (Ctrl+Alt+T): desde VS Code no se abren las pestañas."
    echo "        Para ver los comandos y abrirlos a mano: ./lanzar.sh --solo-mostrar"
    exit 1
fi

PY=$DIR/.venv/bin/python
[ -x "$PY" ] || PY=$HOME/aided_building/.venv/bin/python
[ -x "$PY" ] || { echo "[error] no encuentro .venv: mira SETUP.md"; exit 1; }
[ "$MOSTRAR" = 1 ] || command -v gnome-terminal >/dev/null || { echo "[error] hace falta gnome-terminal"; exit 1; }

parar_todo() {
    # El supervisor primero: al cerrarse manda PAUSED, y sin él el controlador para el robot en 0.5 s
    if [ "$MOSTRAR" = 1 ]; then
        echo "(cerraría supervisor, controlador, percepción, pantalla, diálogo y el stack, en ese orden)"
        return
    fi
    echo "Cerrando todo..."
    pkill -INT -f 'supervisor/[s]upervisor(_teclado)?\.py'
    sleep 1
    pkill -INT -f 'dialogo/[d]ialogo\.py|pantalla/[p]antalla\.py|percepcion/([r]ostro|[v]oz|[p]laca|[m]anos)\.py|robot/[c]ontrolador_ciclo\.py'
    if ps -eo args | grep -qE '[b]uild/franka_interface|[r]oscore|python3 [c]ontrolador_ciclo'; then
        echo "Para cerrar los programas del robot hace falta tu contraseña:"
        sudo docker exec "$CONTENEDOR" bash -c "pkill -INT -f '[c]ontrolador_ciclo'; sleep 2
            pkill -INT -f '[f]ranka_gripper.launch'; sleep 2
            pkill -INT -f '[f]ranka_ros_interface.launch'; sleep 2
            pkill -INT -f 'build/[f]ranka_interface'; sleep 2
            pkill -INT -x roscore; sleep 2"
        if ps -eo args | grep -qE '[b]uild/franka_interface|bin/[r]oscore'; then
            echo "Algunos siguen abiertos: reinicio el contenedor."
            sudo docker restart "$CONTENEDOR" >/dev/null
        fi
    fi
    echo "Todo cerrado. Las pestañas antiguas puedes cerrarlas: ya no ejecutan nada."
}
[ "$PARAR" = 1 ] && parar_todo
[ "$SEGUIR" = 1 ] || exit 0

primera=1
pestana() {                             # pestana <título> <comando>; queda abierta para leer errores
    if [ "$MOSTRAR" = 1 ]; then
        printf '[%s]\n  %s\n' "$1" "$2"
        return
    fi
    local donde=--tab
    [ "$primera" = 1 ] && donde=--window && primera=0
    gnome-terminal "$donde" --title="$1" -- bash -c "$2; echo; echo '[$1] terminó'; exec bash"
    sleep 0.7
}
esperar() {
    [ "$MOSTRAR" = 1 ] && { echo "(pausa: $1)"; return; }
    read -r -p "$1 " _
}

if [ "$SIMULADO" = 1 ]; then
    pestana Controlador "python3 '$DIR/robot/controlador_ciclo.py' --simulado $RUTINA --escala-max $ESCALA"
else
    if [ "$MOSTRAR" = 0 ]; then
        interfaz=$(ip route get 147.250.35.11 2>/dev/null | grep -oP 'dev \K\S+')
        velocidad=$(cat "/sys/class/net/$interfaz/speed" 2>/dev/null)
        [ "$velocidad" = 1000 ] || echo "[aviso] enlace con el robot: ${velocidad:-sin conexión} Mb/s; tiene que ser 1000"
    fi
    echo "Antes de seguir: en Desk, articulaciones desbloqueadas (luz azul) y FCI activa."
    esperar "Paro de emergencia a mano y manos fuera del alcance del brazo. Enter para abrir el stack..."
    if ps -eo args | grep -q '[b]uild/franka_interface'; then
        echo "El stack ya está corriendo: no lo abro otra vez."
    else
        pestana Stack "sudo docker start $CONTENEDOR >/dev/null && sudo docker exec -it $CONTENEDOR bash '$RUTA_CONTENEDOR/robot/arrancar_stack.sh'"
        echo "Escribe la contraseña en la pestaña Stack y espera a que diga 'stack listo'."
        esperar "Enter cuando lo diga..."
    fi
    pestana Controlador "sudo docker exec -it $CONTENEDOR bash -c 'source /root/frankapy/catkin_ws/devel/setup.bash && cd \"$RUTA_CONTENEDOR/robot\" && python3 controlador_ciclo.py $RUTINA --escala-max $ESCALA'"
    echo "En la pestaña Controlador: contraseña, revisa lo que imprime y pulsa Enter."
    esperar "Enter cuando esté esperando al supervisor..."
fi

pestana Supervisor "python3 '$DIR/supervisor/supervisor.py' --v-normal $V_NORMAL --v-cautious $V_CAUTIOUS"
pestana Cara "'$PY' '$DIR/percepcion/rostro.py'"
pestana Voz "'$PY' '$DIR/percepcion/voz.py'"
pestana Pantalla "'$PY' '$DIR/pantalla/pantalla.py'"
pestana Manos "'$PY' '$DIR/percepcion/manos.py'"      # manos junto a la placa + comprobación de la placa
sleep 3                                 # que la voz mida el ruido de fondo antes de que el robot salude
pestana Dialogo "'$PY' '$DIR/dialogo/dialogo.py'"

echo
echo "Listo. Quédate en silencio unos segundos."
echo "Para calibrar a esta persona: k en la pestaña Supervisor, antes de empezar."
echo "Contesta al robot: 'yes' para empezar."
echo "Para parar: p en la pestaña Supervisor, o el paro de emergencia."
echo "Para apagar: q en Supervisor y Ctrl+C en Stack; luego, en Desk, desactiva la FCI y bloquea las articulaciones."
