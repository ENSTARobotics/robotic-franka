#!/bin/bash
# Abre en pestañas todo lo necesario. Se ejecuta en el host, en un terminal de Ubuntu.
#   ./lanzar.sh                 # robot real, velocidad lenta, rutina sobre la placa
#   ./lanzar.sh --normal        # velocidad normal (0.6)
#   ./lanzar.sh --ciclo         # ciclo entre poses en vez de la placa
#   ./lanzar.sh --simulado      # sin robot ni Docker
#   ./lanzar.sh --solo-mostrar  # enseña los comandos sin abrir nada
# El robot no se mueve hasta pulsar Enter en el controlador y dar paso con "yes" o con la tecla a.
DIR=$(dirname "$(realpath "$0")")
CONTENEDOR=franka_container
# Con -v ~/carpeta:/root/carpeta, la ruta del contenedor es la misma con /root en vez de $HOME
RUTA_CONTENEDOR=/root/${DIR#"$HOME"/}

ESCALA=0.3; V_NORMAL=0.3; V_CAUTIOUS=0.15; RUTINA=--placa; SIMULADO=0; MOSTRAR=0
for arg in "$@"; do
    case $arg in
        --normal) ESCALA=0.6; V_NORMAL=0.6; V_CAUTIOUS=0.3 ;;
        --ciclo) RUTINA="--ciclos 1" ;;
        --simulado) SIMULADO=1 ;;
        --solo-mostrar) MOSTRAR=1 ;;
        *) sed -n '2,7p' "$0"; exit 1 ;;
    esac
done

PY=$DIR/.venv/bin/python
[ -x "$PY" ] || PY=$HOME/aided_building/.venv/bin/python
[ -x "$PY" ] || { echo "[error] no encuentro .venv: mira SETUP.md"; exit 1; }
[ "$MOSTRAR" = 1 ] || command -v gnome-terminal >/dev/null || { echo "[error] hace falta gnome-terminal"; exit 1; }

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
if [ -f "$DIR/datos/placa_referencia.png" ]; then
    pestana Placa "'$PY' '$DIR/percepcion/placa.py'"
else
    echo "Sin comprobar la placa: falta la referencia (percepcion/placa.py --referencia, con la mesa vacía)."
fi
sleep 3                                 # que la voz mida el ruido de fondo antes de que el robot salude
pestana Dialogo "'$PY' '$DIR/dialogo/dialogo.py'"

echo
echo "Listo. Quédate en silencio unos segundos y contesta al robot: 'yes' para empezar."
echo "Para parar: p en la pestaña Supervisor, o el paro de emergencia."
echo "Para apagar: q en Supervisor y Ctrl+C en Stack; luego, en Desk, desactiva la FCI y bloquea las articulaciones."
