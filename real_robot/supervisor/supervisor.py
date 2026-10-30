#!/usr/bin/env python3
"""Supervisor de confianza (host): percepción -> T(t) -> NORMAL/CAUTIOUS/PAUSED -> robot.

    python3 supervisor.py --v-normal 0.6 --v-cautious 0.3

Recibe la percepción (rostro.py, voz.py) en UDP 5010, calcula T(t) y la máquina de estados
de confianza.py, y manda el estado al controlador por UDP 5005, igual que
supervisor_teclado.py. Envía un resumen a pantalla.py por UDP 5011.
Órdenes de voz de dialogo.py (UDP 5013): empezar = a; pausa = p; lento = CAUTIOUS 20 s;
seguir = automático desde CAUTIOUS 20 s. Mientras el robot habla, ignora la emoción de la voz.
Teclas:  a = AUTOMÁTICO (el estado lo decide T(t))
         n / c / p / r = MANUAL: NORMAL / CAUTIOUS / PAUSED / RETRACT (anulan el automático)
         q = salir
Arranca en MANUAL + PAUSED: el robot no se mueve hasta que pulses a (o n/c).
CSV en logs/supervisor_auto_<fecha>.csv con T(t), cada fuente, el estado y el robot.
"""
import argparse
import csv
import json
import os
import select
import socket
import sys
import termios
import time
import tty

from confianza import Confianza, MaquinaEstados

IP = "127.0.0.1"
PUERTO_CMD = 5005
PUERTO_ESTADO = 5006
PUERTO_PERCEPCION = 5010
PUERTO_PANEL = 5011         # resumen para pantalla.py
PUERTO_COMANDOS = 5013      # órdenes de dialogo.py (voz)
PUERTO_DIALOGO = 5014       # resumen para dialogo.py
HZ_PANEL = 10.0
MARGEN_ECO = 0.5            # s tras hablar el robot en que se ignora la emoción de la voz
S_VOZ_LENTO = 20.0          # s que dura un "slow down" / "continue" por voz antes de poder subir
HZ = 20.0
HZ_PANTALLA = 5.0           # redibujados por segundo como máximo (el envío UDP sigue a HZ)
TIMEOUT_ROBOT = 0.5
FUENTES = ("rostro", "audio")

TECLAS = {"n": "NORMAL", "c": "CAUTIOUS", "p": "PAUSED", "r": "RETRACT"}
COLOR = {"NORMAL": "\033[42;30m", "CAUTIOUS": "\033[43;30m", "PAUSED": "\033[41;37m",
         "RETRACT": "\033[44;37m"}
RESET = "\033[0m"

COLUMNAS = (["t_host", "tecla", "modo", "cmd_state", "cmd_speed_scale", "cmd_retract", "T", "estres",
             "sin_datos", "motivo"]
            + ["%s_%s" % (f, k) for f in FUENTES for k in ("estres", "confianza", "etiqueta", "edad_s")]
            + ["rob_edad_s", "rob_state", "rob_segment", "rob_action", "rob_speed_scale", "rob_event",
               "ee_x", "ee_y", "ee_z"])


def color(estado):
    return COLOR.get(estado, "") + " %-8s " % estado + RESET


def leer_todo(sock):
    """Todos los JSON pendientes en el socket (los que no parsean se ignoran)."""
    mensajes = []
    while True:
        try:
            datos, _ = sock.recvfrom(65536)
        except (BlockingIOError, socket.error):
            return mensajes
        try:
            mensajes.append(json.loads(datos.decode("utf-8")))
        except ValueError:
            pass


def main():
    p = argparse.ArgumentParser(description="Supervisor de confianza: percepción -> estado del robot")
    p.add_argument("--v-normal", type=float, default=0.3, help="speed_scale en NORMAL")
    p.add_argument("--v-cautious", type=float, default=0.15, help="speed_scale en CAUTIOUS")
    p.add_argument("--log-dir", default=os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, "logs")))
    args = p.parse_args()
    velocidad = {"NORMAL": args.v_normal, "CAUTIOUS": args.v_cautious, "PAUSED": 0.0, "RETRACT": 0.0}
    if not sys.stdin.isatty():
        sys.exit("[error] Hace falta una terminal para leer el teclado.")

    os.makedirs(args.log_dir, exist_ok=True)
    ruta_csv = os.path.join(args.log_dir, time.strftime("supervisor_auto_%Y%m%d_%H%M%S.csv"))
    f_csv = open(ruta_csv, "w", newline="")
    log = csv.writer(f_csv)
    log.writerow(COLUMNAS)

    sock_cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_estado = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_estado.bind((IP, PUERTO_ESTADO))
    sock_estado.setblocking(False)
    sock_perc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_perc.bind((IP, PUERTO_PERCEPCION))
    sock_perc.setblocking(False)
    sock_voz = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_voz.bind((IP, PUERTO_COMANDOS))
    sock_voz.setblocking(False)
    silencio_audio_hasta = 0.0

    confianza, fsm = Confianza(), MaquinaEstados()
    modo, manual = "MANUAL", "PAUSED"
    robot, t_robot, ultimo_evento = {}, None, None
    texto_prev, t_pantalla, t_flush, t_panel = "", 0.0, 0.0, 0.0

    def linea(texto):
        nonlocal texto_prev
        sys.stdout.write("\r\033[K" + texto + "\n")
        texto_prev = ""

    print("Supervisor de confianza. a = AUTOMÁTICO | n/c/p/r = manual NORMAL/CAUTIOUS/PAUSED/RETRACT"
          " | q = salir")
    print("NORMAL %.2f, CAUTIOUS %.2f. Empieza en MANUAL + PAUSED. Percepción en UDP %d. Log: %s\n"
          % (args.v_normal, args.v_cautious, PUERTO_PERCEPCION, ruta_csv))

    fd = sys.stdin.fileno()
    config_terminal = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    periodo = 1.0 / HZ
    try:
        while True:
            t_tick = time.time()

            # Teclado
            tecla = ""
            if select.select([fd], [], [], 0)[0]:
                tecla = os.read(fd, 32).decode("utf-8", "ignore")[-1:].lower()
                if tecla == "q":
                    break
                if tecla == "a":
                    modo = "AUTO"
                    linea("[tecla a] modo AUTOMÁTICO: el estado lo decide T(t)")
                elif tecla in TECLAS:
                    modo, manual = "MANUAL", TECLAS[tecla]
                    linea("[tecla %s] modo MANUAL -> %s" % (tecla, color(manual)))
                elif tecla.strip():
                    linea("[tecla %r] no reconocida (a, n, c, p, r, q)" % tecla)

            # Órdenes de voz (dialogo.py); se registran en la columna "tecla" como voz:<orden>
            for m in leer_todo(sock_voz):
                orden = m.get("comando")
                if orden == "hablando":
                    silencio_audio_hasta = float(m.get("hasta", 0.0)) + MARGEN_ECO
                    continue
                if orden == "empezar":
                    modo = "AUTO"
                elif orden == "pausa":
                    modo, manual = "MANUAL", "PAUSED"
                elif orden == "seguir":
                    modo = "AUTO"
                    fsm.forzar("CAUTIOUS", "voz", S_VOZ_LENTO, t_tick)
                elif orden == "lento":
                    if modo == "AUTO":
                        fsm.forzar("CAUTIOUS", "voz", S_VOZ_LENTO, t_tick)
                    else:
                        manual = "CAUTIOUS"
                else:
                    continue
                tecla = "voz:" + orden
                linea("[voz] %s -> modo %s, %s" % (orden, modo, color(fsm.estado if modo == "AUTO" else manual)))

            # Percepción -> T(t) -> máquina de estados (corre siempre, también en manual)
            for m in leer_todo(sock_perc):
                if m.get("fuente") == "audio" and t_tick < silencio_audio_hasta:
                    continue                    # es el robot hablando, no el operador
                confianza.recibir(m, t_tick)
            T = confianza.actualizar(t_tick)
            sin_datos = confianza.sin_datos(t_tick)
            if fsm.actualizar(T, sin_datos, t_tick):
                linea("[confianza] %s %s: %s" % ("->" if modo == "AUTO" else "(sugerido, modo manual)",
                                                 color(fsm.estado), fsm.motivo))

            estado = fsm.estado if modo == "AUTO" else manual
            cmd = {"t": time.time(), "enabled": estado != "PAUSED", "speed_scale": velocidad[estado],
                   "retract": estado == "RETRACT", "state": "PAUSED" if estado == "RETRACT" else estado,
                   "trust": None if T is None else round(T, 3),
                   "reason": fsm.motivo if modo == "AUTO" else "manual"}
            sock_cmd.sendto(json.dumps(cmd).encode("utf-8"), (IP, PUERTO_CMD))

            # Estado del robot
            for m in leer_todo(sock_estado):
                robot, t_robot = m, t_tick
            edad = t_tick - t_robot if t_robot is not None else None
            if robot.get("event") and robot.get("event") != ultimo_evento:
                ultimo_evento = robot["event"]
                linea("[robot] " + ultimo_evento)

            # Resumen para la pantalla del operador
            validas = confianza.validas(t_tick)
            if t_tick - t_panel >= 1.0 / HZ_PANEL:
                t_panel = t_tick
                fuentes = {}
                for f, (t_f, m) in confianza.ultimo.items():
                    fuentes[f] = {"etiqueta": m.get("etiqueta") or "", "estres": m.get("estres"),
                                  "confianza": m.get("confianza"), "valida": f in validas,
                                  "edad": round(t_tick - t_f, 2)}
                panel = {"t": t_tick, "modo": modo, "estado": estado,
                         "causa": fsm.causa if modo == "AUTO" else "manual", "motivo": cmd["reason"],
                         "T": cmd["trust"], "fuentes": fuentes,
                         "robot": {"conectado": edad is not None and edad <= TIMEOUT_ROBOT,
                                   "segment": robot.get("segment", ""), "action": robot.get("action", ""),
                                   "seg_index": robot.get("seg_index", 0), "n_segments": robot.get("n_segments", 0),
                                   "done": robot.get("done", False)}}
                for puerto in (PUERTO_PANEL, PUERTO_DIALOGO):
                    sock_cmd.sendto(json.dumps(panel).encode("utf-8"), (IP, puerto))

            # Pantalla del terminal
            fuentes_txt = []
            for f in FUENTES:
                if f in validas:
                    m = confianza.ultimo[f][1]
                    fuentes_txt.append("%s %s e%.2f c%.2f" % (f, m.get("etiqueta") or "", validas[f][0], validas[f][1]))
                elif f in confianza.ultimo:
                    fuentes_txt.append("%s --" % f)
            T_txt = "T=--  " if T is None else "T=%.2f" % T
            if edad is None:
                rob_txt = "\033[41;37m SIN ROBOT \033[0m"
            elif edad > TIMEOUT_ROBOT:
                rob_txt = "\033[41;37m ROBOT PERDIDO \033[0m"
            else:
                rob_txt = "%s %s" % (robot.get("segment", "?"), {"segmento": "MOVIENDO", "retraccion": "SUBIENDO",
                                     "soldando": "SOLDANDO"}.get(robot.get("action"), "quieto"))
            texto = "\r\033[K%-6s %s %s | %s | %s" % (modo, T_txt, color(estado),
                                                     ", ".join(fuentes_txt) or "sin percepción", rob_txt)
            if texto != texto_prev and t_tick - t_pantalla >= 1.0 / HZ_PANTALLA:
                sys.stdout.write(texto)
                sys.stdout.flush()
                texto_prev, t_pantalla = texto, t_tick

            # CSV
            fila = [round(t_tick, 3), tecla, modo, cmd["state"], cmd["speed_scale"], cmd["retract"],
                    "" if T is None else round(T, 4), "" if confianza.estres is None else round(confianza.estres, 4),
                    sin_datos, fsm.motivo]
            for f in FUENTES:
                if f in confianza.ultimo:
                    t_f, m = confianza.ultimo[f]
                    fila += [m.get("estres"), m.get("confianza"), m.get("etiqueta"), round(t_tick - t_f, 3)]
                else:
                    fila += ["", "", "", ""]
            ee = robot.get("ee_xyz") or [""] * 3
            fila += ["" if edad is None else round(edad, 3)] + [robot.get(k, "") for k in
                     ("state", "segment", "action", "speed_scale", "event")] + list(ee)
            log.writerow(fila)
            if t_tick - t_flush >= 1.0:
                f_csv.flush()
                t_flush = t_tick

            time.sleep(max(0.0, periodo - (time.time() - t_tick)))
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(5):
            sock_cmd.sendto(json.dumps({"t": time.time(), "enabled": False, "speed_scale": 0.0,
                                        "retract": False, "state": "PAUSED"}).encode("utf-8"),
                            (IP, PUERTO_CMD))
            time.sleep(0.02)
        termios.tcsetattr(fd, termios.TCSADRAIN, config_terminal)
        f_csv.close()
        print("\nSupervisor cerrado (último comando: PAUSED). Log en %s" % ruta_csv)


if __name__ == "__main__":
    main()
