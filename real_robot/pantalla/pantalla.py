#!/usr/bin/env python3
"""Pantalla para el operador: qué ve y oye el sistema y qué hace el robot. Se ejecuta en el host.

    python3 pantalla.py               # ventana 1280x720
    python3 pantalla.py --completa    # pantalla completa (para el monitor del operador)

Recibe por UDP:
    5011  resumen del supervisor.py (estado, causa, T(t), rostro, voz, robot)
    5012  imagen de la cámara ya anotada por rostro.py (JPEG)
Si falta alguno, lo dice en la pantalla y sigue esperando. q o Esc para salir.
"""
import argparse
import functools
import json
import os
import socket
import time

# Las bibliotecas numéricas usan todos los núcleos por defecto y saturan la CPU: sin kernel RT eso
# provoca communication_constraints_violation en el robot. Tiene que ir antes de importar numpy.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

cv2.setNumThreads(1)

# Fuentes para las ventanas de OpenCV (Qt): al importarse apunta a una carpeta suya que no existe y avisa
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"

IP = "127.0.0.1"
PUERTO_PANEL = 5011
PUERTO_IMAGEN = 5012
ANCHO, ALTO = 1280, 720
ANCHO_CAMARA = 760
HZ = 10.0
SIN_DATOS_S = 1.0           # s sin mensajes -> "sin conexión"

FUENTE = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FUENTE_NEGRITA = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

FONDO, TEXTO, GRIS = (24, 26, 30), (235, 235, 235), (140, 140, 140)
COLOR_ESTADO = {"NORMAL": (46, 160, 67), "CAUTIOUS": (230, 160, 20), "PAUSED": (200, 50, 50),
                "RETRACT": (40, 110, 200), None: (90, 90, 90)}
TITULO = {"NORMAL": "TRABAJANDO", "CAUTIOUS": "MÁS DESPACIO", "PAUSED": "DETENIDO",
          "RETRACT": "ME ALEJO", None: "SIN CONEXIÓN"}
# Mensaje para el operador según estado y causa (confianza.py)
MENSAJE = {
    ("NORMAL", "calma"): "Todo bien: velocidad normal",
    ("CAUTIOUS", "tension"): "Voy más lento: noto tensión",
    ("CAUTIOUS", "sin_percepcion"): "Voy más lento: no te veo ni te oigo bien",
    ("CAUTIOUS", "calma"): "Te noto más tranquilo: voy retomando",
    ("CAUTIOUS", "voz"): "Voy más lento, como me pediste",
    ("PAUSED", "estres_alto"): "Me detuve: noto estrés. Tómate un momento",
    ("PAUSED", "inicio"): "Listo para empezar",
    ("NORMAL", "manual"): "Velocidad normal",
    ("CAUTIOUS", "manual"): "Voy más lento",
    ("PAUSED", "manual"): "En pausa",
    ("RETRACT", "manual"): "Me alejé de la placa",
}
EXPRESION = {"angry": "Enfado", "disgust": "Asco", "fear": "Miedo", "fearful": "Miedo",
             "happy": "Alegría", "sad": "Tristeza", "surprise": "Sorpresa", "surprised": "Sorpresa",
             "neutral": "Neutral"}


@functools.lru_cache(maxsize=None)
def fuente(tam, negrita=False):
    """Cacheada: abrir la fuente en cada texto de cada cuadro gasta mucha CPU."""
    return ImageFont.truetype(FUENTE_NEGRITA if negrita else FUENTE, tam)


def texto_ajustado(d, xy, texto, ancho_max, tam, negrita=False, color=TEXTO):
    """Escribe el texto reduciendo la letra hasta que quepa en ancho_max."""
    while tam > 14 and d.textlength(texto, font=fuente(tam, negrita)) > ancho_max:
        tam -= 1
    d.text(xy, texto, font=fuente(tam, negrita), fill=color)


def color_estres(e):
    """Verde (0) -> amarillo (0.5) -> rojo (1)."""
    e = min(1.0, max(0.0, e))
    return (int(min(1.0, 2 * e) * 220), int(min(1.0, 2 * (1 - e)) * 180), 40)


def barra(d, x, y, ancho, valor, color):
    d.rounded_rectangle((x, y, x + ancho, y + 14), 7, fill=(55, 58, 64))
    if valor is not None:
        d.rounded_rectangle((x, y, x + max(14, int(ancho * min(1.0, max(0.0, valor)))), y + 14), 7, fill=color)


def actividad_robot(robot, estado):
    """Texto y progreso (0..1) de lo que hace el robot."""
    if not robot or not robot.get("conectado"):
        return "Robot no conectado", None
    if robot.get("done"):
        return "Rutina terminada", 1.0
    seg, accion = robot.get("segment") or "", robot.get("action") or ""
    n = max(1, robot.get("n_segments") or 1)
    progreso = min(1.0, (robot.get("seg_index") or 0) / float(n))
    punto = seg.split("_p")[-1] if "_p" in seg else ""
    if accion == "retraccion":
        texto = "Alejándome de la placa"
    elif accion == "soldando":
        texto = "Soldando el punto %s" % punto if punto else "Soldando"
    else:
        texto = {"sobre": "Yendo al punto %s" % punto, "baja": "Bajando al punto %s" % punto,
                 "sube": "Subiendo del punto %s" % punto}.get(seg.split("_")[0]) or {
                 "home": "Volviendo a la posición inicial", "aproximacion": "Acercándome a la placa",
                 "soldadura": "Bajando a soldar", "retraccion": "Retirándome", "fin": "Rutina terminada"
                 }.get(seg, seg)
    total = " de %d" % max(1, (n - 2) // 3) if punto else ""
    if not accion and estado in ("PAUSED", "RETRACT"):
        texto = "En pausa (punto %s%s)" % (punto, total) if punto else "En pausa"
    elif punto:
        texto += total
    return texto, progreso


def seccion_fuente(d, x, y, titulo, datos, vacio):
    d.text((x, y), titulo, font=fuente(18, True), fill=GRIS)
    if datos and datos.get("valida"):
        exp = EXPRESION.get((datos.get("etiqueta") or "").lower(), datos.get("etiqueta") or "?")
        estres = float(datos.get("estres") or 0.0)
        d.text((x, y + 26), exp, font=fuente(30, True), fill=TEXTO)
        d.text((x + 250, y + 36), "confianza %.0f%%" % (100 * float(datos.get("confianza") or 0)),
               font=fuente(15), fill=GRIS)
        d.text((x, y + 70), "estrés", font=fuente(15), fill=GRIS)
        barra(d, x + 60, y + 72, 380, estres, color_estres(estres))
    elif datos and datos.get("etiqueta"):             # último resultado, ya sin peso en la confianza
        exp = EXPRESION.get(datos["etiqueta"].lower(), datos["etiqueta"])
        d.text((x, y + 26), "%s (%s)" % (exp, vacio), font=fuente(26), fill=GRIS)
        d.text((x, y + 70), "estrés", font=fuente(15), fill=GRIS)
        barra(d, x + 60, y + 72, 380, float(datos.get("estres") or 0.0), GRIS)
    else:
        d.text((x, y + 26), vacio if datos else "sin señal", font=fuente(26), fill=GRIS)
        d.text((x, y + 70), "estrés", font=fuente(15), fill=GRIS)
        barra(d, x + 60, y + 72, 380, None, None)


def componer(panel, t_panel, camara, t_camara, ahora):
    img = Image.new("RGB", (ANCHO, ALTO), FONDO)
    d = ImageDraw.Draw(img)

    # Cámara (izquierda)
    if camara is not None and ahora - t_camara <= SIN_DATOS_S:
        rgb = cv2.cvtColor(camara, cv2.COLOR_BGR2RGB)
        escala = min(ANCHO_CAMARA / float(rgb.shape[1]), (ALTO - 40) / float(rgb.shape[0]))
        foto = Image.fromarray(rgb).resize((int(rgb.shape[1] * escala), int(rgb.shape[0] * escala)))
        img.paste(foto, (20 + (ANCHO_CAMARA - foto.width) // 2, (ALTO - foto.height) // 2))
    else:
        d.rounded_rectangle((20, 20, 20 + ANCHO_CAMARA, ALTO - 20), 12, outline=(70, 70, 70), width=2)
        d.text((60, ALTO // 2 - 20), "Sin cámara (¿está corriendo rostro.py?)", font=fuente(24), fill=GRIS)

    # Panel (derecha)
    x = ANCHO_CAMARA + 50
    ancho = ANCHO - x - 20
    conectado = panel is not None and ahora - t_panel <= SIN_DATOS_S
    estado = panel.get("estado") if conectado else None
    d.rounded_rectangle((x, 20, x + ancho, 150), 14, fill=COLOR_ESTADO.get(estado, COLOR_ESTADO[None]))
    d.text((x + 20, 32), TITULO.get(estado, estado or ""), font=fuente(44, True), fill=(255, 255, 255))
    if conectado:
        mensaje = MENSAJE.get((estado, panel.get("causa")), panel.get("motivo") or "")
    else:
        mensaje = "¿Está corriendo supervisor.py?"
    texto_ajustado(d, (x + 20, 98), mensaje, ancho - 40, 21, color=(255, 255, 255))

    fuentes = (panel or {}).get("fuentes", {}) if conectado else {}
    seccion_fuente(d, x, 180, "TU CARA", fuentes.get("rostro"), "no te veo")
    seccion_fuente(d, x, 290, "TU VOZ", fuentes.get("audio"), "silencio")

    d.text((x, 400), "CONFIANZA", font=fuente(18, True), fill=GRIS)
    T = panel.get("T") if conectado else None
    d.text((x + 140, 400), "--" if T is None else "%.0f%%" % (100 * T), font=fuente(18, True), fill=TEXTO)
    barra(d, x, 430, ancho - 10, T, color_estres(1.0 - T) if T is not None else None)

    d.text((x, 480), "ROBOT", font=fuente(18, True), fill=GRIS)
    texto, progreso = actividad_robot((panel or {}).get("robot") if conectado else None, estado)
    texto_ajustado(d, (x, 508), texto, ancho - 10, 26, negrita=True)
    barra(d, x, 550, ancho - 10, progreso, (90, 140, 220))

    pie = "modo %s" % ("automático" if conectado and panel.get("modo") == "AUTO" else "manual") if conectado else ""
    d.text((x, ALTO - 45), pie, font=fuente(15), fill=GRIS)
    d.text((x + ancho - 70, ALTO - 45), time.strftime("%H:%M:%S"), font=fuente(15), fill=GRIS)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def main():
    p = argparse.ArgumentParser(description="Pantalla para el operador")
    p.add_argument("--completa", action="store_true", help="pantalla completa")
    args = p.parse_args()

    sock_panel = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_panel.bind((IP, PUERTO_PANEL))
    sock_panel.setblocking(False)
    sock_imagen = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_imagen.bind((IP, PUERTO_IMAGEN))
    sock_imagen.setblocking(False)

    ventana = "Robot soldador"
    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    if args.completa:
        cv2.setWindowProperty(ventana, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(ventana, ANCHO, ALTO)

    panel, t_panel, camara, t_camara = None, 0.0, None, 0.0
    try:
        while True:
            t_tick = time.time()
            while True:
                try:
                    datos, _ = sock_panel.recvfrom(65536)
                    panel, t_panel = json.loads(datos.decode("utf-8")), t_tick
                except (BlockingIOError, socket.error):
                    break
                except ValueError:
                    pass
            while True:
                try:
                    datos, _ = sock_imagen.recvfrom(65536)
                except (BlockingIOError, socket.error):
                    break
                imagen = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
                if imagen is not None:
                    camara, t_camara = imagen, t_tick

            cv2.imshow(ventana, componer(panel, t_panel, camara, t_camara, t_tick))
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
            time.sleep(max(0.0, 1.0 / HZ - (time.time() - t_tick)))
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
