#!/usr/bin/env python3
"""¿Está la placa en su sitio? Busca un rectángulo de otro color sobre la base blanca.

    python3 placa.py          # avisa a dialogo.py por UDP 5015 dos veces por segundo
    python3 placa.py --ver    # igual, con una ventana que muestra la zona y lo que detecta

Los 4 ArUco de la mesa (datos/mesa.json) dicen dónde cae cada punto de la mesa en la imagen, y las
esquinas grabadas de la placa (datos/poses_info.json) dicen qué zona mirar. Dentro de ella se separa lo
que no es del color de la base, y hay placa si es una forma rectangular que cubre la zona. Una mano no
cuenta porque no es rectangular. Sirve con un cuaderno, un cartón o una PCB, siempre que no sean blancos.

Envía {"t": .., "presente": true|false|null, "puntuacion": 0..1}; null = no ve los marcadores.
"""
import argparse
import json
import os
import socket
import sys
import time

from rostro import buscar_camara     # limita los hilos antes de importar numpy

import cv2
import numpy as np

cv2.setNumThreads(1)

DIR_DATOS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "datos"))
IP = "127.0.0.1"
PUERTO_PLACA = 5015
HZ = 2.0
CAMARA = "C930e"            # la que ve la mesa y los 4 marcadores
ANCHO, ALTO = 1280, 720
PX_POR_M = 1000.0           # resolución de la zona rectificada
MARGEN_DENTRO = 0.015       # m que se ignoran dentro del borde de la placa (la imagen no es exacta)
MARGEN_FUERA = 0.04         # m de base alrededor: de ahí se saca el color de la base
UMBRAL_COLOR = 30.0         # distancia de color (Lab) al color de la base para decir que no es base
MIN_COBERTURA = 0.60        # fracción de la zona que debe cubrir la forma
MIN_RECTANGULO = 0.80       # área de la forma / área de su rectángulo envolvente
ESTABLE_S = 1.5             # s que debe mantenerse un cambio de estado


def cargar_mesa():
    with open(os.path.join(DIR_DATOS, "mesa.json")) as f:
        mesa = json.load(f)
    with open(os.path.join(DIR_DATOS, "poses_info.json")) as f:
        info = json.load(f)
    esquinas = np.array([info["esquina_%d" % i]["punta_xyz"][:2] for i in range(1, 5)], np.float32)
    return mesa["arucos"], esquinas


class Zona(object):
    """Rectifica la zona de la placa (con un anillo de mesa alrededor) a partir de los ArUco."""

    def __init__(self):
        self.arucos, e = cargar_mesa()
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                                cv2.aruco.DetectorParameters())
        self.homografia = None
        largo_u, largo_v = np.linalg.norm(e[1] - e[0]), np.linalg.norm(e[3] - e[0])
        au, av = MARGEN_FUERA / largo_u, MARGEN_FUERA / largo_v
        # Esquinas agrandadas: e[0..3] = u, v (0,0), (1,0), (1,1), (0,1) como en plan_placa.bilineal

        def punto(u, v):
            return (1 - u) * (1 - v) * e[0] + u * (1 - v) * e[1] + u * v * e[2] + (1 - u) * v * e[3]

        self.mesa = np.array([punto(-au, -av), punto(1 + au, -av), punto(1 + au, 1 + av), punto(-au, 1 + av)],
                             np.float32)
        self.ancho = int(round((largo_u + 2 * MARGEN_FUERA) * PX_POR_M))
        self.alto = int(round((largo_v + 2 * MARGEN_FUERA) * PX_POR_M))
        self.dentro = np.zeros((self.alto, self.ancho), bool)
        borde = int((MARGEN_FUERA + MARGEN_DENTRO) * PX_POR_M)
        self.dentro[borde:-borde, borde:-borde] = True
        self.anillo = np.ones((self.alto, self.ancho), bool)
        self.anillo[int(MARGEN_FUERA * PX_POR_M):-int(MARGEN_FUERA * PX_POR_M),
                    int(MARGEN_FUERA * PX_POR_M):-int(MARGEN_FUERA * PX_POR_M)] = False

    def actualizar(self, gris):
        """Recalcula la homografía si ve los 4 marcadores; si no, conserva la última."""
        esquinas, ids, _ = self.detector.detectMarkers(gris)
        vistos = {} if ids is None else {str(int(i)): c[0].mean(axis=0) for i, c in zip(ids.ravel(), esquinas)}
        if all(k in vistos for k in self.arucos):
            origen = np.array([self.arucos[k] for k in self.arucos], np.float32)
            destino = np.array([vistos[k] for k in self.arucos], np.float32)
            self.homografia = cv2.findHomography(origen, destino)[0]
        return len(vistos)

    def esquinas_imagen(self):
        return cv2.perspectiveTransform(self.mesa[None], self.homografia)[0]

    def rectificar(self, bgr):
        destino = np.array([[0, 0], [self.ancho, 0], [self.ancho, self.alto], [0, self.alto]], np.float32)
        m = cv2.getPerspectiveTransform(self.esquinas_imagen(), destino)
        return cv2.GaussianBlur(cv2.warpPerspective(bgr, m, (self.ancho, self.alto)), (0, 0), 3)

    def analizar(self, recorte):
        """(cobertura de la zona, cuán rectangular es) de la mayor forma que no es del color de la base."""
        lab = cv2.cvtColor(recorte, cv2.COLOR_BGR2LAB).astype(np.float32)
        base = np.median(lab[self.anillo], axis=0)
        forma = (np.linalg.norm(lab - base, axis=2) > UMBRAL_COLOR).astype(np.uint8)
        forma = cv2.morphologyEx(forma, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        contornos, _ = cv2.findContours(forma, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos:
            return 0.0, 0.0
        mayor = max(contornos, key=cv2.contourArea)
        ancho, alto = cv2.minAreaRect(mayor)[1]
        rellena = np.zeros_like(forma)
        cv2.drawContours(rellena, [mayor], -1, 1, -1)
        cobertura = float(rellena[self.dentro].mean())
        return cobertura, float(cv2.contourArea(mayor) / max(1.0, ancho * alto))


def abrir_camara(nombre):
    camara = cv2.VideoCapture(buscar_camara(nombre), cv2.CAP_V4L2)
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)
    if not camara.isOpened():
        sys.exit("[error] no puedo abrir la cámara %s: ¿la está usando otro programa?" % nombre)
    return camara


def main():
    p = argparse.ArgumentParser(description="Detecta si la placa está en su sitio")
    p.add_argument("--ver", action="store_true", help="ventana con la zona y la puntuación")
    p.add_argument("--camara", default=CAMARA, help="nombre o número de la webcam (por defecto %s)" % CAMARA)
    args = p.parse_args()

    zona = Zona()
    camara = abrir_camara(args.camara)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("Mirando la zona de la placa. Enviando a %s:%d. Ctrl+C para salir." % (IP, PUERTO_PLACA))
    presente, candidato, t_candidato = None, None, 0.0
    try:
        while True:
            t_tick = time.time()
            for _ in range(2):          # descartar la imagen que quedó en el búfer
                ok, bgr = camara.read()
            if not ok:
                sys.exit("[error] la cámara dejó de enviar imágenes")
            gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            vistos = zona.actualizar(gris)
            puntuacion, rectangulo = None, 0.0
            if zona.homografia is not None:
                puntuacion, rectangulo = zona.analizar(zona.rectificar(bgr))
                ahora = puntuacion >= MIN_COBERTURA and rectangulo >= MIN_RECTANGULO
                if ahora != candidato:
                    candidato, t_candidato = ahora, t_tick
                if presente is None or t_tick - t_candidato >= ESTABLE_S:
                    presente = candidato
            sock.sendto(json.dumps({"t": time.time(), "presente": presente,
                                    "puntuacion": None if puntuacion is None else round(puntuacion, 3)}).encode("utf-8"),
                        (IP, PUERTO_PLACA))
            sys.stdout.write("\r\033[K%s  cubre %s  rectangular %.0f%%  marcadores %d/4"
                             % ({True: "placa presente", False: "sin placa", None: "no veo los marcadores"}[presente],
                                "--" if puntuacion is None else "%.0f%%" % (100 * puntuacion), 100 * rectangulo, vistos))
            sys.stdout.flush()
            if args.ver:
                if zona.homografia is not None:
                    cv2.polylines(bgr, [zona.esquinas_imagen().astype(np.int32)], True,
                                  (0, 200, 0) if presente else (0, 0, 255), 2)
                cv2.imshow("placa (q para salir)", bgr)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            time.sleep(max(0.0, 1.0 / HZ - (time.time() - t_tick)))
    except KeyboardInterrupt:
        pass
    finally:
        camara.release()
        cv2.destroyAllWindows()
        print()


if __name__ == "__main__":
    main()
