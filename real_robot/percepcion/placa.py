#!/usr/bin/env python3
"""¿Está la placa en su sitio? Compara la zona de la placa con una foto de la mesa vacía.

    python3 placa.py --referencia     # una vez, con la mesa VACÍA: guarda la referencia
    python3 placa.py                  # avisa a dialogo.py por UDP 5015 dos veces por segundo
    python3 placa.py --ver            # igual, con una ventana que muestra la zona y la puntuación

Los 4 ArUco de la mesa (datos/mesa.json) dicen dónde cae cada punto de la mesa en la imagen, y las
esquinas grabadas de la placa (datos/poses_info.json) dicen qué zona mirar. Sirve con un cuaderno, un
cartón o una PCB sin entrenar nada, y comprueba que está donde el robot va a soldar. Detecta "hay algo
ahí", no qué es: por eso pide que el cambio se mantenga ESTABLE_S (una mano de paso no cuenta).

Envía {"t": .., "presente": true|false|null, "puntuacion": 0..1}; null = no ve los marcadores.
"""
import argparse
import json
import os
import socket
import sys
import time

import cv2
import numpy as np

from rostro import buscar_camara

DIR_DATOS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "datos"))
REFERENCIA = os.path.join(DIR_DATOS, "placa_referencia.png")
IP = "127.0.0.1"
PUERTO_PLACA = 5015
HZ = 2.0
CAMARA = "C930e"            # la que ve la mesa y los 4 marcadores
ANCHO, ALTO = 1280, 720
PX_POR_M = 1000.0           # resolución de la zona rectificada
MARGEN_DENTRO = 0.015       # m que se ignoran dentro del borde de la placa (la imagen no es exacta)
MARGEN_FUERA = 0.04         # m de mesa alrededor, para compensar los cambios de luz
UMBRAL_GRIS = 25            # diferencia de gris a partir de la cual un píxel "cambió"
UMBRAL_ZONA = 0.35          # fracción de la zona que debe cambiar para decir que hay placa
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

    def rectificar(self, gris):
        destino = np.array([[0, 0], [self.ancho, 0], [self.ancho, self.alto], [0, self.alto]], np.float32)
        m = cv2.getPerspectiveTransform(self.esquinas_imagen(), destino)
        return cv2.GaussianBlur(cv2.warpPerspective(gris, m, (self.ancho, self.alto)), (0, 0), 3)

    def puntuacion(self, recorte, referencia):
        """Fracción de la zona que cambió, compensando la luz con el anillo de mesa."""
        ganancia = float(np.clip(referencia[self.anillo].mean() / max(1.0, recorte[self.anillo].mean()), 0.5, 2.0))
        cambio = np.abs(recorte.astype(np.float32) * ganancia - referencia.astype(np.float32)) > UMBRAL_GRIS
        return float(cambio[self.dentro].mean())


def abrir_camara(nombre):
    camara = cv2.VideoCapture(buscar_camara(nombre), cv2.CAP_V4L2)
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)
    if not camara.isOpened():
        sys.exit("[error] no puedo abrir la cámara %s: ¿la está usando otro programa?" % nombre)
    return camara


def grabar_referencia(camara, zona):
    for _ in range(15):                 # la cámara ajusta la exposición al arrancar
        camara.read()
    recortes = []
    for _ in range(10):
        ok, bgr = camara.read()
        gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if ok and zona.actualizar(gris) == len(zona.arucos):
            recortes.append(zona.rectificar(gris))
    if not recortes:
        sys.exit("[error] la cámara no ve los 4 marcadores de la mesa.")
    cv2.imwrite(REFERENCIA, np.median(recortes, axis=0).astype(np.uint8))
    print("Referencia guardada: %s. Repítela si cambia la cámara o la luz." % REFERENCIA)


def main():
    p = argparse.ArgumentParser(description="Detecta si la placa está en su sitio")
    p.add_argument("--referencia", action="store_true", help="guardar la referencia, con la mesa vacía")
    p.add_argument("--ver", action="store_true", help="ventana con la zona y la puntuación")
    p.add_argument("--camara", default=CAMARA, help="nombre o número de la webcam (por defecto %s)" % CAMARA)
    args = p.parse_args()

    zona = Zona()
    camara = abrir_camara(args.camara)
    if args.referencia:
        grabar_referencia(camara, zona)
        return
    referencia = cv2.imread(REFERENCIA, cv2.IMREAD_GRAYSCALE)
    if referencia is None or referencia.shape != (zona.alto, zona.ancho):
        sys.exit("[error] falta la referencia o las esquinas cambiaron. Con la mesa vacía: placa.py --referencia")

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
            puntuacion = None
            if zona.homografia is not None:
                puntuacion = zona.puntuacion(zona.rectificar(gris), referencia)
                ahora = puntuacion > UMBRAL_ZONA
                if ahora != candidato:
                    candidato, t_candidato = ahora, t_tick
                if presente is None or t_tick - t_candidato >= ESTABLE_S:
                    presente = candidato
            sock.sendto(json.dumps({"t": time.time(), "presente": presente,
                                    "puntuacion": None if puntuacion is None else round(puntuacion, 3)}).encode("utf-8"),
                        (IP, PUERTO_PLACA))
            sys.stdout.write("\r\033[K%s  cambio %s  marcadores %d/4"
                             % ({True: "placa presente", False: "sin placa", None: "no veo los marcadores"}[presente],
                                "--" if puntuacion is None else "%.0f%%" % (100 * puntuacion), vistos))
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
