#!/usr/bin/env python3
"""¿Hay manos en la placa, junto al robot? YOLO de manos sobre la cámara de la mesa.

    python3 manos.py                # avisa al supervisor por UDP 5016 y a dialogo.py (placa) por 5015
    python3 manos.py --ver          # igual, con una ventana: zona vigilada y manos detectadas
    python3 manos.py --margen-cm 8  # vigilar 8 cm alrededor de la placa en vez de 5

Usa la misma cámara que placa.py (la que ve los 4 ArUco), y una cámara sólo se puede abrir desde un
programa: por eso este hace también la comprobación de la placa y envía lo mismo que placa.py.
Los ArUco dicen dónde caen en la imagen las esquinas grabadas de la placa; la zona vigilada es la
placa agrandada --margen-cm. Una mano cuenta si su recuadro toca esa zona. La mano está por encima
de la mesa y la cámara la ve en perspectiva: el margen cubre ese error.

Envía {"t": .., "cerca": true|false|null, "manos": n, "confianza": 0..1}; null = no ve los marcadores
y no sabe dónde está la placa. "cerca" se activa con 2 de las 3 últimas imágenes y se apaga tras
1 s sin manos en la zona: avisar rápido, soltar despacio.

Modelo: modelos/manos_yolov8n.pt (YOLOv8n de manos, ver modelos/MANOS_YOLO.txt).
"""
import argparse
import collections
import json
import os
import socket
import sys
import time

# Ultralytics manda estadísticas de uso por defecto: aquí no
os.environ.setdefault("YOLO_OFFLINE", "1")

from rostro import buscar_camara     # limita los hilos antes de importar numpy  # noqa: F401
from placa import Zona, DetectorPlaca, abrir_camara, cargar_mesa, IP, PUERTO_PLACA

import cv2
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
MODELO = os.path.join(DIR, "modelos", "manos_yolov8n.pt")
PUERTO_MANOS = 5016
CAMARA = "C930e"            # la que ve la mesa y los 4 marcadores, como placa.py
HZ = 10.0
HZ_PLACA = 2.0
MARGEN_CM = 5.0             # alrededor de la placa
CONF_MIN = 0.4              # confianza mínima de YOLO
IMGSZ = 640
VENTANA = 3                 # "cerca" si hay manos en la zona en ...
MIN_DETECCIONES = 2         # ... al menos estas de las últimas VENTANA imágenes
SOLTAR_S = 1.0              # s seguidos sin manos en la zona para dejar de avisar


def zona_vigilada(margen):
    """Esquinas de la placa agrandadas `margen` m, en el marco de la base del robot (como Zona.mesa)."""
    _, e = cargar_mesa()
    largo_u, largo_v = np.linalg.norm(e[1] - e[0]), np.linalg.norm(e[3] - e[0])
    au, av = margen / largo_u, margen / largo_v

    def punto(u, v):
        return (1 - u) * (1 - v) * e[0] + u * (1 - v) * e[1] + u * v * e[2] + (1 - u) * v * e[3]

    return np.array([punto(-au, -av), punto(1 + au, -av), punto(1 + au, 1 + av), punto(-au, 1 + av)], np.float32)


def toca_zona(caja, zona_img):
    """True si el recuadro (x1, y1, x2, y2) en píxeles se solapa con el polígono de la zona."""
    x1, y1, x2, y2 = caja
    rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    area, _ = cv2.intersectConvexConvex(rect, zona_img)
    return area > 0.0


class Vigilancia(object):
    """Pasa de detecciones por imagen a un "cerca" estable."""

    def __init__(self):
        self.ultimas = collections.deque(maxlen=VENTANA)
        self.cerca = False
        self.t_ultima = None        # última imagen con manos en la zona

    def actualizar(self, hay, t):
        self.ultimas.append(hay)
        if hay:
            self.t_ultima = t
        if sum(self.ultimas) >= MIN_DETECCIONES:
            self.cerca = True
        elif self.cerca and (self.t_ultima is None or t - self.t_ultima >= SOLTAR_S):
            self.cerca = False
        return self.cerca


def main():
    p = argparse.ArgumentParser(description="Detecta manos en la zona de la placa")
    p.add_argument("--ver", action="store_true", help="ventana con la zona y las manos")
    p.add_argument("--camara", default=CAMARA, help="nombre o número de la webcam (por defecto %s)" % CAMARA)
    p.add_argument("--margen-cm", type=float, default=MARGEN_CM, help="cm vigilados alrededor de la placa")
    p.add_argument("--conf", type=float, default=CONF_MIN, help="confianza mínima de YOLO (0..1)")
    p.add_argument("--imgsz", type=int, default=IMGSZ, help="tamaño de entrada de YOLO")
    p.add_argument("--hz", type=float, default=HZ)
    p.add_argument("--sin-placa", action="store_true", help="no comprobar la placa (si placa.py usa otra cámara)")
    p.add_argument("--cpu", action="store_true", help="usar la CPU aunque haya GPU")
    args = p.parse_args()

    import torch
    from ultralytics import YOLO, settings
    settings.update({"sync": False})
    torch.set_num_threads(1)
    dispositivo = 0 if torch.cuda.is_available() and not args.cpu else "cpu"
    modelo = YOLO(MODELO)
    print("YOLO de manos en %s" % (torch.cuda.get_device_name(0) if dispositivo == 0 else "CPU"))

    zona = Zona()
    placa = None if args.sin_placa else DetectorPlaca(zona)
    zona_mesa = zona_vigilada(args.margen_cm / 100.0)
    vigilancia = Vigilancia()
    camara = abrir_camara(args.camara)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("Vigilando %.0f cm alrededor de la placa. Manos a %s:%d%s. Ctrl+C para salir."
          % (args.margen_cm, IP, PUERTO_MANOS, "" if args.sin_placa else ", placa a %d" % PUERTO_PLACA))
    t_placa = 0.0
    try:
        while True:
            t_tick = time.time()
            for _ in range(2):          # descartar la imagen que quedó en el búfer
                ok, bgr = camara.read()
            if not ok:
                sys.exit("[error] la cámara dejó de enviar imágenes")
            vistos = zona.actualizar(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))

            resultado = modelo(bgr, imgsz=args.imgsz, conf=args.conf, device=dispositivo, verbose=False)[0]
            cajas = [(c.xyxy[0].tolist(), float(c.conf[0])) for c in resultado.boxes]
            zona_img, en_zona = None, []
            if zona.homografia is not None:
                zona_img = cv2.perspectiveTransform(zona_mesa[None], zona.homografia)[0]
                en_zona = [conf for caja, conf in cajas if toca_zona(caja, zona_img)]
                cerca = vigilancia.actualizar(bool(en_zona), t_tick)
            else:
                cerca = None
            sock.sendto(json.dumps({"t": time.time(), "cerca": cerca, "manos": len(en_zona),
                                    "confianza": round(max(en_zona), 3) if en_zona else 0.0}).encode("utf-8"),
                        (IP, PUERTO_MANOS))

            if placa is not None and t_tick - t_placa >= 1.0 / HZ_PLACA:
                t_placa = t_tick
                placa.actualizar(bgr, t_tick)
                sock.sendto(json.dumps(placa.mensaje()).encode("utf-8"), (IP, PUERTO_PLACA))

            sys.stdout.write("\r\033[K%s  manos vistas %d, en la zona %d  marcadores %d/4%s"
                             % ({True: "MANOS CERCA", False: "zona libre", None: "no veo los marcadores"}[cerca],
                                len(cajas), len(en_zona), vistos, "" if placa is None else "  | " + placa.texto()))
            sys.stdout.flush()
            if args.ver:
                if zona_img is not None:
                    cv2.polylines(bgr, [zona_img.astype(np.int32)], True, (0, 0, 255) if cerca else (0, 200, 0), 2)
                for (x1, y1, x2, y2), conf in cajas:
                    cv2.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
                    cv2.putText(bgr, "%.2f" % conf, (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 200, 255), 2)
                cv2.imshow("manos (q para salir)", bgr)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            time.sleep(max(0.0, 1.0 / args.hz - (time.time() - t_tick)))
    except KeyboardInterrupt:
        pass
    finally:
        camara.release()
        cv2.destroyAllWindows()
        print()


if __name__ == "__main__":
    main()
