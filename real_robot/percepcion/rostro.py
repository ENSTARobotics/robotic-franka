#!/usr/bin/env python3
"""Estrés del operador a partir de la cara (webcam) -> UDP al supervisor. Se ejecuta en el host.

    python3 rostro.py --ver                   # webcam C920, con ventana de vista previa
    python3 rostro.py --camara C930e --ver    # otra cámara, por nombre o por número de /dev/videoN
    python3 rostro.py --imagen foto.jpg       # prueba con una imagen fija (sin cámara)

Detector YuNet (OpenCV) + mini-Xception (FER-2013, 7 expresiones) del TD4 de ROB311 de
Joab ("face recognition": 61% de exactitud en PrivateTest), pasada a PyTorch y ejecutada
en la GPU con CUDA (--cpu para forzar la CPU).
Envía a UDP 5010 unas 10 veces por segundo:
    {"t": .., "fuente": "rostro", "estres": 0..1 o null, "confianza": 0..1,
     "etiqueta": "Fear", "probs": {"Angry": .., ...}, "cara": true}
Sin cara visible: estres null y confianza 0 (el supervisor la ignora en la fusión).
También envía la imagen anotada (JPEG) a UDP 5012 para pantalla.py.

Ojo: una expresión facial no es una medida fiable de la emoción interna; FER-2013 tiene
fotos de baja resolución y clases desbalanceadas. Por eso se envía también la confianza.
"""
import argparse
import glob
import json
import math
import os
import socket
import sys
import time

import cv2
import numpy as np

DIR_MODELOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelos")
IP = "127.0.0.1"
PUERTO_PERCEPCION = 5010
PUERTO_IMAGEN = 5012        # imagen anotada para pantalla.py
ANCHO_IMAGEN = 640
CAMARA = "C920"             # la C930e es para las manos
HZ = 10.0

ETIQUETAS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
# Cuánto aporta cada expresión al estrés (0 = nada, 1 = todo). Ajustable tras el piloto.
PESO_ESTRES = {"Angry": 1.0, "Disgust": 1.0, "Fear": 1.0, "Sad": 0.6,
               "Surprise": 0.5, "Happy": 0.0, "Neutral": 0.0}
UMBRAL_DETECCION = 0.75     # confianza mínima de YuNet (la misma del TD4)


class Detector(object):
    """YuNet; devuelve la cara más grande como (x, y, w, h) o None."""

    def __init__(self):
        self.yunet = cv2.FaceDetectorYN.create(
            os.path.join(DIR_MODELOS, "face_detection_yunet_2023mar.onnx"), "",
            (320, 320), UMBRAL_DETECCION, 0.3, 5000)

    def cara_mayor(self, bgr):
        alto, ancho = bgr.shape[:2]
        escala = min(1.0, 640.0 / max(alto, ancho))
        peque = cv2.resize(bgr, None, fx=escala, fy=escala) if escala < 1 else bgr
        self.yunet.setInputSize((peque.shape[1], peque.shape[0]))
        _, caras = self.yunet.detect(peque)
        if caras is None or not len(caras):
            return None
        x, y, w, h = max(caras[:, :4], key=lambda c: c[2] * c[3]) / escala
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(ancho, int(x + w)), min(alto, int(y + h))
        return (x0, y0, x1 - x0, y1 - y0) if x1 - x0 >= 20 and y1 - y0 >= 20 else None


class Expresion(object):
    """mini-Xception 48x48 en PyTorch (mini_xception.py), en la GPU si hay CUDA."""

    def __init__(self, dispositivo):
        import torch
        from mini_xception import cargar
        self.torch = torch
        self.dispositivo = dispositivo
        self.red = cargar(os.path.join(DIR_MODELOS, "mini_xception_48.pt"), dispositivo)
        self.probabilidades(np.zeros((48, 48, 3), np.uint8))     # calentar

    def probabilidades(self, recorte_bgr):
        gris = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2GRAY)
        gris = cv2.resize(gris, (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
        x = self.torch.from_numpy(gris / 127.5 - 1.0)[None, None].to(self.dispositivo)
        with self.torch.no_grad():
            return self.red(x)[0].cpu().numpy()


def estres_y_confianza(p):
    """Estrés = suma ponderada de expresiones negativas; confianza = 1 - entropía normalizada."""
    estres = float(sum(PESO_ESTRES[e] * float(pi) for e, pi in zip(ETIQUETAS, p)))
    entropia = -sum(float(pi) * math.log(float(pi)) for pi in p if pi > 1e-9)
    return min(1.0, max(0.0, estres)), max(0.0, 1.0 - entropia / math.log(len(ETIQUETAS)))


def buscar_camara(camara):
    """Número de /dev/videoN de la cámara: por número, o por parte de su nombre (no cambia al
    reconectarla). Cada webcam crea varios nodos; el de imagen es el de index 0."""
    if camara.isdigit():
        return int(camara)
    disponibles = {}
    for ruta in glob.glob("/sys/class/video4linux/video*"):
        with open(os.path.join(ruta, "index")) as f_index, open(os.path.join(ruta, "name")) as f_name:
            if f_index.read().strip() == "0":
                disponibles[f_name.read().strip()] = int(os.path.basename(ruta)[5:])
    for nombre, numero in sorted(disponibles.items()):
        if camara.lower() in nombre.lower():
            return numero
    sys.exit("[error] no encuentro la cámara '%s'. Disponibles: %s. Usa --camara <nombre o número>."
             % (camara, ", ".join("%s (%d)" % d for d in sorted(disponibles.items())) or "ninguna"))


def anotar(bgr, caja, mensaje):
    if caja is not None:
        x, y, w, h = caja
        color = (0, 0, 255) if mensaje["estres"] > 0.5 else (0, 200, 0)
        cv2.rectangle(bgr, (x, y), (x + w, y + h), color, 2)
        cv2.putText(bgr, "%s  estres %.2f  conf %.2f" % (mensaje["etiqueta"], mensaje["estres"],
                    mensaje["confianza"]), (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        cv2.putText(bgr, "sin cara", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return bgr


def enviar_imagen(sock, bgr):
    """Imagen anotada para pantalla.py (JPEG de 640 px de ancho, cabe en un datagrama UDP)."""
    escala = ANCHO_IMAGEN / float(bgr.shape[1])
    peque = cv2.resize(bgr, None, fx=escala, fy=escala) if escala < 1 else bgr
    for calidad in (70, 50, 30):
        ok, jpg = cv2.imencode(".jpg", peque, [cv2.IMWRITE_JPEG_QUALITY, calidad])
        if ok and len(jpg) < 60000:
            sock.sendto(jpg.tobytes(), (IP, PUERTO_IMAGEN))
            return


def main():
    p = argparse.ArgumentParser(description="Estrés por expresión facial -> UDP al supervisor")
    p.add_argument("--camara", default=CAMARA, help="nombre o número de la webcam (por defecto %s)" % CAMARA)
    p.add_argument("--imagen", help="usar una imagen fija en vez de la cámara (pruebas)")
    p.add_argument("--ver", action="store_true", help="ventana con la cara y la expresión")
    p.add_argument("--hz", type=float, default=HZ)
    p.add_argument("--cpu", action="store_true", help="usar la CPU aunque haya GPU")
    args = p.parse_args()

    import torch
    if args.cpu:
        dispositivo = "cpu"
    elif torch.cuda.is_available():
        dispositivo = "cuda"
    else:
        sys.exit("[error] PyTorch no ve la GPU (CUDA). Revisa el driver de NVIDIA o usa --cpu.")
    detector = Detector()
    red = Expresion(dispositivo)
    print("mini-Xception en %s" % (torch.cuda.get_device_name(0) if dispositivo == "cuda" else "CPU"))
    if args.imagen:
        fija = cv2.imread(args.imagen)
        if fija is None:
            sys.exit("[error] no puedo leer %s" % args.imagen)
        camara = None
    else:
        numero = buscar_camara(args.camara)
        camara = cv2.VideoCapture(numero, cv2.CAP_V4L2)
        if not camara.isOpened():
            sys.exit("[error] no puedo abrir /dev/video%d: ¿la está usando otro programa?" % numero)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("Enviando a %s:%d a %.0f Hz. Ctrl+C para salir." % (IP, PUERTO_PERCEPCION, args.hz))

    periodo, t_impresion = 1.0 / args.hz, 0.0
    try:
        while True:
            t_tick = time.time()
            if camara is not None:
                ok, bgr = camara.read()
                if not ok:
                    sys.exit("[error] la cámara dejó de enviar imágenes")
            else:
                bgr = fija.copy()

            caja = detector.cara_mayor(bgr)
            mensaje = {"t": time.time(), "fuente": "rostro", "estres": None, "confianza": 0.0,
                       "etiqueta": "", "probs": {}, "cara": caja is not None}
            if caja is not None:
                x, y, w, h = caja
                probs = red.probabilidades(bgr[y:y + h, x:x + w])
                mensaje["estres"], mensaje["confianza"] = estres_y_confianza(probs)
                mensaje["etiqueta"] = ETIQUETAS[int(np.argmax(probs))]
                mensaje["probs"] = {e: round(float(pi), 3) for e, pi in zip(ETIQUETAS, probs)}
            sock.sendto(json.dumps(mensaje).encode("utf-8"), (IP, PUERTO_PERCEPCION))

            if t_tick - t_impresion >= 0.5:
                t_impresion = t_tick
                if caja is None:
                    sys.stdout.write("\r\033[Ksin cara")
                else:
                    sys.stdout.write("\r\033[K%-8s estres %.2f  confianza %.2f"
                                     % (mensaje["etiqueta"], mensaje["estres"], mensaje["confianza"]))
                sys.stdout.flush()
            anotada = anotar(bgr, caja, mensaje)
            enviar_imagen(sock, anotada)
            if args.ver:
                cv2.imshow("rostro (q para salir)", anotada)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            time.sleep(max(0.0, periodo - (time.time() - t_tick)))
    except KeyboardInterrupt:
        pass
    finally:
        if camara is not None:
            camara.release()
        cv2.destroyAllWindows()
        print("\nrostro cerrado")


if __name__ == "__main__":
    main()
