#!/usr/bin/env python3
"""Estrés del operador a partir de la voz (micrófono) -> UDP al supervisor. Se ejecuta en el host.

    python3 voz.py                        # micrófono de la C930e (la cámara de las manos)
    python3 voz.py --dispositivo C920     # otro micrófono, por parte de su nombre
    python3 voz.py --dispositivo sistema  # el predeterminado de Ubuntu
    python3 voz.py --listar               # ver los micrófonos disponibles
    python3 voz.py --wav prueba.wav       # prueba con un archivo (sin micrófono)

CNN "scratch" del proyecto voice-emotion-recognition (Jair y Lucas, curso de Aprendizaje
para la robótica), reentrenada con RAVDESS en inglés (entrenar_ravdess.py), en la GPU.
Cada 0.1 s analiza los últimos 3 s (log-mel 64 x 301, igual que en el entrenamiento) y
envía a UDP 5010:
    {"t": .., "fuente": "audio", "estres": 0..1 o null, "confianza": 0..1,
     "etiqueta": "fearful", "probs": {..}, "voz": 0.8}
"voz" es la fracción de la ventana con voz. En silencio repite el último resultado con
"mantenido": true y su confianza bajando a 0 en 5 s: se sigue viendo, pero deja de contar. Los primeros 3 s mide el ruido de fondo: quédate en silencio.

Ojo: RAVDESS son actores leyendo dos frases en inglés en un estudio; con voz espontánea,
otro idioma y el ruido del laboratorio la exactitud será menor. Por eso se envía la confianza.
"""
import argparse
import collections
import json
import math
import os
import queue
import socket
import sys
import time

import numpy as np

import microfono

DIR_MODELOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelos")
IP = "127.0.0.1"
PUERTO_PERCEPCION = 5010
HZ = 10.0

# Mismos parámetros que el entrenamiento (EmotionDetection.ipynb / entrenar_ravdess.py)
SR, N_SAMPLES, WIN, HOP, N_FFT, N_MELS = 16000, 3 * 16000, 400, 160, 512, 64
# Cuánto aporta cada emoción al estrés: los mismos pesos que el rostro (FER-2013)
PESO_ESTRES = {"angry": 1.0, "disgust": 1.0, "fearful": 1.0, "sad": 0.6,
               "surprised": 0.5, "happy": 0.0, "neutral": 0.0}
ARRANQUE_S = 1.0            # s iniciales que se descartan (ceros o transitorio al abrir el micro)
CALIBRACION_S = 2.0         # s siguientes para medir el ruido de fondo
MARGEN_VOZ_DB = 6.0         # una trama es voz si supera el ruido de fondo en tantos dB
VOZ_MIN = 0.10              # fracción mínima de voz en la ventana (0.3 s de 3 s) para clasificar
RUIDO_ALTO_DB = -30.0       # un silencio por encima de esto es un micrófono muy ruidoso
RUIDO_MIN_DB = -100.0       # por debajo son ceros: no está llegando audio
MANTENER_S = 5.0            # s de silencio en que el último resultado pierde su peso
VENTANA_RUIDO_S = 15.0      # el ruido de fondo se reestima con los últimos segundos


def conv_block(cin, cout):
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


def cargar_red(ruta, dispositivo):
    """ScratchCNN del notebook (arquitectura idéntica) con el checkpoint de RAVDESS."""
    import torch
    import torch.nn as nn

    class ScratchCNN(nn.Module):
        def __init__(self, n_classes):
            super().__init__()
            self.features = nn.Sequential(conv_block(1, 32), conv_block(32, 64),
                                          conv_block(64, 128), conv_block(128, 256))
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                      nn.Dropout(0.3), nn.Linear(256, n_classes))

        def forward(self, x):
            return self.head(self.features(x))

    ck = torch.load(ruta, map_location="cpu", weights_only=False)
    red = ScratchCNN(len(ck["classes"]))
    red.load_state_dict(ck["model"])
    return red.to(dispositivo).eval(), ck["mu"], ck["sd"], ck["classes"]


def logmel(x):
    """Igual que wav_to_logmel del entrenamiento: pico a 1, 3 s centrales, log-mel 64 x 301."""
    import librosa
    x = x.astype(np.float32)
    x = x / (np.abs(x).max() + 1e-9)
    if len(x) < N_SAMPLES:
        x = np.pad(x, (0, N_SAMPLES - len(x)))
    else:
        s = (len(x) - N_SAMPLES) // 2
        x = x[s:s + N_SAMPLES]
    S = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=N_FFT, win_length=WIN,
                                       hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(S, ref=1.0).astype(np.float32)


def energia_db(x, trama=WIN):
    """Energía en dB de tramas de 25 ms."""
    n = len(x) // trama
    if n == 0:
        return np.array([-120.0])
    tramas = x[:n * trama].reshape(n, trama)
    return 10 * np.log10(np.mean(tramas ** 2, axis=1) + 1e-12)


class SueloRuido(object):
    """Ruido de fondo: percentil 10 de la energía reciente. Sigue al micrófono si su nivel cambia
    (al abrirlo suele estar más alto) y las pausas al hablar bastan para no confundirlo con voz."""

    def __init__(self):
        self.tramas = collections.deque(maxlen=int(VENTANA_RUIDO_S * SR / WIN))

    def agregar(self, x16k):
        self.tramas.extend(float(e) for e in energia_db(x16k - x16k.mean()) if e > RUIDO_MIN_DB)

    def valor(self):
        return float(np.percentile(self.tramas, 10)) if self.tramas else None


def estres_y_confianza(p, clases, fraccion_voz):
    estres = float(sum(PESO_ESTRES.get(c, 0.0) * float(pi) for c, pi in zip(clases, p)))
    entropia = -sum(float(pi) * math.log(float(pi)) for pi in p if pi > 1e-9)
    nitidez = max(0.0, 1.0 - entropia / math.log(len(clases)))
    # Menos voz en la ventana = menos fiable
    return min(1.0, max(0.0, estres)), nitidez * min(1.0, fraccion_voz / 0.5)


class Clasificador(object):
    def __init__(self, dispositivo):
        import torch
        self.torch = torch
        self.dispositivo = dispositivo
        self.red, self.mu, self.sd, self.clases = cargar_red(
            os.path.join(DIR_MODELOS, "voz_ravdess_scratch.pt"), dispositivo)
        self.probabilidades(np.zeros(N_SAMPLES, np.float32))      # calentar

    def probabilidades(self, x16k):
        m = (logmel(x16k) - self.mu) / self.sd
        t = self.torch.from_numpy(m)[None, None].to(self.dispositivo)
        with self.torch.no_grad():
            return self.torch.softmax(self.red(t), dim=1)[0].cpu().numpy()


def analizar(x16k, ruido_db, clf):
    """Mensaje para el supervisor a partir de los últimos 3 s a 16 kHz."""
    x16k = x16k - x16k.mean()       # sin componente continua (algunos micros la tienen al abrirlos)
    fraccion = float(np.mean(energia_db(x16k) > ruido_db + MARGEN_VOZ_DB))
    mensaje = {"t": time.time(), "fuente": "audio", "estres": None, "confianza": 0.0,
               "etiqueta": "", "probs": {}, "voz": round(fraccion, 2)}
    if fraccion >= VOZ_MIN:
        p = clf.probabilidades(x16k)
        mensaje["estres"], mensaje["confianza"] = estres_y_confianza(p, clf.clases, fraccion)
        mensaje["etiqueta"] = clf.clases[int(np.argmax(p))]
        mensaje["probs"] = {c: round(float(pi), 3) for c, pi in zip(clf.clases, p)}
    return mensaje


def mantener(mensaje, ultimo, t_ultimo, ahora):
    """En silencio, repite el último resultado con la confianza bajando hasta 0 en MANTENER_S."""
    resto = max(0.0, 1.0 - (ahora - t_ultimo) / MANTENER_S)
    mensaje.update(estres=ultimo["estres"], etiqueta=ultimo["etiqueta"], probs=ultimo["probs"],
                   confianza=ultimo["confianza"] * resto, mantenido=True)
    return mensaje


def imprimir(mensaje):
    if mensaje["estres"] is None:
        sys.stdout.write("\r\033[Ksilencio (voz %.0f%%)" % (100 * mensaje["voz"]))
    elif mensaje.get("mantenido"):
        sys.stdout.write("\r\033[Ksilencio; último: %-9s estres %.2f  confianza %.2f"
                         % (mensaje["etiqueta"], mensaje["estres"], mensaje["confianza"]))
    else:
        sys.stdout.write("\r\033[K%-9s estres %.2f  confianza %.2f  (voz %.0f%%)"
                         % (mensaje["etiqueta"], mensaje["estres"], mensaje["confianza"], 100 * mensaje["voz"]))
    sys.stdout.flush()


def main():
    p = argparse.ArgumentParser(description="Estrés por la voz -> UDP al supervisor")
    p.add_argument("--dispositivo", default=microfono.MICROFONO,
                   help="micrófono: parte de su nombre, o 'sistema' (por defecto %s)" % microfono.MICROFONO)
    p.add_argument("--listar", action="store_true", help="listar micrófonos y salir")
    p.add_argument("--wav", help="analizar un archivo en vez del micrófono (pruebas)")
    p.add_argument("--hz", type=float, default=HZ)
    p.add_argument("--cpu", action="store_true", help="usar la CPU aunque haya GPU")
    args = p.parse_args()

    if args.listar:
        microfono.listar()
        return
    import sounddevice as sd
    import librosa
    import torch
    if args.cpu:
        dispositivo = "cpu"
    elif torch.cuda.is_available():
        dispositivo = "cuda"
    else:
        sys.exit("[error] PyTorch no ve la GPU (CUDA). Revisa el driver de NVIDIA o usa --cpu.")
    clf = Clasificador(dispositivo)
    print("CNN de voz (RAVDESS, %d clases) en %s"
          % (len(clf.clases), torch.cuda.get_device_name(0) if dispositivo == "cuda" else "CPU"))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    if args.wav:
        x, _ = librosa.load(args.wav, sr=SR, mono=True)
        ruido = float(np.percentile(energia_db(x), 10))
        mensaje = analizar(x, ruido, clf)
        sock.sendto(json.dumps(mensaje).encode("utf-8"), (IP, PUERTO_PERCEPCION))
        imprimir(mensaje)
        print("\nprobabilidades: %s" % mensaje["probs"])
        return

    mic = microfono.Microfono(args.dispositivo)
    sr_mic = int(sd.query_devices(mic.dispositivo, "input")["default_samplerate"])
    bufer = np.zeros(int(3 * sr_mic), np.float32)          # últimos 3 s a la frecuencia del micro
    bloques = queue.Queue()                                # el callback no pierde audio
    perdidas = []

    def recibir(datos, frames, tiempo, estado):
        if estado.input_overflow:
            perdidas.append(frames)
        bloques.put(datos[:, 0].copy())

    print("Micrófono en uso: %s (%d Hz)" % (mic.descripcion, sr_mic))
    print("Silencio %.0f s para medir el ruido de fondo..." % (ARRANQUE_S + CALIBRACION_S))
    t_inicio = time.time()
    suelo, calibrado = SueloRuido(), False
    ultimo, t_ultimo = None, 0.0

    with mic.abrir(lambda: sd.InputStream(device=mic.dispositivo, channels=1, samplerate=sr_mic,
                                          dtype="float32", callback=recibir)):
        print("Enviando a %s:%d a %.0f Hz. Ctrl+C para salir." % (IP, PUERTO_PERCEPCION, args.hz))
        try:
            while True:
                t_tick = time.time()
                nuevos = [bloques.get_nowait() for _ in range(bloques.qsize())]
                if nuevos:
                    nuevos = np.concatenate(nuevos)
                    bufer = np.concatenate([bufer, nuevos])[-len(bufer):]
                x16k = librosa.resample(bufer, orig_sr=sr_mic, target_sr=SR)

                # El primer segundo se descarta: al abrir un micrófono llegan ceros o un transitorio
                if len(nuevos) and t_tick - t_inicio >= ARRANQUE_S:
                    suelo.agregar(librosa.resample(nuevos, orig_sr=sr_mic, target_sr=SR))

                if not calibrado:
                    if t_tick - t_inicio >= ARRANQUE_S + CALIBRACION_S:
                        if suelo.valor() is None:
                            sys.exit("[error] no llega audio del micrófono. Elige otro en Configuración -> Sonido "
                                     "(o con --dispositivo, ver --listar).")
                        calibrado = True
                        print("ruido de fondo: %.1f dB. Ya puedes hablar." % suelo.valor())
                        if suelo.valor() > RUIDO_ALTO_DB:
                            print("[aviso] micrófono muy ruidoso: la voz se distingue peor. Mejor uno más cercano.")
                    mensaje = {"t": time.time(), "fuente": "audio", "estres": None, "confianza": 0.0,
                               "etiqueta": "", "probs": {}, "voz": 0.0}
                else:
                    mensaje = analizar(x16k, suelo.valor(), clf)
                    if mensaje["estres"] is not None:
                        ultimo, t_ultimo = mensaje, t_tick
                    elif ultimo is not None:
                        mensaje = mantener(mensaje, ultimo, t_ultimo, t_tick)
                    imprimir(mensaje)
                if perdidas:
                    print("\n[aviso] se perdió audio (%d muestras): laptop muy cargada" % sum(perdidas))
                    del perdidas[:]
                sock.sendto(json.dumps(mensaje).encode("utf-8"), (IP, PUERTO_PERCEPCION))
                time.sleep(max(0.0, 1.0 / args.hz - (time.time() - t_tick)))
        except KeyboardInterrupt:
            print("\nvoz cerrada")


if __name__ == "__main__":
    main()
