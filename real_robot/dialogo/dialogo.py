#!/usr/bin/env python3
"""Voz del robot: habla en inglés (Piper) y escucha palabras clave (Vosk). Se ejecuta en el host.

    python3 dialogo.py                     # micrófono y altavoz por defecto (PipeWire)
    python3 dialogo.py --teclado --mudo    # pruebas: escribes las palabras y el robot "habla" en texto

Modelos: python3 descargar_modelos.py (una vez).
Escucha sólo estas palabras: stop, wait, pause, yes, yeah, no, slower, slow down, continue,
ready, go on. "stop/wait/pause" se atienden en cuanto se oyen (resultado parcial); el resto
con el resultado final, que es más fiable.
Habla con el supervisor por UDP: recibe su resumen en 5014 y le manda comandos a 5013:
    {"comando": "empezar" | "pausa" | "seguir" | "lento" | "hablando", "hasta": t}
Mientras el robot habla (y 0.5 s después) el micrófono se ignora: aquí no se reconocen
palabras y el supervisor descarta la emoción de la voz (no se oye a sí mismo).

"stop" por voz es un canal de seguridad ADICIONAL: el paro de emergencia sigue mandando.
"""
import argparse
import json
import os
import queue
import select
import socket
import sys
import threading
import time

import numpy as np

DIR_MODELOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelos")
IP = "127.0.0.1"
PUERTO_COMANDOS = 5013
PUERTO_DIALOGO = 5014
HZ = 20.0
SR = 16000
MARGEN_ECO = 0.5            # s tras hablar en que se sigue ignorando el micrófono
PAUSA_FRASES = 5.0          # s mínimos entre avisos no urgentes
SIN_SUPERVISOR = 1.0

PALABRAS = ["stop", "wait", "pause", "yes", "yeah", "no", "slower", "slow down", "continue",
            "ready", "go on", "[unk]"]
PARAR = {"stop", "wait", "pause"}
SI = {"yes", "yeah", "continue", "ready", "go on"}
NO = {"no"}
LENTO = {"slower", "slow down"}

# Lo que dice el robot (evitando las palabras clave en lo posible)
FRASES = {
    "bienvenida": "Welcome! I'm your soldering assistant. Shall we begin?",
    "empezamos": "Great, let's start. Say stop at any time.",
    "aun_no": "No problem. Say ready when you want to start.",
    "parado": "Stopping. Sorry if I was going too fast.",
    "seguir?": "Shall I continue?",
    "sigo_lento": "Okay, I'll continue more slowly.",
    "espero": "Okay, I'll wait. Say ready when you want to continue.",
    "lento": "Sure, I'll slow down. Don't worry.",
    "tension": "I noticed some tension, so I'm slowing down. Take your time.",
    "sin_percepcion": "I can't see or hear you well, so I'm going slower.",
    "estres_alto": "I've paused for a moment. I'll continue when you feel calmer.",
    "retomo_lento": "Okay, continuing slowly.",
    "normal": "You seem relaxed again. Back to normal speed.",
    "fin": "All done. Thank you for working with me!",
}


class Voz(object):
    """Piper en un hilo: decir() no bloquea. callback_hablando(t_fin) avisa de cada frase."""

    def __init__(self, modelo, mudo, callback_hablando):
        self.mudo, self.callback_hablando = mudo, callback_hablando
        self.cola = queue.Queue()
        self.hablando_hasta = 0.0
        if not mudo:
            from piper import PiperVoice
            import sounddevice as sd
            self.sd = sd
            self.piper = PiperVoice.load(modelo)
        threading.Thread(target=self._bucle, daemon=True).start()

    def decir(self, clave, urgente=False):
        if urgente:                         # cortar lo que esté diciendo
            while not self.cola.empty():
                self.cola.get_nowait()
            if not self.mudo:
                self.sd.stop()
        self.cola.put(FRASES[clave])

    def ocupado(self):
        return not self.cola.empty() or time.time() < self.hablando_hasta

    def _bucle(self):
        while True:
            texto = self.cola.get()
            if self.mudo:
                print("\r\033[K[robot dice] %s" % texto)
                self.hablando_hasta = time.time() + 0.06 * len(texto)
                time.sleep(0.06 * len(texto))
                continue
            trozos = list(self.piper.synthesize(texto))
            audio = np.concatenate([t.audio_float_array for t in trozos])
            sr = trozos[0].sample_rate
            self.hablando_hasta = time.time() + len(audio) / float(sr)
            self.callback_hablando(self.hablando_hasta)
            print("\r\033[K[robot dice] %s" % texto)
            self.sd.play(audio, sr)
            self.sd.wait()


class Oido(object):
    """Vosk con gramática cerrada sobre el micrófono; pone las palabras oídas en una cola."""

    def __init__(self, dispositivo, voz):
        import sounddevice as sd
        from vosk import Model, KaldiRecognizer, SetLogLevel
        SetLogLevel(-1)
        self.voz = voz
        self.palabras = queue.Queue()
        self.rec = KaldiRecognizer(Model(os.path.join(DIR_MODELOS, "vosk-model-small-en-us-0.15")),
                                   SR, json.dumps(PALABRAS))
        self._ya_parado = False
        disp = int(dispositivo) if dispositivo and dispositivo.isdigit() else dispositivo
        self.flujo = sd.RawInputStream(device=disp, samplerate=SR, channels=1, dtype="int16",
                                       blocksize=1600, callback=self._audio)
        self.flujo.start()

    def _audio(self, datos, frames, tiempo, estado):
        if time.time() < self.voz.hablando_hasta + MARGEN_ECO:
            self.rec.Reset()                # no oírse a sí mismo
            self._ya_parado = False
            return
        if self.rec.AcceptWaveform(bytes(datos)):
            texto = json.loads(self.rec.Result()).get("text", "")
            if not (self._ya_parado and set(texto.split()) <= PARAR):
                self._poner(texto)
            self._ya_parado = False
        else:
            parcial = json.loads(self.rec.PartialResult()).get("partial", "")
            if not self._ya_parado and PARAR & set(parcial.split()):
                self._ya_parado = True      # stop urgente: no esperar al final
                self._poner("stop")

    def _poner(self, texto):
        texto = texto.replace("[unk]", "").strip()
        if texto:
            self.palabras.put(texto)


def clasificar(texto):
    palabras = set(texto.split())
    if "slow" in palabras or LENTO & palabras:
        return "lento"
    if PARAR & palabras:
        return "parar"
    if NO & palabras:
        return "no"
    if SI & palabras or "go" in palabras:
        return "si"
    return None


def main():
    p = argparse.ArgumentParser(description="Voz del robot: habla y escucha palabras clave")
    p.add_argument("--voz", default="en_US-lessac-medium", help="modelo Piper en dialogo/modelos/")
    p.add_argument("--dispositivo", help="micrófono: índice o parte del nombre (por defecto, el del sistema)")
    p.add_argument("--teclado", action="store_true", help="escribir las palabras en vez de hablar (pruebas)")
    p.add_argument("--mudo", action="store_true", help="mostrar las frases en vez de decirlas (pruebas)")
    args = p.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_resumen = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_resumen.bind((IP, PUERTO_DIALOGO))
    sock_resumen.setblocking(False)

    def comando(nombre, **extra):
        extra.update({"t": time.time(), "comando": nombre})
        sock.sendto(json.dumps(extra).encode("utf-8"), (IP, PUERTO_COMANDOS))

    voz = Voz(os.path.join(DIR_MODELOS, args.voz + ".onnx"), args.mudo,
              lambda hasta: comando("hablando", hasta=hasta))
    oido = None if args.teclado else Oido(args.dispositivo, voz)
    print("Diálogo listo (%s). Palabras: stop, wait, yes, no, slow down, continue, ready. Ctrl+C para salir."
          % ("teclado" if args.teclado else "micrófono"))

    fase = "esperando_supervisor"     # -> espera_inicio -> trabajando <-> pregunta_seguir / espera_reanudar -> fin
    resumen, t_resumen, previo, t_ultimo_aviso = None, 0.0, None, 0.0
    pendiente = None                  # (aviso, estado al que se refiere): se dirá cuando toque
    try:
        while True:
            t_tick = time.time()
            while True:
                try:
                    resumen, t_resumen = json.loads(sock_resumen.recv(65536).decode("utf-8")), t_tick
                except (BlockingIOError, socket.error):
                    break
                except ValueError:
                    pass
            conectado = resumen is not None and t_tick - t_resumen <= SIN_SUPERVISOR

            # Palabras oídas (o escritas)
            oidas = []
            if args.teclado and select.select([sys.stdin], [], [], 0)[0]:
                oidas.append(sys.stdin.readline().strip().lower())
            while oido is not None and not oido.palabras.empty():
                oidas.append(oido.palabras.get_nowait())

            for texto in oidas:
                intencion = clasificar(texto)
                print("\r\033[K[oído] %r -> %s" % (texto, intencion or "nada"))
                if intencion is None or not conectado:
                    continue
                if intencion == "parar" and fase in ("trabajando", "pregunta_seguir", "espera_reanudar"):
                    comando("pausa")                    # primero parar, luego hablar
                    if fase == "trabajando":
                        voz.decir("parado", urgente=True)
                        voz.decir("seguir?")
                        fase = "pregunta_seguir"
                elif intencion == "lento" and fase == "trabajando":
                    comando("lento")
                    voz.decir("lento")
                    t_ultimo_aviso = t_tick
                elif intencion == "si" and fase == "espera_inicio":
                    comando("empezar")
                    voz.decir("empezamos")
                    fase = "trabajando"
                elif intencion == "no" and fase == "espera_inicio":
                    voz.decir("aun_no")
                elif intencion == "si" and fase in ("pregunta_seguir", "espera_reanudar"):
                    comando("seguir")
                    voz.decir("sigo_lento")
                    fase, t_ultimo_aviso = "trabajando", t_tick
                elif intencion == "no" and fase == "pregunta_seguir":
                    voz.decir("espero")
                    fase = "espera_reanudar"

            # Lo que hace el sistema por su cuenta
            if conectado and fase == "esperando_supervisor":
                voz.decir("bienvenida")
                fase = "espera_inicio"
            if conectado and fase == "trabajando":
                robot = resumen.get("robot") or {}
                actual = (resumen.get("estado"), resumen.get("causa"))
                if robot.get("done"):
                    voz.decir("fin")
                    fase = "fin"
                elif resumen.get("modo") == "AUTO" and previo is not None and actual != previo:
                    aviso = {("CAUTIOUS", "tension"): "tension", ("CAUTIOUS", "sin_percepcion"): "sin_percepcion",
                             ("PAUSED", "estres_alto"): "estres_alto"}.get(actual)
                    if actual == ("NORMAL", "calma") and previo[0] == "CAUTIOUS":
                        aviso = "normal"
                    elif actual == ("CAUTIOUS", "calma") and previo[0] == "PAUSED":
                        aviso = "retomo_lento"
                    if aviso:
                        pendiente = (aviso, actual)
                # Decir el aviso pendiente si sigue siendo cierto: PAUSED al momento, el resto
                # espaciado PAUSA_FRASES para no saturar al operador
                if pendiente and pendiente[1] != actual:
                    pendiente = None
                if pendiente and (actual[0] == "PAUSED" or
                                  (t_tick - t_ultimo_aviso >= PAUSA_FRASES and not voz.ocupado())):
                    voz.decir(pendiente[0], urgente=actual[0] == "PAUSED")
                    pendiente, t_ultimo_aviso = None, t_tick
                previo = actual
            elif conectado:
                previo = (resumen.get("estado"), resumen.get("causa"))

            time.sleep(max(0.0, 1.0 / HZ - (time.time() - t_tick)))
    except KeyboardInterrupt:
        print("\ndiálogo cerrado")


if __name__ == "__main__":
    main()
