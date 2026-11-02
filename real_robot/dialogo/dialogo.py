#!/usr/bin/env python3
"""Voz del robot: habla en inglés (Piper) y escucha palabras clave (Vosk). Se ejecuta en el host.

    python3 dialogo.py                     # micrófono de la C930e y altavoz del sistema
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "percepcion"))
import microfono

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
    # Cambios manuales desde el teclado del supervisor
    "manual_normal": "Okay, normal speed.",
    "manual_lento": "Okay, going slower.",
    "manual_pausa": "Okay, pausing.",
    "manual_retirar": "Okay, moving away from the board.",
}


# Narración de la tarea. Comprobadas contra Vosk, con repeticiones porque Piper varía, para que no se oigan
# como "stop". Evitar "start" y "soldering", que sí se confunden a veces.
NUMEROS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
           "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
           "twenty one", "twenty two", "twenty three", "twenty four", "twenty five"]
FRASES_CICLO = {"aproximacion": "Approaching the board.", "soldadura": "Working on the board.",
                "retraccion": "Moving away from the board.", "home": "Returning to home."}


class Voz(object):
    """Piper en un hilo: decir() no bloquea. callback_hablando(t_fin) avisa de cada frase."""

    def __init__(self, modelo, mudo, callback_hablando):
        self.mudo, self.callback_hablando = mudo, callback_hablando
        self.cola = queue.Queue()
        self.hablando_hasta = 0.0
        self.narrando = False               # la frase actual es narración: se sigue oyendo "stop"
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
        self.cola.put((FRASES[clave], False))

    def narrar(self, texto):
        self.cola.put((texto, True))

    def ocupado(self):
        return not self.cola.empty() or time.time() < self.hablando_hasta

    def _bucle(self):
        while True:
            texto, self.narrando = self.cola.get()
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
        mic = microfono.Microfono(dispositivo)
        print("Micrófono en uso: %s" % mic.descripcion)
        self.flujo = mic.abrir(lambda: sd.RawInputStream(device=mic.dispositivo, samplerate=SR, channels=1,
                                                         dtype="int16", blocksize=1600, callback=self._audio))
        self.flujo.start()

    def _audio(self, datos, frames, tiempo, estado):
        hablando = time.time() < self.voz.hablando_hasta + MARGEN_ECO
        if hablando and not self.voz.narrando:
            self.rec.Reset()                # no oírse a sí mismo
            self._ya_parado = False
            return
        # Mientras narra sólo vale "stop": sus frases se eligieron para no confundirse con ella, y si no
        # el operador no podría pararlo durante gran parte de la tarea
        parar = {"stop"} if hablando else PARAR
        if self.rec.AcceptWaveform(bytes(datos)):
            palabras = set(json.loads(self.rec.Result()).get("text", "").split())
            if hablando:
                if "stop" in palabras and not self._ya_parado:
                    self._poner("stop")
            elif not (self._ya_parado and palabras <= PARAR):
                self._poner(" ".join(sorted(palabras)))
            self._ya_parado = False
        else:
            parcial = json.loads(self.rec.PartialResult()).get("partial", "")
            if not self._ya_parado and parar & set(parcial.split()):
                self._ya_parado = True      # stop urgente: no esperar al final
                self._poner("stop")

    def _poner(self, texto):
        if texto.replace("[unk]", "").strip():      # [unk] se conserva: clasificar() lo necesita
            self.palabras.put(texto)


def narracion(robot):
    """(id, frase) de la etapa actual de la tarea, o None si no hay nada que anunciar."""
    segmento, accion = robot.get("segment") or "", robot.get("action") or ""
    if "_p" in segmento:
        punto = int(segmento.split("_p")[-1])
        total = max(1, ((robot.get("n_segments") or 0) - 2) // 3)
        if accion == "soldando":
            return "soldar%d" % punto, "Working on point %s." % NUMEROS[punto - 1]
        if segmento.startswith("sobre") and accion == "segmento":
            return "ir%d" % punto, "Moving to point %s of %s." % (NUMEROS[punto - 1], NUMEROS[total - 1])
    elif accion == "segmento" and segmento in FRASES_CICLO:
        return segmento, FRASES_CICLO[segmento]
    return None


def clasificar(texto):
    palabras = set(texto.split())
    if "slow" in palabras or LENTO & palabras:
        return "lento"
    if PARAR & palabras:
        return "parar"
    # "yes" pone el robot en marcha: sólo vale la palabra sola, no una frase cualquiera encajada a la fuerza
    if "[unk]" in palabras:
        return None
    if NO & palabras:
        return "no"
    if SI & palabras or "go" in palabras:
        return "si"
    return None


def main():
    p = argparse.ArgumentParser(description="Voz del robot: habla y escucha palabras clave")
    p.add_argument("--voz", default="en_US-lessac-medium", help="modelo Piper en dialogo/modelos/")
    p.add_argument("--dispositivo", default=microfono.MICROFONO,
                   help="micrófono: parte de su nombre, o 'sistema' (por defecto %s)" % microfono.MICROFONO)
    p.add_argument("--sin-narracion", action="store_true", help="no anunciar cada punto de la tarea")
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
    t_orden_voz = 0.0                 # el resumen tarda en reflejar una orden de voz recién enviada
    ultima_etapa = None               # última etapa de la tarea anunciada
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
                t_orden_voz = t_tick
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
            # Si arranca o sigue desde el teclado (a, n, c, r), el diálogo no se queda esperando un "yes"
            if conectado and fase in ("espera_inicio", "pregunta_seguir", "espera_reanudar") \
                    and t_tick - t_orden_voz > 1.0 \
                    and (resumen.get("modo") == "AUTO" or resumen.get("estado") != "PAUSED"):
                fase = "trabajando"
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
                elif resumen.get("modo") == "MANUAL" and previo is not None and actual != previo \
                        and t_tick - t_orden_voz > 1.0:
                    aviso = {"NORMAL": "manual_normal", "CAUTIOUS": "manual_lento", "PAUSED": "manual_pausa",
                             "RETRACT": "manual_retirar"}.get(actual[0])
                    if aviso:
                        pendiente = (aviso, actual)
                # Decir el aviso pendiente si sigue siendo cierto: las paradas y lo que pide el teclado al
                # momento, el resto espaciado PAUSA_FRASES para no saturar al operador
                if pendiente and pendiente[1] != actual:
                    pendiente = None
                urgente = actual[0] in ("PAUSED", "RETRACT") or actual[1] == "manual"
                if pendiente and (urgente or (t_tick - t_ultimo_aviso >= PAUSA_FRASES and not voz.ocupado())):
                    voz.decir(pendiente[0], urgente=urgente)
                    pendiente, t_ultimo_aviso = None, t_tick
                # Narración: sólo si no hay nada más urgente. Si está hablando, espera y anuncia la
                # etapa que toque entonces, sin acumular frases atrasadas
                if not args.sin_narracion and not pendiente and actual[0] in ("NORMAL", "CAUTIOUS") \
                        and not voz.ocupado():
                    etapa = narracion(robot)
                    if etapa and etapa[0] != ultima_etapa:
                        voz.narrar(etapa[1])
                        ultima_etapa = etapa[0]
                previo = actual
            elif conectado:
                previo = (resumen.get("estado"), resumen.get("causa"))

            time.sleep(max(0.0, 1.0 / HZ - (time.time() - t_tick)))
    except KeyboardInterrupt:
        print("\ndiálogo cerrado")


if __name__ == "__main__":
    main()
