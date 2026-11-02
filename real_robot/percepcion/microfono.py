"""Elige el micrófono por nombre con PipeWire, sin cambiar el predeterminado del sistema.

Abrirlo a través de PipeWire permite que voz.py y dialogo.py lo compartan a la vez.
"""
import json
import os
import subprocess
import sys

MICROFONO = "C930e"         # por defecto: la cámara de las manos, la más cercana a la persona
SISTEMA = "sistema"         # --dispositivo sistema: el predeterminado de Ubuntu


def _fuentes():
    """(lista de (nombre de nodo, descripción), nombre del nodo predeterminado) según PipeWire."""
    datos = json.loads(subprocess.check_output(["pw-dump"]))
    fuentes, predeterminada = [], None
    for objeto in datos:
        if objeto.get("type") == "PipeWire:Interface:Node":
            props = objeto["info"]["props"]
            if props.get("media.class") == "Audio/Source":
                fuentes.append((props["node.name"], props.get("node.description", props["node.name"])))
        elif objeto.get("type") == "PipeWire:Interface:Metadata":
            for entrada in objeto.get("metadata") or []:
                if entrada.get("key") == "default.audio.source":
                    predeterminada = entrada["value"]["name"]
    return fuentes, predeterminada


class Microfono(object):
    """dispositivo: lo que se pasa a sounddevice. descripcion: el micrófono que se usa de verdad."""

    def __init__(self, nombre=MICROFONO):
        self.dispositivo, self.nodo = None, None
        try:
            fuentes, predeterminada = _fuentes()
        except (OSError, ValueError, subprocess.CalledProcessError):
            self.descripcion = "predeterminado del sistema (PipeWire no responde)"
            return
        if nombre != SISTEMA:
            for nodo, descripcion in fuentes:
                if nombre.lower() in descripcion.lower() or nombre.lower() in nodo.lower():
                    self.dispositivo, self.nodo, self.descripcion = "pipewire", nodo, descripcion
                    return
            disponibles = ", ".join(d for _, d in fuentes) or "ninguno"
            if nombre != MICROFONO:
                sys.exit("[error] no encuentro el micrófono '%s'. Disponibles: %s." % (nombre, disponibles))
            print("[aviso] no está el micrófono %s. Disponibles: %s. Uso el del sistema." % (nombre, disponibles))
        nombres = dict(fuentes)
        self.descripcion = "%s (predeterminado del sistema)" % nombres.get(predeterminada, "desconocido")

    def abrir(self, crear_flujo):
        """Crea el flujo de entrada dirigido a este micrófono. PIPEWIRE_NODE sólo está activa al
        abrirlo: si no, también dirigiría la salida de audio (la voz del robot) a este nodo."""
        if self.nodo is None:
            return crear_flujo()
        os.environ["PIPEWIRE_NODE"] = self.nodo
        try:
            return crear_flujo()
        finally:
            del os.environ["PIPEWIRE_NODE"]


def listar():
    fuentes, predeterminada = _fuentes()
    for nodo, descripcion in fuentes:
        print("%s %s" % ("*" if nodo == predeterminada else " ", descripcion))
