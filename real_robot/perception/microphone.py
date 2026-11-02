"""Picks the microphone by name with PipeWire, without changing the system default.

Opening it through PipeWire lets voice.py and dialogue.py share it at the same time.
"""
import json
import os
import subprocess
import sys

MICROPHONE = "C930e"        # default: the hands camera, the closest one to the person
SYSTEM = "system"           # --device system: Ubuntu's default one


def _sources():
    """(list of (node name, description), name of the default node) according to PipeWire."""
    data = json.loads(subprocess.check_output(["pw-dump"]))
    sources, default = [], None
    for obj in data:
        if obj.get("type") == "PipeWire:Interface:Node":
            props = obj["info"]["props"]
            if props.get("media.class") == "Audio/Source":
                sources.append((props["node.name"], props.get("node.description", props["node.name"])))
        elif obj.get("type") == "PipeWire:Interface:Metadata":
            for entry in obj.get("metadata") or []:
                if entry.get("key") == "default.audio.source":
                    default = entry["value"]["name"]
    return sources, default


class Microphone(object):
    """device: what is passed to sounddevice. description: the microphone that is really used."""

    def __init__(self, name=MICROPHONE):
        self.device, self.node = None, None
        try:
            sources, default = _sources()
        except (OSError, ValueError, subprocess.CalledProcessError):
            self.description = "system default (PipeWire does not answer)"
            return
        if name != SYSTEM:
            for node, description in sources:
                if name.lower() in description.lower() or name.lower() in node.lower():
                    self.device, self.node, self.description = "pipewire", node, description
                    return
            available = ", ".join(d for _, d in sources) or "none"
            if name != MICROPHONE:
                sys.exit("[error] microphone '%s' not found. Available: %s." % (name, available))
            print("[warning] microphone %s is not there. Available: %s. Using the system one." % (name, available))
        names = dict(sources)
        self.description = "%s (system default)" % names.get(default, "unknown")

    def open(self, create_stream):
        """Creates the input stream aimed at this microphone. PIPEWIRE_NODE is only set while opening
        it: otherwise it would also send the audio output (the robot's voice) to this node."""
        if self.node is None:
            return create_stream()
        os.environ["PIPEWIRE_NODE"] = self.node
        try:
            return create_stream()
        finally:
            del os.environ["PIPEWIRE_NODE"]


def list_microphones():
    sources, default = _sources()
    for node, description in sources:
        print("%s %s" % ("*" if node == default else " ", description))
