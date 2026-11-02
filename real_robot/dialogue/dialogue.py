#!/usr/bin/env python3
"""The robot's voice: speaks English (Piper) and listens for keywords (Vosk). Runs on the host.

    python3 dialogue.py                     # C930e microphone and the system speaker
    python3 dialogue.py --keyboard --mute   # tests: you type the words and the robot "speaks" in text

Models: python3 download_models.py (once).
It only listens for these words: stop, wait, pause, yes, yeah, no, slower, slow down, continue,
ready, go on. "stop/wait/pause" are acted on as soon as they are heard (partial result); the rest
with the final result, which is more reliable.
Talks to the supervisor over UDP: receives its summary on 5014 and sends it commands on 5013:
    {"command": "start" | "pause" | "resume" | "slow" | "speaking", "until": t}
While the robot speaks (and 0.5 s after) the microphone is ignored: no words are recognized here
and the supervisor throws away the voice emotion (it does not hear itself).
Hands next to the PCB (the supervisor marks it in its summary): says that it stops and moves away,
asks to move them away, and when the zone is clear waits for "continue" (or "yes", "ready") to go on.

"stop" by voice is an ADDITIONAL safety channel: the emergency stop is still the main one.
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

# Numerical libraries use every core by default and saturate the CPU: without an RT kernel that
# causes communication_constraints_violation on the robot. It must come before importing numpy.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "perception"))
import microphone

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
IP = "127.0.0.1"
COMMANDS_PORT = 5013
DIALOGUE_PORT = 5014
PCB_PORT = 5015             # perception/pcb.py (or hands.py); if it is not open, the PCB is not checked
PCB_WAIT_S = 2.0            # s it waits for its first message before greeting without it
HZ = 20.0
SR = 16000
ECHO_MARGIN = 0.5           # s after speaking in which the microphone is still ignored
PHRASE_GAP = 5.0            # minimum s between non-urgent announcements
NO_SUPERVISOR = 1.0

WORDS = ["stop", "wait", "pause", "yes", "yeah", "no", "slower", "slow down", "continue",
         "ready", "go on", "[unk]"]
STOP = {"stop", "wait", "pause"}
YES = {"yes", "yeah", "continue", "ready", "go on"}
NO = {"no"}
SLOW = {"slower", "slow down"}

# What the robot says (avoiding the keywords as far as possible)
PHRASES = {
    "welcome": "Welcome! I'm your soldering assistant. Shall we begin?",
    "pcb_missing": "Welcome! Please place the PCB on the table.",
    "pcb_reminder": "Please place the PCB on the table first.",
    "pcb_lost": "I can't see the PCB. Please place it in the work area and move your hands away.",
    "pcb_ok": "PCB detected. Shall we begin?",
    "welcome_pcb": "Welcome! PCB detected. Shall we begin?",
    "starting": "Great, let's start. Say stop at any time.",
    "not_yet": "No problem. Say ready when you want to start.",
    "stopped": "Stopping. Sorry if I was going too fast.",
    "resume?": "Shall I continue?",
    "resume_slow": "Okay, I'll continue more slowly.",
    "waiting": "Okay, I'll wait. Say ready when you want to continue.",
    "slow": "Sure, I'll slow down. Don't worry.",
    "tension": "I noticed some tension, so I'm slowing down. Take your time.",
    "no_perception": "I can't see or hear you well, so I'm going slower.",
    "high_stress": "I've paused for a moment. I'll continue when you feel calmer.",
    "picking_up_slowly": "Okay, continuing slowly.",
    "normal": "You seem relaxed again. Back to normal speed.",
    "end": "All done. Thank you for working with me!",
    "calibrate": "Calibrating. Please look at the camera and talk normally for a moment.",
    "calibrated_ok": "Calibration done.",
    "calibrated_no_face": "Calibration done, but I could not see your face.",
    "calibrated_no_audio": "Calibration done, but I could not hear your voice.",
    "calibrated_no_data": "Calibration failed. I could not see or hear you.",
    "calibrated_cancelled": "Calibration cancelled.",
    "begin?": "Shall we begin?",
    # Manual changes from the supervisor's keyboard
    "manual_normal": "Okay, normal speed.",
    "manual_slow": "Okay, going slower.",
    "manual_pause": "Okay, pausing.",
    "manual_retract": "Okay, moving away from the board.",
    # Hands next to the PCB (perception/hands.py)
    "hands_near": "Careful, your hands are near the board. I'm pausing and moving away.",
    "hands_clear": "Thank you, the area is clear. Say continue when you are ready.",
    "hands_still": "Please move your hands away from the board first.",
    "hands_again": "Careful, your hands are near the board again.",
    "back_to_normal": "Back to normal speed.",
}


# Narration of the task. Checked against Vosk, with repetitions because Piper varies, so that they are not
# heard as "stop". Avoid "start" and "soldering", which are sometimes confused with it.
NUMBERS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
           "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
           "twenty one", "twenty two", "twenty three", "twenty four", "twenty five"]
CYCLE_PHRASES = {"approach": "Approaching the board.", "solder": "Working on the board.",
                 "retreat": "Moving away from the board.", "home": "Returning to home."}


class Voice(object):
    """Piper in a thread: say() does not block. speaking_callback(t_end) reports each phrase.

    Only this thread touches the audio output: it plays each phrase in 0.1 s blocks and an urgent say() just
    raises a flag that it checks between blocks. Stopping sounddevice from another thread (sd.stop() while
    sd.wait() runs here) hung the whole program: the robot went silent and stopped listening."""

    BLOCK_S = 0.1

    def __init__(self, model, mute, speaking_callback):
        self.mute, self.speaking_callback = mute, speaking_callback
        self.queue = queue.Queue()
        self.cut = threading.Event()        # set by an urgent say(): stop the phrase being played
        self.speaking_until = 0.0
        self.narrating = False              # the current phrase is narration: "stop" is still heard
        if not mute:
            from piper import PiperVoice
            import sounddevice as sd
            self.sd = sd
            self.piper = PiperVoice.load(model)
        threading.Thread(target=self._loop, daemon=True).start()

    def say(self, key, urgent=False):
        if urgent:                          # cut off whatever it is saying
            while not self.queue.empty():
                self.queue.get_nowait()
            self.cut.set()
        self.queue.put((PHRASES[key], False))

    def narrate(self, text):
        self.queue.put((text, True))

    def busy(self):
        return not self.queue.empty() or time.time() < self.speaking_until

    def _loop(self):
        while True:
            text, self.narrating = self.queue.get()
            self.cut.clear()
            if self.mute:
                print("\r\033[K[robot says] %s" % text)
                self.speaking_until = time.time() + 0.06 * len(text)
                self.cut.wait(0.06 * len(text))
                continue
            chunks = list(self.piper.synthesize(text))
            audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
            sr = chunks[0].sample_rate
            self.speaking_until = time.time() + len(audio) / float(sr)
            self.speaking_callback(self.speaking_until)
            print("\r\033[K[robot says] %s" % text)
            block = int(sr * self.BLOCK_S)
            out = self.sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
            out.start()
            try:
                for i in range(0, len(audio), block):
                    if self.cut.is_set():
                        break
                    out.write(audio[i:i + block].reshape(-1, 1))
            finally:
                if self.cut.is_set():
                    out.abort()             # drop what is buffered: the urgent phrase comes now
                    self.speaking_until = time.time()
                else:
                    out.stop()              # let the end of the phrase play
                out.close()


class Ear(object):
    """Vosk with a closed grammar on the microphone; puts the words it hears in a queue."""

    def __init__(self, device, voice):
        import sounddevice as sd
        from vosk import Model, KaldiRecognizer, SetLogLevel
        SetLogLevel(-1)
        self.voice = voice
        self.words = queue.Queue()
        self.rec = KaldiRecognizer(Model(os.path.join(MODELS_DIR, "vosk-model-small-en-us-0.15")),
                                   SR, json.dumps(WORDS))
        self._already_stopped = False
        mic = microphone.Microphone(device)
        print("Microphone in use: %s" % mic.description)
        self.stream = mic.open(lambda: sd.RawInputStream(device=mic.device, samplerate=SR, channels=1,
                                                         dtype="int16", blocksize=1600, callback=self._audio))
        self.stream.start()

    def _audio(self, data, frames, timestamp, status):
        speaking = time.time() < self.voice.speaking_until + ECHO_MARGIN
        if speaking and not self.voice.narrating:
            self.rec.Reset()                # do not hear itself
            self._already_stopped = False
            return
        # While narrating only "stop" counts: its phrases were chosen so as not to be confused with it, and
        # otherwise the operator could not stop it for a large part of the task
        stop = {"stop"} if speaking else STOP
        if self.rec.AcceptWaveform(bytes(data)):
            words = set(json.loads(self.rec.Result()).get("text", "").split())
            if speaking:
                if "stop" in words and not self._already_stopped:
                    self._put("stop")
            elif not (self._already_stopped and words <= STOP):
                self._put(" ".join(sorted(words)))
            self._already_stopped = False
        else:
            partial = json.loads(self.rec.PartialResult()).get("partial", "")
            if not self._already_stopped and stop & set(partial.split()):
                self._already_stopped = True    # urgent stop: do not wait for the end
                self._put("stop")

    def _put(self, text):
        if text.replace("[unk]", "").strip():       # [unk] is kept: classify() needs it
            self.words.put(text)


def narration(robot):
    """(id, phrase) of the current stage of the task, or None if there is nothing to announce."""
    segment, action = robot.get("segment") or "", robot.get("action") or ""
    if "_p" in segment:
        point = int(segment.split("_p")[-1])
        total = max(1, ((robot.get("n_segments") or 0) - 2) // 3)
        if action == "soldering":
            return "solder%d" % point, "Working on point %s." % NUMBERS[point - 1]
        if segment.startswith("above") and action == "segment":
            return "go%d" % point, "Moving to point %s of %s." % (NUMBERS[point - 1], NUMBERS[total - 1])
    elif action == "segment" and segment in CYCLE_PHRASES:
        return segment, CYCLE_PHRASES[segment]
    return None


def classify(text):
    words = set(text.split())
    if "slow" in words or SLOW & words:
        return "slow"
    if STOP & words:
        return "stop"
    # "yes" starts the robot: only the word on its own counts, not any phrase forced into the grammar
    if "[unk]" in words:
        return None
    if NO & words:
        return "no"
    if YES & words or "go" in words:
        return "yes"
    return None


def main():
    p = argparse.ArgumentParser(description="The robot's voice: speaks and listens for keywords")
    p.add_argument("--voice", default="en_US-lessac-medium", help="Piper model in dialogue/models/")
    p.add_argument("--device", default=microphone.MICROPHONE,
                   help="microphone: part of its name, or 'system' (default %s)" % microphone.MICROPHONE)
    p.add_argument("--no-narration", action="store_true", help="do not announce each point of the task")
    p.add_argument("--keyboard", action="store_true", help="type the words instead of speaking (tests)")
    p.add_argument("--mute", action="store_true", help="show the phrases instead of saying them (tests)")
    args = p.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_summary = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_summary.bind((IP, DIALOGUE_PORT))
    sock_summary.setblocking(False)
    sock_pcb = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_pcb.bind((IP, PCB_PORT))
    sock_pcb.setblocking(False)

    def command(name, **extra):
        extra.update({"t": time.time(), "command": name})
        sock.sendto(json.dumps(extra).encode("utf-8"), (IP, COMMANDS_PORT))

    voice = Voice(os.path.join(MODELS_DIR, args.voice + ".onnx"), args.mute,
                  lambda until: command("speaking", until=until))
    ear = None if args.keyboard else Ear(args.device, voice)
    print("Dialogue ready (%s). Words: stop, wait, yes, no, slow down, continue, ready. Ctrl+C to quit."
          % ("keyboard" if args.keyboard else "microphone"))

    phase = "waiting_supervisor"      # -> waiting_start -> working <-> asking_resume / waiting_resume
                                      #    / waiting_hands -> end
    told_clear = False                # in waiting_hands: it already said the zone is clear
    summary, t_summary, previous, t_last_notice = None, 0.0, None, 0.0
    pending = None                    # (notice, state it refers to): it will be said when it is time
    t_voice_order = 0.0               # the summary takes a while to reflect a voice order just sent
    pcb, t_pcb, t_connected = None, 0.0, None   # pcb = True / False / None (unknown)
    calibrating = False               # during calibration the operator talks: they are not listened to
    last_stage = None                 # last stage of the task announced
    try:
        while True:
            t_tick = time.time()
            while True:
                try:
                    summary, t_summary = json.loads(sock_summary.recv(65536).decode("utf-8")), t_tick
                except (BlockingIOError, socket.error):
                    break
                except ValueError:
                    pass
            connected = summary is not None and t_tick - t_summary <= NO_SUPERVISOR
            while True:
                try:
                    pcb, t_pcb = json.loads(sock_pcb.recv(4096).decode("utf-8")).get("present"), t_tick
                except (BlockingIOError, socket.error):
                    break
                except ValueError:
                    pass
            if t_tick - t_pcb > NO_SUPERVISOR:
                pcb = None
            if connected and (summary.get("cause") == "calibrating") != calibrating:
                calibrating = not calibrating
                if calibrating:
                    voice.say("calibrate", urgent=True)
                else:
                    result = summary.get("calibration", "ok")
                    voice.say("calibrated_" + result, urgent=result == "cancelled")
                    if phase == "waiting_start":
                        voice.say("begin?")

            # Words heard (or typed)
            heard = []
            if args.keyboard and select.select([sys.stdin], [], [], 0)[0]:
                heard.append(sys.stdin.readline().strip().lower())
            while ear is not None and not ear.words.empty():
                heard.append(ear.words.get_nowait())

            for text in heard:
                intent = classify(text)
                print("\r\033[K[heard] %r -> %s" % (text, intent or "nothing"))
                if intent is None or not connected or calibrating:
                    continue
                t_voice_order = t_tick
                hands = summary.get("hands") or {}
                if intent == "yes" and phase == "waiting_hands":
                    if hands.get("near"):
                        voice.say("hands_still")        # the supervisor would ignore it too
                    else:
                        command("resume")
                        voice.say("resume_slow")
                        phase, t_last_notice = "working", t_tick
                elif intent == "stop" and phase in ("working", "asking_resume", "waiting_resume", "waiting_hands"):
                    command("pause")                    # stop first, then speak
                    if phase == "working":
                        voice.say("stopped", urgent=True)
                        voice.say("resume?")
                        phase = "asking_resume"
                elif intent == "slow" and phase == "working":
                    command("slow")
                    voice.say("slow")
                    t_last_notice = t_tick
                elif intent == "yes" and (phase == "waiting_pcb" or (phase == "waiting_start" and pcb is False)):
                    voice.say("pcb_reminder")
                    phase = "waiting_pcb"
                elif intent == "yes" and phase == "waiting_start":
                    command("start")
                    voice.say("starting")
                    phase = "working"
                elif intent == "no" and phase == "waiting_start":
                    voice.say("not_yet")
                elif intent == "yes" and phase in ("asking_resume", "waiting_resume"):
                    command("resume")
                    voice.say("resume_slow")
                    phase, t_last_notice = "working", t_tick
                elif intent == "no" and phase == "asking_resume":
                    voice.say("waiting")
                    phase = "waiting_resume"

            # What the system does on its own
            # Greeting: checks the PCB if pcb.py is open; otherwise it greets as usual
            if connected and t_connected is None:
                t_connected = t_tick
            if connected and phase == "waiting_supervisor":
                if pcb is True:
                    voice.say("welcome_pcb")
                    phase = "waiting_start"
                elif pcb is False:
                    voice.say("pcb_missing")
                    phase = "waiting_pcb"
                elif t_tick - t_connected >= PCB_WAIT_S:
                    voice.say("welcome")
                    phase = "waiting_start"
            elif connected and phase == "waiting_pcb" and pcb is not False and not calibrating and not voice.busy():
                voice.say("pcb_ok" if pcb else "welcome")
                phase = "waiting_start"
            # A hand over the PCB also hides it. It only warns before starting: while working, the arm covers it
            elif connected and phase == "waiting_start" and pcb is False and not calibrating and not voice.busy():
                voice.say("pcb_lost")
                phase = "waiting_pcb"
            # Hands next to the PCB: the supervisor already moved the robot away; here it explains and waits for "continue"
            hands = (summary.get("hands") or {}) if connected else {}
            # After a "continue", summaries from before the supervisor removed the lock still arrive
            up_to_date = connected and summary.get("t", 0.0) > t_voice_order + 0.15
            if up_to_date and hands.get("lock") and phase in ("working", "asking_resume", "waiting_resume"):
                voice.say("hands_near", urgent=True)
                phase, told_clear, pending = "waiting_hands", False, None
            elif connected and phase == "waiting_hands":
                if not hands.get("lock") and t_tick - t_voice_order > 1.0:
                    phase = "working"                   # resumed from the keyboard (a, n, c)
                elif hands.get("near"):
                    if told_clear:                      # they came back after it said the area was clear
                        voice.say("hands_again", urgent=True)
                    told_clear = False                  # if they leave again, it says so again
                elif not told_clear and not voice.busy():
                    voice.say("hands_clear")
                    told_clear = True
            # If it starts or resumes from the keyboard (a, n, c, r), the dialogue does not keep waiting for a "yes"
            if connected and phase in ("waiting_pcb", "waiting_start", "asking_resume", "waiting_resume") \
                    and t_tick - t_voice_order > 1.0 \
                    and (summary.get("mode") == "AUTO" or summary.get("state") != "PAUSED"):
                phase = "working"
            if connected and phase == "working":
                robot = summary.get("robot") or {}
                current = (summary.get("state"), summary.get("cause"))
                if robot.get("done"):
                    voice.say("end")
                    phase = "end"
                elif summary.get("mode") == "AUTO" and previous is not None and current != previous:
                    notice = {("CAUTIOUS", "tension"): "tension", ("CAUTIOUS", "no_perception"): "no_perception",
                              ("PAUSED", "high_stress"): "high_stress"}.get(current)
                    if current == ("NORMAL", "calm") and previous[0] == "CAUTIOUS":
                        notice = "normal"
                    elif current == ("CAUTIOUS", "calm") and previous[0] == "PAUSED":
                        notice = "picking_up_slowly"
                    elif current[0] == "NORMAL" and previous == ("CAUTIOUS", "after_hands"):
                        notice = "back_to_normal"
                    if notice:
                        pending = (notice, current)
                elif summary.get("mode") == "MANUAL" and previous is not None and current != previous \
                        and t_tick - t_voice_order > 1.0 and "calibrating" not in (current[1], previous[1]) \
                        and current[1] != "after_hands":
                    notice = {"NORMAL": "manual_normal", "CAUTIOUS": "manual_slow", "PAUSED": "manual_pause",
                              "RETRACT": "manual_retract"}.get(current[0])
                    if previous[1] == "after_hands":   # the slow stretch after a hands stop ended on its own
                        notice = "back_to_normal" if current[0] == "NORMAL" else None
                    if notice:
                        pending = (notice, current)
                # Say the pending notice if it is still true: stops and what the keyboard asks for right away,
                # the rest spaced PHRASE_GAP apart so as not to overwhelm the operator
                if pending and pending[1] != current:
                    pending = None
                urgent = current[0] in ("PAUSED", "RETRACT") or current[1] == "manual"
                if pending and (urgent or (t_tick - t_last_notice >= PHRASE_GAP and not voice.busy())):
                    voice.say(pending[0], urgent=urgent)
                    pending, t_last_notice = None, t_tick
                # Narration: only if there is nothing more urgent. If it is speaking, it waits and announces the
                # stage that is current then, without piling up late phrases
                if not args.no_narration and not pending and current[0] in ("NORMAL", "CAUTIOUS") \
                        and not voice.busy():
                    stage = narration(robot)
                    if stage and stage[0] != last_stage:
                        voice.narrate(stage[1])
                        last_stage = stage[0]
                previous = current
            elif connected:
                previous = (summary.get("state"), summary.get("cause"))

            time.sleep(max(0.0, 1.0 / HZ - (time.time() - t_tick)))
    except KeyboardInterrupt:
        print("\ndialogue closed")


if __name__ == "__main__":
    main()
