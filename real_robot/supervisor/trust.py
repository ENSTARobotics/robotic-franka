"""Fuses perception into a trust index T(t), and the safety state machine.

Each source (face, audio, ...) sends a stress in [0, 1] and a confidence in [0, 1]:
    fused stress = sum(weight_i * confidence_i * stress_i) / (PRIOR_CONF + sum(weight_i * confidence_i))
    T(t)         = 1 - fused stress, smoothed with time constant TAU_T
A source only counts if its last message is younger than MAX_AGE s and its confidence > MIN_CONF.

PRIOR_CONF pulls toward "no stress" when the total confidence is low: without it, a single source
with confidence 0.1 counted as if it were sure and stopped the robot with a false alarm.

Calibration: each source's baseline is the median of its stress while the operator is calm, and
afterwards only what goes above it counts: stress' = max(0, stress - baseline) / (1 - baseline).

State machine (with hysteresis):
    going down is immediate:  NORMAL -> CAUTIOUS if T < T_CAUTIOUS;  any -> PAUSED if T < T_PAUSE
    going up needs sustained calm:  PAUSED -> CAUTIOUS if T > T_RESUME for S_RESUME s
                                    CAUTIOUS -> NORMAL if T > T_NORMAL for S_NORMAL s
    no valid perception for NO_DATA s: NORMAL -> CAUTIOUS (careful, without stopping)
The thresholds are a starting point: they are tuned with the pilot's data.
"""
import time

WEIGHTS = {"face": 0.7, "audio": 0.3}
MAX_AGE = 1.0               # s: an older message does not count
MIN_CONF = 0.05
PRIOR_CONF = 0.1            # weight of the "no stress" hypothesis in the mean
MAX_BASELINE = 0.8          # cap on the calibrated baseline: above it there would be no room left to detect
MIN_SAMPLES = 10            # samples of a source needed to calibrate it; otherwise it stays uncalibrated
TAU_T = 0.5                 # s: smoothing of T(t)

T_CAUTIOUS = 0.6            # T below -> CAUTIOUS
T_PAUSE = 0.35              # T below -> PAUSED
T_RESUME = 0.5              # PAUSED -> CAUTIOUS if T above ...
S_RESUME = 2.0              # ... for these seconds
T_NORMAL = 0.75             # CAUTIOUS -> NORMAL if T above ...
S_NORMAL = 3.0              # ... for these seconds
NO_DATA = 2.0               # s without any valid source -> CAUTIOUS at most


class Trust(object):
    """Keeps the last message of each source and computes T(t)."""

    def __init__(self):
        self.last = {}              # source -> (t_received, message)
        self.baseline = {}          # source -> this operator's normal stress level
        self.samples = None         # source -> stress received, while calibrating
        self.T = None
        self.stress = None
        self._t_prev = None
        self.t_last_valid = None

    def receive(self, message, now=None):
        source = message.get("source")
        if source:
            self.last[source] = (time.time() if now is None else now, message)
            if self.samples is not None and message.get("stress") is not None and not message.get("held") \
                    and float(message.get("confidence") or 0.0) > MIN_CONF:
                self.samples.setdefault(source, []).append(float(message["stress"]))

    def start_calibration(self):
        self.samples = {}

    def cancel_calibration(self):
        self.samples = None         # the baseline from the previous calibration, if any, is kept

    def finish_calibration(self):
        """Replaces the baseline with the measured one; a source with few samples stays uncalibrated."""
        self.baseline = {source: min(MAX_BASELINE, sorted(values)[len(values) // 2])
                         for source, values in (self.samples or {}).items() if len(values) >= MIN_SAMPLES}
        self.samples = None
        return dict(self.baseline)

    def valid(self, now):
        """{source: (stress, confidence)} of the sources that count right now."""
        out = {}
        for source, (t, m) in self.last.items():
            stress, conf = m.get("stress"), float(m.get("confidence") or 0.0)
            if now - t <= MAX_AGE and stress is not None and conf > MIN_CONF:
                base = self.baseline.get(source, 0.0)
                out[source] = (min(1.0, max(0.0, (float(stress) - base) / (1.0 - base))), min(1.0, conf))
        return out

    def update(self, now=None):
        now = time.time() if now is None else now
        valid = self.valid(now)
        sum_w = sum(WEIGHTS.get(f, 1.0) * c for f, (_, c) in valid.items())
        if sum_w > 0:
            self.stress = sum(WEIGHTS.get(f, 1.0) * c * s for f, (s, c) in valid.items()) / (PRIOR_CONF + sum_w)
            # From the most recent valid message (not from now): NO_DATA counts from it
            self.t_last_valid = max(self.last[f][0] for f in valid)
            T_inst = 1.0 - self.stress
            if self.T is None or self._t_prev is None:
                self.T = T_inst
            else:
                a = min(1.0, (now - self._t_prev) / TAU_T)
                self.T += a * (T_inst - self.T)
        self._t_prev = now
        return self.T

    def no_data(self, now):
        return self.t_last_valid is None or now - self.t_last_valid > NO_DATA


class StateMachine(object):
    """NORMAL / CAUTIOUS / PAUSED from T(t). Starts in PAUSED (safe)."""

    def __init__(self):
        self.state = "PAUSED"
        self.reason = "start"
        self.cause = "start"        # code for the screen: start, tension, high_stress,
                                    # no_perception, calm, voice
        self._calm_since = None     # since when T allows going up a state
        self._no_rise_until = 0.0   # after force(): do not go up a state before this

    def force(self, state, cause, seconds, now=None):
        """State requested from outside (e.g. "slow down" by voice) that does not go up before `seconds`.
        Going down is still possible: if T drops, the machine goes to PAUSED anyway."""
        now = time.time() if now is None else now
        self.state, self.cause, self.reason = state, cause, "requested by %s" % cause
        self._no_rise_until = now + seconds
        self._calm_since = None

    def update(self, T, no_data, now=None):
        now = time.time() if now is None else now
        previous = self.state
        if no_data or T is None:
            self._calm_since = None
            if self.state == "NORMAL":
                self.state, self.reason, self.cause = "CAUTIOUS", "no perception", "no_perception"
            elif T is None and self.state == "PAUSED" and self.reason == "start":
                self.reason = "waiting for perception"
            return self.state != previous

        if T < T_PAUSE:
            self.state, self.reason, self.cause = "PAUSED", "very low trust (T=%.2f)" % T, "high_stress"
        elif T < T_CAUTIOUS and self.state == "NORMAL":
            self.state, self.reason, self.cause = "CAUTIOUS", "tension detected (T=%.2f)" % T, "tension"
        else:
            threshold, wait, following = {"PAUSED": (T_RESUME, S_RESUME, "CAUTIOUS"),
                                          "CAUTIOUS": (T_NORMAL, S_NORMAL, "NORMAL")}.get(
                                              self.state, (None, None, None))
            if threshold is not None and T > threshold and now >= self._no_rise_until:
                if self._calm_since is None:
                    self._calm_since = now
                elif now - self._calm_since >= wait:
                    self.state, self.reason, self.cause = following, "calm operator (T=%.2f)" % T, "calm"
            else:
                self._calm_since = None
        if self.state != previous:
            self._calm_since = None
        return self.state != previous
