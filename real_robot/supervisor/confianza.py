"""Fusión de la percepción en un índice de confianza T(t) y máquina de estados de seguridad.

Cada fuente (rostro, audio, ...) envía estrés en [0, 1] y una confianza en [0, 1]:
    estrés fusionado = sum(peso_i * confianza_i * estres_i) / (CONF_PREVIA + sum(peso_i * confianza_i))
    T(t)             = 1 - estrés fusionado, suavizado con constante de tiempo TAU_T
Una fuente cuenta sólo si su último mensaje tiene menos de EDAD_MAX s y confianza > CONF_MIN.

CONF_PREVIA tira hacia "sin estrés" cuando hay poca confianza total: sin ella, una sola fuente con
confianza 0.1 contaba como si fuera segura y paraba el robot por una falsa alarma.

Calibración: el nivel normal de cada fuente es la mediana de su estrés mientras el operador está
tranquilo, y luego sólo cuenta lo que lo supere: estrés' = max(0, estrés - normal) / (1 - normal).

Máquina de estados (con histéresis):
    bajar es inmediato:  NORMAL -> CAUTIOUS si T < T_CAUTIOUS;  cualquiera -> PAUSED si T < T_PAUSA
    subir pide calma sostenida:  PAUSED -> CAUTIOUS si T > T_REANUDAR durante S_REANUDAR s
                                 CAUTIOUS -> NORMAL si T > T_NORMAL durante S_NORMAL s
    sin percepción válida durante SIN_DATOS s: NORMAL -> CAUTIOUS (prudente, sin parar)
Los umbrales son un punto de partida: se ajustan con los datos del piloto.
"""
import time

PESOS = {"rostro": 0.7, "audio": 0.3}
EDAD_MAX = 1.0              # s: un mensaje más viejo no cuenta
CONF_MIN = 0.05
CONF_PREVIA = 0.1           # peso de la hipótesis "sin estrés" en la media
BASE_MAX = 0.8              # tope del nivel normal calibrado: por encima no quedaría margen para detectar
MIN_MUESTRAS = 10           # muestras de una fuente para calibrarla; si no, se queda sin calibrar
TAU_T = 0.5                 # s: suavizado de T(t)

T_CAUTIOUS = 0.6            # T por debajo -> CAUTIOUS
T_PAUSA = 0.35              # T por debajo -> PAUSED
T_REANUDAR = 0.5            # PAUSED -> CAUTIOUS si T por encima ...
S_REANUDAR = 2.0            # ... durante estos segundos
T_NORMAL = 0.75             # CAUTIOUS -> NORMAL si T por encima ...
S_NORMAL = 3.0              # ... durante estos segundos
SIN_DATOS = 2.0             # s sin ninguna fuente válida -> como mucho CAUTIOUS


class Confianza(object):
    """Guarda el último mensaje de cada fuente y calcula T(t)."""

    def __init__(self):
        self.ultimo = {}            # fuente -> (t_recepcion, mensaje)
        self.normal = {}            # fuente -> nivel de estrés normal de este operador
        self.muestras = None        # fuente -> estrés recibido, mientras se calibra
        self.T = None
        self.estres = None
        self._t_previo = None
        self.t_ultima_valida = None

    def recibir(self, mensaje, ahora=None):
        fuente = mensaje.get("fuente")
        if fuente:
            self.ultimo[fuente] = (time.time() if ahora is None else ahora, mensaje)
            if self.muestras is not None and mensaje.get("estres") is not None and not mensaje.get("mantenido") \
                    and float(mensaje.get("confianza") or 0.0) > CONF_MIN:
                self.muestras.setdefault(fuente, []).append(float(mensaje["estres"]))

    def empezar_calibracion(self):
        self.muestras = {}

    def cancelar_calibracion(self):
        self.muestras = None        # se queda el nivel normal de la calibración anterior, si la hubo

    def terminar_calibracion(self):
        """Sustituye el nivel normal por el medido; una fuente con pocas muestras queda sin calibrar."""
        self.normal = {fuente: min(BASE_MAX, sorted(valores)[len(valores) // 2])
                       for fuente, valores in (self.muestras or {}).items() if len(valores) >= MIN_MUESTRAS}
        self.muestras = None
        return dict(self.normal)

    def validas(self, ahora):
        """{fuente: (estres, confianza)} de las fuentes que cuentan ahora mismo."""
        out = {}
        for fuente, (t, m) in self.ultimo.items():
            estres, conf = m.get("estres"), float(m.get("confianza") or 0.0)
            if ahora - t <= EDAD_MAX and estres is not None and conf > CONF_MIN:
                base = self.normal.get(fuente, 0.0)
                out[fuente] = (min(1.0, max(0.0, (float(estres) - base) / (1.0 - base))), min(1.0, conf))
        return out

    def actualizar(self, ahora=None):
        ahora = time.time() if ahora is None else ahora
        validas = self.validas(ahora)
        suma_w = sum(PESOS.get(f, 1.0) * c for f, (_, c) in validas.items())
        if suma_w > 0:
            self.estres = sum(PESOS.get(f, 1.0) * c * s for f, (s, c) in validas.items()) / (CONF_PREVIA + suma_w)
            # Desde el mensaje válido más reciente (no desde ahora): SIN_DATOS cuenta desde él
            self.t_ultima_valida = max(self.ultimo[f][0] for f in validas)
            T_inst = 1.0 - self.estres
            if self.T is None or self._t_previo is None:
                self.T = T_inst
            else:
                a = min(1.0, (ahora - self._t_previo) / TAU_T)
                self.T += a * (T_inst - self.T)
        self._t_previo = ahora
        return self.T

    def sin_datos(self, ahora):
        return self.t_ultima_valida is None or ahora - self.t_ultima_valida > SIN_DATOS


class MaquinaEstados(object):
    """NORMAL / CAUTIOUS / PAUSED a partir de T(t). Empieza en PAUSED (seguro)."""

    def __init__(self):
        self.estado = "PAUSED"
        self.motivo = "inicio"
        self.causa = "inicio"       # código para la pantalla: inicio, tension, estres_alto,
                                    # sin_percepcion, calma, voz
        self._desde_calma = None    # desde cuándo T permite subir de estado
        self._sin_subir_hasta = 0.0  # tras forzar(): no subir de estado antes de esto

    def forzar(self, estado, causa, segundos, ahora=None):
        """Estado pedido desde fuera (p. ej. "slow down" por voz) que no sube antes de `segundos`.
        Bajar sigue siendo posible: si T cae, la máquina pasa a PAUSED igualmente."""
        ahora = time.time() if ahora is None else ahora
        self.estado, self.causa, self.motivo = estado, causa, "pedido por %s" % causa
        self._sin_subir_hasta = ahora + segundos
        self._desde_calma = None

    def actualizar(self, T, sin_datos, ahora=None):
        ahora = time.time() if ahora is None else ahora
        previo = self.estado
        if sin_datos or T is None:
            self._desde_calma = None
            if self.estado == "NORMAL":
                self.estado, self.motivo, self.causa = "CAUTIOUS", "sin percepción", "sin_percepcion"
            elif T is None and self.estado == "PAUSED" and self.motivo == "inicio":
                self.motivo = "esperando percepción"
            return self.estado != previo

        if T < T_PAUSA:
            self.estado, self.motivo, self.causa = "PAUSED", "confianza muy baja (T=%.2f)" % T, "estres_alto"
        elif T < T_CAUTIOUS and self.estado == "NORMAL":
            self.estado, self.motivo, self.causa = "CAUTIOUS", "tensión detectada (T=%.2f)" % T, "tension"
        else:
            umbral, espera, siguiente = {"PAUSED": (T_REANUDAR, S_REANUDAR, "CAUTIOUS"),
                                         "CAUTIOUS": (T_NORMAL, S_NORMAL, "NORMAL")}.get(
                                             self.estado, (None, None, None))
            if umbral is not None and T > umbral and ahora >= self._sin_subir_hasta:
                if self._desde_calma is None:
                    self._desde_calma = ahora
                elif ahora - self._desde_calma >= espera:
                    self.estado, self.motivo, self.causa = siguiente, "operador tranquilo (T=%.2f)" % T, "calma"
            else:
                self._desde_calma = None
        if self.estado != previo:
            self._desde_calma = None
        return self.estado != previo
