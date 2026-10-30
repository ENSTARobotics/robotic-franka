#!/usr/bin/env python3
"""Averigua el diccionario de los ArUco de una foto (no usa la cámara).

    python3 detectar_aruco.py foto_de_la_mesa.jpg

Prueba todos los diccionarios de OpenCV y dice con cuál se detectan los marcadores y qué ids
tienen. Los diccionarios del mismo tamaño se contienen (4X4_50 está dentro de 4X4_100...):
se recomienda el más pequeño que contenga todos los ids encontrados.
"""
import sys

import cv2


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    imagen = cv2.imread(sys.argv[1])
    if imagen is None:
        sys.exit("[error] no puedo leer %s" % sys.argv[1])
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    resultados = []
    for nombre in sorted(n for n in dir(cv2.aruco) if n.startswith("DICT_")):
        detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, nombre)),
                                           cv2.aruco.DetectorParameters())
        esquinas, ids, _ = detector.detectMarkers(gris)
        if ids is not None and len(ids):
            lado = sum(cv2.arcLength(e, True) / 4 for e in esquinas) / len(esquinas)
            resultados.append((len(ids), nombre, sorted(int(i) for i in ids.ravel()), lado))

    if not resultados:
        sys.exit("No encontré ningún marcador. Prueba con una foto más cercana, de frente y bien iluminada.")
    maximo = max(r[0] for r in resultados)
    print("Diccionarios con los que se detectan marcadores:")
    for n, nombre, ids, lado in sorted(resultados, key=lambda r: (-r[0], r[1])):
        print("  %-24s %d marcador(es), ids %s, lado medio %.0f px" % (nombre, n, ids, lado))
    mejores = [r for r in resultados if r[0] == maximo]
    def tamano(nombre):
        partes = nombre.split("_")
        return int(partes[-1]) if partes[-1].isdigit() else 10 ** 6
    mejor = min(mejores, key=lambda r: tamano(r[1]))
    print("\nRecomendado: %s (ids %s)" % (mejor[1], mejor[2]))


if __name__ == "__main__":
    main()
