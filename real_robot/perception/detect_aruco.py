#!/usr/bin/env python3
"""Finds out the dictionary of the ArUco markers in a photo (does not use the camera).

    python3 detect_aruco.py photo_of_the_table.jpg

Tries every OpenCV dictionary and says with which ones the markers are detected and which ids
they have. Dictionaries of the same size contain each other (4X4_50 is inside 4X4_100...):
the smallest one that contains every id found is recommended.
"""
import sys

import cv2


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    image = cv2.imread(sys.argv[1])
    if image is None:
        sys.exit("[error] cannot read %s" % sys.argv[1])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    results = []
    for name in sorted(n for n in dir(cv2.aruco) if n.startswith("DICT_")):
        detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name)),
                                           cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None and len(ids):
            side = sum(cv2.arcLength(c, True) / 4 for c in corners) / len(corners)
            results.append((len(ids), name, sorted(int(i) for i in ids.ravel()), side))

    if not results:
        sys.exit("No marker found. Try a closer photo, taken straight on and well lit.")
    most = max(r[0] for r in results)
    print("Dictionaries with which markers are detected:")
    for n, name, ids, side in sorted(results, key=lambda r: (-r[0], r[1])):
        print("  %-24s %d marker(s), ids %s, mean side %.0f px" % (name, n, ids, side))
    best = [r for r in results if r[0] == most]

    def size(name):
        parts = name.split("_")
        return int(parts[-1]) if parts[-1].isdigit() else 10 ** 6
    chosen = min(best, key=lambda r: size(r[1]))
    print("\nRecommended: %s (ids %s)" % (chosen[1], chosen[2]))


if __name__ == "__main__":
    main()
