#!/bin/bash
# Password window for "sudo -A" (panel.py): asked once, then sudo remembers it while the panel runs
zenity --password --title="Aided Building: your password, for Docker" 2>/dev/null
