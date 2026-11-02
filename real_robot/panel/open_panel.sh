#!/bin/bash
# Opens the control panel: starts panel.py if it is not running, and shows it as a Chrome app window.
#   panel/open_panel.sh             # open it
#   panel/open_panel.sh --install   # add "Aided Building" to the Ubuntu app menu (then open it from there)
DIR=$(dirname "$(realpath "$0")")
PORT=8765
URL=http://127.0.0.1:$PORT/

if [ "$1" = --install ]; then
    mkdir -p ~/.local/share/applications
    cat > ~/.local/share/applications/aided-building-panel.desktop <<EOF
[Desktop Entry]
Type=Application
Name=Aided Building
Comment=Control panel for the soldering robot
Exec=$DIR/open_panel.sh
Icon=$DIR/icon.svg
StartupWMClass=chrome-127.0.0.1__-Default
Terminal=false
Categories=Development;
EOF
    echo "Added: look for \"Aided Building\" in the app menu (Super key)."
    exit 0
fi

if ! curl -s -o /dev/null "$URL"; then
    mkdir -p "$DIR/../logs"
    # The panel's own messages go to logs/panel.log; it keeps running if the window is closed
    nohup python3 "$DIR/panel.py" --port "$PORT" >> "$DIR/../logs/panel.log" 2>&1 &
    for _ in $(seq 50); do curl -s -o /dev/null "$URL" && break; sleep 0.1; done
fi

if command -v google-chrome >/dev/null; then
    google-chrome --app="$URL" --window-size=1400,900 >/dev/null 2>&1 &
else
    xdg-open "$URL" >/dev/null 2>&1 &
fi
