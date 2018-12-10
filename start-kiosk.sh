#!/usr/bin/env bash

set -uo pipefail

# replace /home/pi/.config/lxsession/LXDE-pi/autostart with:
#
# @/home/pi/raspberrypi/start-kiosk.sh

SERVER=http://localhost

kiosk_mode() {
    xset -dpms # disable DPMS (Energy Star) features.
    xset s off # disable screen saver
    xset s noblank # don’t blank the video device
    unclutter & # hide mouse pointer
}

check_server() {
    curl --silent --fail --head -o /dev/null "$SERVER"
}

wait_for_server() {
    while true; do
        check_server
        if [ $? ]; then
            break
        fi
        echo 'waiting for server to come up...'
        sleep 5
    done
}

kiosk_mode
wait_for_server
midori -e Fullscreen -a "$SERVER"
