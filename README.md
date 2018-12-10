# HOWTO

## Clone project including submodules
1. `git clone <GitHub URL>`
1. `cd raspberrypi`
1. `git submodule update --init --recursive`

## Set up environment
1. `pip install -r requirements.txt`
1. Enable port forwarding: *local*, source port `5678`, destination port `localhost:5678`

## Install services
1. `sudo apt-get install docker-compose`
1. `sudo ln -s /home/pi/raspberrypi/magic-mirror-server.service /lib/systemd/system`
1. `sudo systemctl daemon-reload`
1. `sudo systemctl enable magic-mirror-server.service`
1. `sudo systemctl start magic-mirror-server.service`
1. `sudo systemctl status magic-mirror-server.service`
