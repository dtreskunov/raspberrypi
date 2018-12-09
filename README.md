# HOWTO

## Clone project including submodules
1. `git clone <GitHub URL>`
1. `cd raspberrypi`
1. `git submodule update --init --recursive`

## Set up environment
1. `pip install -r requirements.txt`
1. Enable port forwarding: *local*, source port `5678`, destination port `localhost:5678`
