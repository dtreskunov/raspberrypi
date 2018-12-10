#!/usr/bin/env python3

import os
import sys

sys.path.append(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'aiyprojects-raspbian/src/aiy'))

import pins
from aiy.toneplayer import TonePlayer
import signal
import datetime
from aiy_vision_hat_motion_sensor import AIYVisionHatMotionSensor

def cur_time():
    return datetime.datetime.now().strftime("%H:%M:%S")

motion_sensor = AIYVisionHatMotionSensor(pins.PIN_A)
player = TonePlayer(22)

def when_motion():
    print('{}: motion'.format(cur_time()))
    player.play('Cs', 'Es', 'Gq')

def when_no_motion():
    print('{}: no_motion'.format(cur_time()))
    player.play('Gs', 'Es', 'Cs')

motion_sensor.when_motion = when_motion
motion_sensor.when_no_motion = when_no_motion

if __name__ == '__main__':
    if '--debug' in sys.argv:
        import ptvsd
        address = ('0.0.0.0', 5678)
        ptvsd.enable_attach(address=address)
        print('Waiting for debugger on {}...'.format(address))
        ptvsd.wait_for_attach()
    print('Waiting for motion...')
    signal.pause()
