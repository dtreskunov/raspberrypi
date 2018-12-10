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

def play_tetris_theme(bpm=240):
    player = TonePlayer(22, bpm=bpm)
    player.play(
        'E5q',
        'Be',
        'C5e',
        'D5e',
        'E5s',
        'D5s',
        'C5s',
        'Be',
        'Bs',
        'Aq',
        'Ae',
        'C5e',
        'E5q',
        'D5e',
        'C5e',
        'Bq',
        'Be',
        'C5e',
        'D5q',
        'E5q',
        'C5q',
        'Aq',
        'Aq'
    )


motion_sensor = AIYVisionHatMotionSensor(pins.PIN_A)
motion_sensor.when_motion = lambda: print('{}: motion'.format(cur_time()))
motion_sensor.when_no_motion = lambda: print('{}: no_motion'.format(cur_time()))

motion_sensor.when_motion = play_tetris_theme

if __name__ == '__main__':
    if '--debug' in sys.argv:
        import ptvsd
        address = ('0.0.0.0', 5678)
        ptvsd.enable_attach(address=address)
        print('Waiting for debugger on {}...'.format(address))
        ptvsd.wait_for_attach()
    print('Waiting for motion...')
    signal.pause()
