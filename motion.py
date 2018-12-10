#!/usr/bin/env python3

import os
import sys

sys.path.append(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'aiyprojects-raspbian/src/aiy'))

import gpiozero
import pins
import signal
import datetime
import functools

def cur_time():
    return datetime.datetime.now().strftime("%H:%M:%S")

class InvertedMotionSensor(gpiozero.MotionSensor):
    pass

InvertedMotionSensor.motion_detected    = lambda *args, **kwds: not super.motion_detected(*args, **kwds)
InvertedMotionSensor.when_motion        = lambda *args, **kwds: super.when_no_motion(*args, **kwds)
InvertedMotionSensor.when_no_motion     = lambda *args, **kwds: super.when_motion(*args, **kwds)
InvertedMotionSensor.wait_for_motion    = lambda *args, **kwds: super.wait_for_no_motion(*args, **kwds)
InvertedMotionSensor.wait_for_no_motion = lambda *args, **kwds: super.wait_for_motion(*args, **kwds)

motion_sensor = InvertedMotionSensor(pins.PIN_A, pull_up=True)
motion_sensor.when_motion = lambda: print('{}: motion'.format(cur_time()))
motion_sensor.when_no_motion = lambda: print('{}: no_motion'.format(cur_time()))

if __name__ == '__main__':
    if '--debug' in sys.argv:
        import ptvsd
        address = ('0.0.0.0', 5678)
        ptvsd.enable_attach(address=address)
        print('Waiting for debugger on {}...'.format(address))
        ptvsd.wait_for_attach()
    print('Waiting for motion...')
    signal.pause()
