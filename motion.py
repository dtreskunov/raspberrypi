#!/usr/bin/env python3

import ptvsd
ptvsd.enable_attach()
ptvsd.wait_for_attach()  # blocks execution until debugger is attached

import os
import sys

sys.path.append(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'aiyprojects-raspbian/src/aiy'))

import gpiozero
import pins

motion_sensor = gpiozero.MotionSensor(pins.PIN_A, pull_up=True)

if __name__ == '__main__':
    print('hello')