#!/usr/bin/env python3

import asyncio
import logging
import sys

import setup_aiy_path
from aiy_vision_hat_motion_sensor import AIYVisionHatMotionSensor
from websocket_broadcaster import WebsocketBroadcaster


async def motion_sensor(callback):
    import aiy.pins
    sensor = AIYVisionHatMotionSensor(aiy.pins.PIN_A)
    sensor.when_motion = lambda: callback('motion_detected')
    while True:
        await asyncio.sleep(10)

async def main():
    async with WebsocketBroadcaster() as server:
        await asyncio.wait([
            motion_sensor(lambda event: server.broadcast({'sensor': 'motion_sensor', 'event': event}))
        ])

if __name__ == '__main__':
    if '--debug' in sys.argv:
        import ptvsd
        address = ('0.0.0.0', 5678)
        ptvsd.enable_attach(address=address)
        print('Waiting for debugger on {}...'.format(address))
        ptvsd.wait_for_attach()

    logging.basicConfig(level=logging.INFO)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
