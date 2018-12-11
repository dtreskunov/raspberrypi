#!/usr/bin/env python3

import asyncio
import logging
import sys

from async_generator import async_generator, yield_

import setup_aiy_path
from aiy_vision_hat_motion_sensor import AIYVisionHatMotionSensor
from websocket_broadcaster import WebsocketBroadcaster


@async_generator
async def motion_sensor_events():
    motion_detected = asyncio.Event()

    async def driver_loop():
        import aiy.pins
        sensor = AIYVisionHatMotionSensor(aiy.pins.PIN_A)
        sensor.when_motion = lambda: motion_detected.set()
        while True:
            await asyncio.sleep(100)

    asyncio.ensure_future(driver_loop())

    while True:
        await motion_detected.wait()
        motion_detected.clear()
        await yield_('motion_detected')


async def main():
    async with WebsocketBroadcaster() as server:
        async def send_motion_sensor():
            async for event in motion_sensor_events():
                server.broadcast({'sensor': 'motion_sensor', 'event': event})

        await asyncio.wait([send_motion_sensor()])

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
