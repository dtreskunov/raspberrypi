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
        await asyncio.sleep(1)

async def temperature_humidity_sensor(callback):
    import aiy.pins
    import Adafruit_DHT
    while True:
        # Try to grab a sensor reading.  Use the read_retry method which will retry up
        # to 15 times to get a sensor reading (waiting 2 seconds between each retry).
        humidity, temperature = Adafruit_DHT.read_retry(Adafruit_DHT.DHT11, aiy.pins.PIN_B.gpio_spec.pin)
        if humidity is not None:
            callback({'humidity': humidity})
        if temperature is not None:
            callback({'temperature': temperature})
        await asyncio.sleep(10)


async def main():
    async with WebsocketBroadcaster() as server:
        tasks = [
            motion_sensor(lambda event: server.broadcast({'sensor': 'motion_sensor', 'event': event})),
            temperature_humidity_sensor(lambda event: server.broadcast({'sensor': 'temperature_humidity_sensor', 'event': event}))
        ]
        for x in asyncio.as_completed(tasks):
            try:
                await x
            except Exception as e:
                print('task finished with exception: {}'.format(e))
        print('all tasks finished')


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
