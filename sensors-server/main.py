#!/usr/bin/env python3

import argparse
import asyncio
import concurrent.futures
import json
import logging
import sys
from functools import partial

import paho.mqtt.client as mqtt

import setup_aiy_path
from aiy_vision_hat_motion_sensor import AIYVisionHatMotionSensor

logger = logging.getLogger(__name__)


async def motion_sensor_task(callback):
    logger.info('starting motion_sensor_task')
    import aiy.pins
    sensor = AIYVisionHatMotionSensor(aiy.pins.PIN_A)
    sensor.when_motion = lambda: callback('motion_detected')
    while True:
        await asyncio.sleep(1)


async def temperature_humidity_sensor_task(callback):
    logger.info('starting temperature_humidity_sensor_task')
    import aiy.pins
    import Adafruit_DHT
    while True:
        # Try to grab a sensor reading.  Use the read_retry method which will retry up
        # to 15 times to get a sensor reading (waiting 2 seconds between each retry).
        humidity, temperature = Adafruit_DHT.read_retry(
            Adafruit_DHT.DHT11, aiy.pins.PIN_B.gpio_spec.pin)
        if humidity is not None:
            callback({'humidity': humidity})
        if temperature is not None:
            callback({'temperature': temperature})
        await asyncio.sleep(10)


def mqtt_client(host, port):
    client = mqtt.Client()
    client.reconnect_delay_set()

    def on_connect(client, userdata, flags, rc):
        # add subscriptions here
        logger.info('Connected to MQTT broker %s:%s, flags=%s, rc=%s',
                    host, port, flags, rc)

    def on_disconnect(client, userdata, rc):
        logger.info('Disconnected from MQTT broker %s:%s, rc=%s',
                    host, port, rc)

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.connect_async(host, port)
    return client


async def main(host, port):
    client = mqtt_client(host, port)
    client.loop_start()

    def publish(topic, data):
        msg = json.dumps(data)
        logger.info('publishing to %s: %s', topic, msg)
        client.publish(topic, msg)

    tasks = [
        motion_sensor_task(partial(publish, 'sensor/motion')),
        temperature_humidity_sensor_task(
            partial(publish, 'sensor/temperature_humidity')),
    ]
    for x in asyncio.as_completed(tasks):
        try:
            await x
            logger.warning('sensor task completed unexpectedly')
        except Exception as e:
            logger.error('sensor task died with exception: %s', e)
    logger.info('all sensor tasks completed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Broadcasts Raspberry Pi sensor readings to MQTT broker')
    parser.add_argument(
        '-d', '--debug', help='enable remote debugger compatible with VS Code', action='store_true')
    parser.add_argument(
        '--host', help='MQTT broker host', default='localhost')
    parser.add_argument(
        '--port', help='MQTT broker port', default=1883, type=int)
    args = parser.parse_args()
    if args.debug:
        import ptvsd
        address = ('0.0.0.0', 5678)
        ptvsd.enable_attach(address=address)
        print('Waiting for debugger on {}...'.format(address))
        ptvsd.wait_for_attach()

    logging.basicConfig(level=logging.INFO)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(args.host, args.port))
