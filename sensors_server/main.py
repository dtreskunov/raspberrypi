#!/usr/bin/env python3

import argparse
import asyncio
import json
import logging
from functools import partial

import paho.mqtt.client as mqtt

import util
from face_recognition.main import FaceRecognitionApp
from motion_sensor.task import motion_sensor_task
from temperature_humidity_sensor.task import temperature_humidity_sensor_task

logger = logging.getLogger(__name__)
stopwatch = util.make_stopwatch(logger)


def mqtt_client(host, port):
    logger.info(
        'Connecting to MQTT broker %s:%s. Hint: you can monitor MQTT traffic by running: mosquitto_sub -h %s -p %s -t \'#\'', host, port, host, port)
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


class FaceRecognitionTask(FaceRecognitionApp):
    def __init__(self, parser, callback):
        super().__init__(parser)
        self._callback = callback

    async def main(self, args):
        super().main(args)

    async def consume(self, data):
        if data:
            self._callback(data.to_dict())
        await asyncio.sleep(0.1)


class SensorsServerApp(util.CLI):
    def __init__(self):
        super().__init__()
        group = self.parser.add_argument_group(title='Sensors server options')
        group.add_argument(
            '--host', help='MQTT broker host', default='localhost')
        group.add_argument(
            '--port', help='MQTT broker port', default=1883, type=int)

    def main(self, args):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_main(args))

    async def async_main(self, args):
        client = mqtt_client(args.host, args.port)
        client.loop_start()

        def publish(topic, data):
            with stopwatch('publishing data'):
                msg = json.dumps(data)
                logger.debug('publishing %d-byte JSON message to %s',
                             len(msg), topic)
                client.publish(topic, msg)

        tasks = [
            FaceRecognitionTask(self.parser, partial(
                publish, 'sensor/face_recognition')).main(args),
            motion_sensor_task(partial(publish, 'sensor/motion')),
            temperature_humidity_sensor_task(
                partial(publish, 'sensor/temperature_humidity')),
        ]
        for x in asyncio.as_completed(tasks):
            try:
                await x
                logger.warning('sensor task completed unexpectedly')
            except Exception:
                logger.exception('sensor task died')
        logger.info('all sensor tasks completed')


if __name__ == '__main__':
    SensorsServerApp().run()
