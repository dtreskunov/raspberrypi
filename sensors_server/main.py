#!/usr/bin/env python3

import argparse
import asyncio
import json
import logging
import threading
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


class FaceRecognitionWrapper(FaceRecognitionApp):
    def main(self, callback, args, shutdown):
        self._callback = callback
        self._shutdown = shutdown
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(None, super().main, args)

    def consume(self, data):
        if self._shutdown.is_set():
            raise Exception('Shutdown signaled')
        if data:
            self._callback(data.to_dict())


class SensorsServerApp(util.CLI):
    def __init__(self):
        super().__init__()
        self._shutdown = threading.Event()
        self._face_recognition_wrapper = FaceRecognitionWrapper(self.parser)

        group = self.parser.add_argument_group(title='Sensors server options')
        group.add_argument(
            '--host', help='MQTT broker host', default='localhost')
        group.add_argument(
            '--port', help='MQTT broker port', default=1883, type=int)

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
            self._face_recognition_wrapper.main(
                partial(publish, 'sensor/face_recognition'),
                args, self._shutdown),
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
    
    def main(self, args):
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(self.async_main(args))
        except (SystemExit, KeyboardInterrupt):
            self._shutdown.set()
            raise

if __name__ == '__main__':
    SensorsServerApp().run()
