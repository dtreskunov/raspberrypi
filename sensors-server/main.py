#!/usr/bin/env python3

import argparse
import asyncio
import json
import logging
from functools import partial

import paho.mqtt.client as mqtt

from .face_recognition.task import face_recognition_task
from .motion_sensor.task import motion_sensor_task
from .temperature_humidity_sensor.task import temperature_humidity_sensor_task
from .util.stopwatch import make_stopwatch

logger = logging.getLogger(__name__)
stopwatch = make_stopwatch(logger)

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


async def main(args):
    client = mqtt_client(args.host, args.port)
    client.loop_start()

    def publish(topic, data):
        with stopwatch('publishing data'):
            msg = json.dumps(data)
            logger.debug('publishing %d-byte JSON message to %s',
                         len(msg), topic)
            client.publish(topic, msg)

    tasks = [
        face_recognition_task(
            partial(publish, 'sensor/face_recognition'), args.face_landmarks_model, args.save_annotated_images_to),
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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Broadcasts Raspberry Pi sensor readings to MQTT broker')
    parser.add_argument(
        '-d', '--debug', help='enable remote debugger compatible with VS Code', action='store_true')
    parser.add_argument(
        '--host', help='MQTT broker host', default='localhost')
    parser.add_argument(
        '--port', help='MQTT broker port', default=1883, type=int)
    parser.add_argument(
        '--loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
    parser.add_argument(
        '--face-landmarks-model', help='model to download from http://dlib.net/files/ (sans .bz2 extension)',
        choices=['shape_predictor_5_face_landmarks.dat',
                 'shape_predictor_68_face_landmarks.dat'],
        default='shape_predictor_68_face_landmarks.dat')
    parser.add_argument(
        '--save-annotated-images-to', help='location for saving images; if not set, images will not be saved')

    args = parser.parse_args()
    if args.debug:
        import ptvsd
        address = ('0.0.0.0', 5678)
        ptvsd.enable_attach(address=address)
        print('Waiting for debugger on {}...'.format(address))
        ptvsd.wait_for_attach()

    logging.basicConfig(level=getattr(logging, args.loglevel))
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(args))
