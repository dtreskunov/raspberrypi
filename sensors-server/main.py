#!/usr/bin/env python3

import argparse
import asyncio
import base64
import concurrent.futures
import contextlib
import io
import json
import logging
import sys
from functools import partial

import paho.mqtt.client as mqtt
import PIL.Image

import Adafruit_DHT
import aiy.pins
import picamera
import picamera.array
from aiy.leds import Color, Leds, Pattern, PrivacyLed
# import setup_aiy_path
from aiy.vision.inference import (CameraInference, InferenceEngine,
                                  InferenceException)
from aiy.vision.models import face_detection
from aiy_vision_hat_motion_sensor import AIYVisionHatMotionSensor

logger = logging.getLogger(__name__)


async def face_detection_task(callback):
    logger.info('starting face_detection_task')

    def capture_image(camera):
        stream = io.BytesIO()
        camera.capture(stream, format='jpeg')
        # "Rewind" the stream to the beginning so we can read its content
        stream.seek(0)
        return PIL.Image.open(stream)

    def image_to_data_uri(image):
        stream = io.BytesIO()
        image.save(stream, format='JPEG')
        return 'data:image/jpeg;base64,{}'.format(
            base64.b64encode(stream.getvalue()).decode())

    def get_scale(inference_result, image):
        res_h, res_w = inference_result.height, inference_result.width
        img_h, img_w = image.height, image.width
        if img_w / img_h != res_w / res_h:
            raise Exception('Inference result has different aspect ratio than camera capture: {}x{} to {}x{}'.format(
                res_w, res_h, img_w, img_h))
        return img_w / res_w

    def process_inference_result(inference_result, camera):
        faces = face_detection.get_faces(inference_result)
        if not faces:
            return

        logging.info(
            'detected %s faces, taking snapshot for recognition', len(faces))
        # inference runs on the vision bonnet, which grabs images from the camera directly
        # we need to capture the image separately on the Raspberry
        image = capture_image(camera)
        scale = get_scale(inference_result, image)

        def sensor_data_iter():
            for face in faces:
                # translate inference result into img coordinates and slice out just the face
                f_x, f_y, f_w, f_h = face.bounding_box
                face_image = image.crop((
                    max(0, int(scale * f_x)),  # left
                    max(0, int(scale * f_y)),  # upper
                    min(image.width, int(scale * (f_x + f_w))),  # right
                    min(image.height, int(scale * (f_y + f_h))),  # lower
                ))
                # TODO: face recognition :)
                yield {
                    'face_score': face.face_score,
                    'joy_score': face.joy_score,
                    'image_uri': image_to_data_uri(
                        face_image.resize(size=(100, 100), resample=PIL.Image.LANCZOS))
                }

        callback(list(sensor_data_iter()))

    with contextlib.ExitStack() as stack:

        def initialize_inference():
            '''
            One time, the process died without stopping inference, which
            made it impossible to start again. So if we get InferenceException
            when trying to initialize CameraInference, we retry after resetting
            the InferenceEngine.
            '''
            for _ in range(2):
                try:
                    return stack.enter_context(
                        CameraInference(face_detection.model()))
                except InferenceException as e:
                    logger.info(
                        'attempting to reset InferenceEngine due to: %s', e)
                    with InferenceEngine() as engine:
                        engine.reset()
            else:
                raise Exception('unable to start CameraInference')

        leds = stack.enter_context(Leds())
        stack.enter_context(PrivacyLed(leds))

        # Forced sensor mode, 1640x1232, full FoV. See:
        # https://picamera.readthedocs.io/en/release-1.13/fov.html#sensor-modes
        # This is the resolution inference run on.
        camera = stack.enter_context(picamera.PiCamera(
            sensor_mode=4, resolution=(1640, 1232)))

        inference = initialize_inference()

        for inference_result in inference.run():
            process_inference_result(inference_result, camera)


async def motion_sensor_task(callback):
    logger.info('starting motion_sensor_task')
    sensor = AIYVisionHatMotionSensor(aiy.pins.PIN_A)
    sensor.when_motion = lambda: callback('motion_detected')
    while True:
        await asyncio.sleep(1)


async def temperature_humidity_sensor_task(callback):
    logger.info('starting temperature_humidity_sensor_task')
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
        face_detection_task(partial(publish, 'sensor/face_detection')),
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
