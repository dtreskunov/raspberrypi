#!/usr/bin/env python3

import argparse
import asyncio
import base64
import bz2
import concurrent.futures
import contextlib
import glob
import io
import json
import logging
import os.path
import sys
import time
import urllib.request
from functools import partial

import numpy
import paho.mqtt.client as mqtt
import PIL.Image

import Adafruit_DHT
import aiy.pins
import dlib
import picamera
import picamera.array
from aiy.leds import Color, Leds, Pattern, PrivacyLed
# import setup_aiy_path
from aiy.vision.inference import (CameraInference, InferenceEngine,
                                  InferenceException)
from aiy.vision.models import face_detection
from aiy_vision_hat_motion_sensor import AIYVisionHatMotionSensor

logger = logging.getLogger(__name__)

INFERENCE_RESOLUTION = (1640, 1232)
CAPTURE_RESOLUTION = (820, 616)
JPEG_QUALITY = 75


@contextlib.contextmanager
def stopwatch(message):
    try:
        begin = time.monotonic()
        yield
    finally:
        end = time.monotonic()
        logger.debug('%s: done in %.3fs', message, end - begin)


async def face_recognition_task(callback):
    logger.info('starting face_recognition_task')

    def capture_image(camera, **kwds):
        with stopwatch('capture_image'):
            stream = io.BytesIO()
            camera.capture(stream, **kwds)
            # "Rewind" the stream to the beginning so we can read its content
            stream.seek(0)
            return PIL.Image.open(stream)

    def image_to_data_uri(image):
        stream = io.BytesIO()
        image.save(stream, format='JPEG', quality=JPEG_QUALITY)
        return 'data:image/jpeg;base64,{}'.format(
            base64.b64encode(stream.getvalue()).decode())

    def get_scale(inference_result, image):
        res_h, res_w = inference_result.height, inference_result.width
        img_h, img_w = image.height, image.width
        if img_w / img_h != res_w / res_h:
            raise Exception('Inference result has different aspect ratio than camera capture: {}x{} to {}x{}'.format(
                res_w, res_h, img_w, img_h))
        return img_w / res_w

    def get_dlib_data():
        '''
        Returns a tuple of filenames, downloading from dlib.net and extracting if necessary:
        1. shape predictor data for dlib.shape_predictor (shape_predictor_5_face_landmarks.dat)
        2. face recognition data for dlib.face_recognition_model_v1 ()
        '''
        def get(url, dest):
            'returns normalized destination path'
            dest = os.path.realpath(os.path.expanduser(dest))
            if not os.path.isfile(dest):
                with stopwatch('downloading from %s and extracting into %s' % (url, dest)):
                    resp = urllib.request.urlopen(url)
                    bz2_file = bz2.BZ2File(io.BytesIO(resp.read()))
                    with open(dest, 'wb') as dest_file:
                        dest_file.write(bz2_file.read())
            return dest

        url_dests = (
            ('http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2',
             '~/.shape_predictor_5_face_landmarks.dat'),
            ('http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2',
             '~/.dlib_face_recognition_resnet_model_v1.dat')
        )
        return tuple(get(url, dest) for url, dest in url_dests)

    def get_face_descriptor(image_arr, rectangle, shape_predictor, face_recognition_model):
        '''
        Compute the 128D vector that describes the face in image.
        In general, if two face descriptor vectors have a Euclidean
        distance between them less than 0.6 then they are from the same
        person, otherwise they are from different people.
        '''
        with stopwatch('get_face_descriptor'):
            left, top, right, bottom = rectangle
            face_landmarks = shape_predictor(
                image_arr,
                dlib.rectangle(left=left, top=top, right=right, bottom=bottom))
            face_descriptor = face_recognition_model.compute_face_descriptor(
                image_arr, face_landmarks)
            return list(face_descriptor)

    def process_inference_result(inference_result, camera, shape_predictor, face_recognition_model):
        faces = face_detection.get_faces(inference_result)
        if not faces:
            return

        with stopwatch('process_inference_result for {} face(s)'.format(len(faces))):
            # inference runs on the vision bonnet, which grabs images from the camera directly
            # we need to capture the image separately on the Raspberry in order to use dlib for face rec
            image = capture_image(camera, format='jpeg', resize=CAPTURE_RESOLUTION, quality=JPEG_QUALITY)

            with stopwatch('numpy.array'):
                image_arr = numpy.array(image)
            
            scale = get_scale(inference_result, image)

            def sensor_data_iter():
                for face in faces:
                    # translate inference result into image coordinates
                    f_x, f_y, f_w, f_h = face.bounding_box
                    f_rectangle = (
                        int(scale * max(0, f_x)),  # left
                        int(scale * max(0, f_y)),  # top
                        int(scale * min(image.width, f_x + f_w)),  # right
                        int(scale * min(image.height, f_y + f_h))  # bottom
                    )
                    face_descriptor = get_face_descriptor(
                        image_arr, f_rectangle, shape_predictor, face_recognition_model)
                    yield {
                        'face_score': face.face_score,
                        'face_descriptor': face_descriptor,
                        'joy_score': face.joy_score,
                        'rectangle': f_rectangle,
                    }
            
            data = {
                'image_uri': image_to_data_uri(image),
                'faces': list(sensor_data_iter()),
            }

            callback(data)

    shape_predictor_data, face_recognition_model_data = get_dlib_data()
    shape_predictor = dlib.shape_predictor(shape_predictor_data)
    face_recognition_model = dlib.face_recognition_model_v1(
        face_recognition_model_data)

    with contextlib.ExitStack() as stack:

        def initialize_inference():
            '''
            One time, the process died without stopping inference, which
            made it impossible to start again. So if we get InferenceException
            when trying to initialize CameraInference, we retry after resetting
            the InferenceEngine.
            '''
            with stopwatch('initialize_inference'):
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
            sensor_mode=4, resolution=INFERENCE_RESOLUTION))

        inference = initialize_inference()

        for inference_result in inference.run():
            process_inference_result(
                inference_result, camera, shape_predictor, face_recognition_model)


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


async def main(host, port):
    client = mqtt_client(host, port)
    client.loop_start()

    def publish(topic, data):
        with stopwatch('publishing data'):
            msg = json.dumps(data)
            logger.debug('publishing %d-byte JSON message to %s', len(msg), topic)
            client.publish(topic, msg)

    tasks = [
        face_recognition_task(partial(publish, 'sensor/face_recognition')),
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
        description='Broadcasts Raspberry Pi sensor readings to MQTT broker')
    parser.add_argument(
        '-d', '--debug', help='enable remote debugger compatible with VS Code', action='store_true')
    parser.add_argument(
        '--host', help='MQTT broker host', default='localhost')
    parser.add_argument(
        '--port', help='MQTT broker port', default=1883, type=int)
    parser.add_argument(
        '--loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')

    args = parser.parse_args()
    if args.debug:
        import ptvsd
        address = ('0.0.0.0', 5678)
        ptvsd.enable_attach(address=address)
        print('Waiting for debugger on {}...'.format(address))
        ptvsd.wait_for_attach()

    logging.basicConfig(level=getattr(logging, args.loglevel))
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(args.host, args.port))