import asyncio
import base64
import bz2
import contextlib
import glob
import io
import json
import logging
import os.path
import sys
import time
import urllib.request
from uuid import UUID, uuid4

import numpy
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

import dlib
import picamera
import picamera.array
from aiy.leds import Color, Leds, Pattern, PrivacyLed
from aiy.vision.inference import (CameraInference, InferenceEngine,
                                  InferenceException)
from aiy.vision.models import face_detection
from util.stopwatch import make_stopwatch

from .classifier import Classifier
from .constants import DATA_DIR
from .entities import DetectedFace, Image, Person, db_connection, db_transaction

logger = logging.getLogger(__name__)
stopwatch = make_stopwatch(logger)

INFERENCE_RESOLUTION = (1640, 1232)
CAPTURE_RESOLUTION = (820, 616)
JPEG_QUALITY = 75
FONT_FILE = '/usr/share/fonts/truetype/freefont/FreeSans.ttf'


async def face_recognition_task(callback, face_landmarks_model, save_annotated_images_to):
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
        1. shape predictor data for dlib.shape_predictor (one of shape_predictor_*_face_landmarks.dat)
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
            ('http://dlib.net/files/{}.bz2'.format(face_landmarks_model),
             '{}/{}'.format(DATA_DIR, face_landmarks_model)),
            ('http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2',
             '{}/dlib_face_recognition_resnet_model_v1.dat'.format(DATA_DIR))
        )
        return tuple(get(url, dest) for url, dest in url_dests)

    def get_face_landmarks(image_arr, rectangle, shape_predictor):
        with stopwatch('get_face_landmarks'):
            left, top, right, bottom = rectangle
            return shape_predictor(
                image_arr,
                dlib.rectangle(left=left, top=top, right=right, bottom=bottom))

    def get_face_descriptor(image_arr, face_landmarks, face_recognition_model):
        '''
        Compute the 128D vector that describes the face in image.
        In general, if two face descriptor vectors have a Euclidean
        distance between them less than 0.6 then they are from the same
        person, otherwise they are from different people.
        '''
        with stopwatch('get_face_descriptor'):
            face_descriptor = face_recognition_model.compute_face_descriptor(
                image_arr, face_landmarks)
            return list(face_descriptor)

    def label_face_landmarks(face_landmarks):
        '''
        :return: A list of dicts of face feature locations (eyes, nose, etc)
        '''
        points = [(p.x, p.y) for p in face_landmarks.parts()]

        # For a definition of each point index, see https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
        if face_landmarks_model == 'shape_predictor_68_face_landmarks.dat':
            return {
                "chin": points[0:17],
                "left_eyebrow": points[17:22],
                "right_eyebrow": points[22:27],
                "nose_bridge": points[27:31],
                "nose_tip": points[31:36],
                "left_eye": points[36:42],
                "right_eye": points[42:48],
                "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
                "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]]
            }
        elif face_landmarks_model == 'shape_predictor_5_face_landmarks.dat':
            return {
                "nose_tip": [points[4]],
                "left_eye": points[2:4],
                "right_eye": points[0:2],
            }
        else:
            logger.warning(
                'unable to label face landmarks - unrecognized model %s', face_landmarks_model)
            return {}

    def annotate_image_with_sensor_data(image, data):
        'draws on top of the image - save a copy before annotating if you need the original!'

        def draw_rectangle(draw, x0, y0, x1, y1, border, fill=None, outline=None):
            assert border % 2 == 1
            for i in range(-border // 2, border // 2 + 1):
                draw.rectangle((x0 + i, y0 + i, x1 - i, y1 - i),
                               fill=fill, outline=outline)

        draw = PIL.ImageDraw.Draw(image)
        font = PIL.ImageFont.truetype(FONT_FILE, size=15)
        for face in data['faces']:
            left, top, right, bottom = face['rectangle']
            person = face['person']
            text = '%s (joy: %.2f)' % (person['name'], face['joy_score'])
            _, text_height = font.getsize(text)
            margin = 3
            text_bottom = bottom + margin + text_height + margin
            draw_rectangle(draw, left, top, right, bottom, 3, outline='white')
            draw_rectangle(draw, left, bottom, right,
                           text_bottom, 3, fill='white', outline='white')
            draw.text((left + 1 + margin, bottom + 1 + margin),
                      text, font=font, fill='black')
            for _, points in face['face_landmarks'].items():
                draw.line(points, fill='red', width=1)

    def save_image(image, file):
        os.makedirs(os.path.dirname(file), exist_ok=True)
        image.save(file)
        logger.debug('saved %s', file)

    def process_inference_result(inference_result, camera, shape_predictor, face_recognition_model, classifier):
        faces = face_detection.get_faces(inference_result)
        if not faces:
            return

        with stopwatch('process_inference_result for {} face(s)'.format(len(faces))):
            # inference runs on the vision bonnet, which grabs images from the camera directly
            # we need to capture the image separately on the Raspberry in order to use dlib for face rec
            image = capture_image(
                camera, format='jpeg', quality=JPEG_QUALITY, use_video_port=True)

            with stopwatch('numpy.array'):
                image_arr = numpy.array(image)

            scale = get_scale(inference_result, image)

            def sensor_data_iter():
                for face in faces:
                    # translate inference result into image coordinates
                    f_x, f_y, f_w, f_h = face.bounding_box
                    f_rectangle = (
                        max(0, int(scale * f_x)),  # left
                        max(0, int(scale * f_y)),  # top
                        min(image.width, int(scale * (f_x + f_w))),  # right
                        min(image.height, int(scale * (f_y + f_h))),  # bottom
                    )
                    face_landmarks = get_face_landmarks(
                        image_arr, f_rectangle, shape_predictor)
                    face_descriptor = get_face_descriptor(
                        image_arr, face_landmarks, face_recognition_model)
                    person, is_new, dist = classifier.recognize_person(
                        face_descriptor)

                    labeled_face_landmarks = label_face_landmarks(
                        face_landmarks)
                    yield {
                        'face_score': face.face_score,
                        'person': {
                            'name': person.name,
                            'uuid': str(person.uuid),
                            'is_new': is_new,
                            'dist': dist,
                        },
                        'face_landmarks': labeled_face_landmarks,
                        'joy_score': face.joy_score,
                        'rectangle': f_rectangle,
                    }

            data = {
                'image_uri': image_to_data_uri(image),
                'faces': list(sensor_data_iter()),
            }

            if save_annotated_images_to:
                timestamp = time.strftime('%Y-%m-%d_%H.%M.%S')
                filename = os.path.expanduser(
                    '{}/{}.jpg'.format(save_annotated_images_to, timestamp))
                with stopwatch('saving annotated image to {}'.format(filename)):
                    annotate_image_with_sensor_data(image, data)
                    save_image(image, filename)

            callback(data)

    with stopwatch('initializing dlib objects'):
        shape_predictor_model_path, face_recognition_model_path = get_dlib_data()
        shape_predictor = dlib.shape_predictor(shape_predictor_model_path)
        face_recognition_model = dlib.face_recognition_model_v1(
            face_recognition_model_path)

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
        with stopwatch('initialize camera'):
            camera = stack.enter_context(picamera.PiCamera(
                sensor_mode=4, resolution=CAPTURE_RESOLUTION))

        inference = initialize_inference()

        db_filename = os.path.join(os.path.expanduser(DATA_DIR), 'camera.sqlite')
        os.makedirs(os.path.dirname(db_filename), exist_ok=True)
        stack.enter_context(db_connection(provider='sqlite', filename=db_filename, create_db=True))

        with stopwatch('initialize classifier'):
            classifier = stack.enter_context(Classifier())

        for inference_result in inference.run():
            with db_transaction:
                process_inference_result(
                    inference_result, camera, shape_predictor, face_recognition_model, classifier)
            # yield so other tasks have a chance to run
            await asyncio.sleep(0.01)
