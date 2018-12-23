import contextlib
import functools
import logging
from collections import namedtuple

import picamera
import util
from aiy.leds import Leds, PrivacyLed
from aiy.vision.inference import (CameraInference, InferenceEngine,
                                  InferenceException)
from aiy.vision.models import face_detection

from .image import MyImage
from .processor import Face, InputOutput, Region

logger = logging.getLogger(__name__)
stopwatch = util.make_stopwatch(logger)

CAPTURE_RESOLUTION = (820, 616)

Size = namedtuple('Size', ('w', 'h'))


def _reset_inference_engine() -> None:
    logger.info('attempting to reset InferenceEngine')
    with InferenceEngine() as engine:
        engine.reset()


@util.retry(_reset_inference_engine, InferenceException)
def _initialize_inference() -> CameraInference:
    '''
    One time, the process died without stopping inference, which
    made it impossible to start again. So if we get InferenceException
    when trying to initialize CameraInference, we retry after resetting
    the InferenceEngine.
    '''
    return CameraInference(face_detection.model())


@functools.lru_cache(maxsize=1)
def _get_scale(inference: Size, image: Size) -> float:
    if image.w / image.h != inference.w / inference.h:
        raise Exception('Inference result has different aspect ratio than camera capture: {}x{} to {}x{}'.format(
            inference.w, inference.h, image.w, image.h))
    return image.w / inference.w


def _get_image_region(aiy_bounding_box: tuple, inference: Size, image: Size) -> Region:
    'translate inference result into image coordinates'
    box_x, box_y, box_w, box_h = aiy_bounding_box
    scale = _get_scale(inference, image)
    adjust_factor = 0.9  # seems to improve face_landmarks
    w = box_w * adjust_factor
    h = box_h * adjust_factor
    x = box_x + (box_w - w)/2
    y = box_y + (box_h - h)/2
    return Region(
        left=max(0, int(scale * x)),
        top=max(0, int(scale * y)),
        right=min(image.w, int(scale * (x + w))),
        bottom=min(image.h, int(scale * (y + h)))
    )

class PiCameraInput:
    def __init__(self):
        self._stack = contextlib.ExitStack()
        self._camera = None
        self._inference = None
        self._person = None
    
    @property
    def camera(self):
        return self._camera
    
    @property
    def person(self):
        return self._person
    
    @person.setter
    def person(self, person):
        self._person = person

    def __enter__(self):
        # Forced sensor mode, 1640x1232, full FoV. See:
        # https://picamera.readthedocs.io/en/release-1.13/fov.html#sensor-modes
        # This is the resolution inference run on.
        with stopwatch('initialize camera'):
            self._camera = self._stack.enter_context(picamera.PiCamera(
                sensor_mode=4, resolution=CAPTURE_RESOLUTION))

        with stopwatch('initialize inference'):
            self._inference = self._stack.enter_context(_initialize_inference())

        leds = self._stack.enter_context(Leds())
        self._stack.enter_context(PrivacyLed(leds))
        return self
    
    def __exit__(self, *args, **kwargs):
        self._stack.close()
        self._camera = None
        self._inference = None

    def iterator(self):
        for inference_result in self._inference.run():
            aiy_faces = face_detection.get_faces(inference_result)
            if not aiy_faces:
                yield None

            # inference runs on the vision bonnet, which grabs images from the camera directly
            # we need to capture the image separately on the Raspberry in order to use dlib for face rec
            image = MyImage.capture(self._camera, use_video_port=True)

            inference_size = Size(
                w=inference_result.width, h=inference_result.height)
            image_size = Size(w=image.width, h=image.height)

            yield InputOutput(
                image=image,
                faces=[
                    Face(
                        image_region=_get_image_region(
                            aiy_face.bounding_box,
                            inference_size,
                            image_size
                        ),
                        face_score=aiy_face.face_score,
                        joy_score=aiy_face.joy_score,
                        person=self._person,
                    )
                    for aiy_face in aiy_faces
                ],
            )
