import contextlib
import logging
import queue
import threading
from typing import Sequence

import PIL.ImageDraw
import PIL.ImageFont

import util
from aiy.vision.models import face_detection

from .classifier import NotFittedError, pickled_classifier
from .constants import DATA_DIR
from .dlib_wrapper import DlibWrapper
from .domain_types import Face, InputOutput, Person, Region
from .image import MyImage
from .preview import Preview

logger = logging.getLogger(__name__)
stopwatch = util.make_stopwatch(logger)

FONT_FILE = '/usr/share/fonts/truetype/freefont/FreeSans.ttf'
FONT = PIL.ImageFont.truetype(FONT_FILE, size=15)


def draw_face(draw: PIL.ImageDraw, face: Face):
    def draw_rectangle(x0, y0, x1, y1, border, fill=None, outline=None):
        assert border % 2 == 1
        for i in range(-border // 2, border // 2 + 1):
            draw.rectangle((x0 + i, y0 + i, x1 - i, y1 - i),
                           fill=fill, outline=outline)

    left = face.image_region.left
    top = face.image_region.top
    right = face.image_region.right
    bottom = face.image_region.bottom

    if face.person:
        text = '%s (dist: %.2f)' % (
            face.person.name, face.person.dist)
    else:
        text = 'Unrecognized'

    _, text_height = FONT.getsize(text)
    margin = 3
    text_bottom = bottom + margin + text_height + margin
    draw_rectangle(left, top, right, bottom, 3, outline='white')
    draw_rectangle(left, bottom, right,
                   text_bottom, 3, fill='white', outline='white')
    draw.text((left + 1 + margin, bottom + 1 + margin),
              text, font=FONT, fill='black')
    if face.labeled_landmarks:
        for _, points in face.labeled_landmarks.items():
            draw.line(points, fill='red', width=1)


def annotate(image: PIL.Image, data: InputOutput) -> PIL.Image:
    ':return annotated copy of PIL.Image'
    draw = PIL.ImageDraw.Draw(image)
    for face in data.faces:
        draw_face(draw, face)
    return image


# DlibWrapper.with_face_landmarks_model(
#     face_landmarks_model='shape_predictor_68_face_landmarks.dat')


class Processor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_info, exc_tb):
        pass

    def process(self, data: InputOutput):
        return data


class ProcessorChain(Processor):
    def __init__(self, *processors: Sequence[Processor]):
        self._processors = list(processors)
        self._exit_stack = contextlib.ExitStack()

    def __enter__(self):
        for processor in self._processors:
            self._exit_stack.enter_context(processor)
        return self

    def __exit__(self, exc_type, exc_info, exc_tb):
        self._exit_stack.close()

    def process(self, data: InputOutput):
        for processor in self._processors:
            result = processor.process(data)
            if not result:
                result = data
        return result


class PreviewProcessor(Preview, Processor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dirty = False

    def process(self, data: InputOutput):
        if data and data.faces:
            if self._dirty:
                self.clear()
            self._dirty = True
            for face in data.faces:
                draw_face(self.draw, face)
            self.update()
        else:
            if self._dirty:
                self._dirty = False
                self.clear()
                self.update()


class LandmarkProcessor(Processor):
    def __init__(self, _dlib: DlibWrapper):
        self._dlib = _dlib

    def _process(self, image: MyImage, face: Face):
        face.raw_landmarks = self._dlib.get_face_landmarks(
            image, face.image_region)
        face.labeled_landmarks = self._dlib.label_face_landmarks(
            face.raw_landmarks)

    def process(self, data: InputOutput):
        if not data:
            return
        for face in data.faces:
            self._process(data.image, face)


class DescriptorProcessor(Processor):
    def __init__(self, _dlib: DlibWrapper):
        self._dlib = _dlib

    def _process(self, image: MyImage, face: Face):
        if not face.raw_landmarks:
            logger.warning(
                'raw_landmarks not present - ensure that LandmarkProcessor ' +
                'is configured before DescriptorProcessor')
            return
        face.descriptor = self._dlib.get_face_descriptor(
            image, face.raw_landmarks)

    def process(self, data: InputOutput):
        if not data:
            return
        for face in data.faces:
            self._process(data.image, face)


class ClassifierProcessor(Processor):
    def __init__(self, classifier):
        self._classifier = classifier
        self._descriptor_person_id_pairs = []

    def __enter__(self):
        if self._classifier.person_count == 0:
            logger.info('classifier not fitted yet, unable to recognize any faces!')
        return self

    def __exit__(self, exc_type, exc_info, exc_tb):
        if self._descriptor_person_id_pairs:
            logger.info('fitting classifier over %d descriptor_person_id_pairs', len(
                self._descriptor_person_id_pairs))
            self._classifier.fit(self._descriptor_person_id_pairs)

    def _process(self, face: Face):
        if not face.descriptor:
            logger.warning(
                'descriptor not present - ensure that DescriptorProcessor ' +
                'is configured before ClassifierProcessor')
            return
        if face.person:
            # training mode
            self._descriptor_person_id_pairs.append(
                (face.descriptor, face.person.id))
        elif self._classifier.person_count > 0:
            # recognition mode
            person_id, dist = self._classifier.recognize_person(
                face.descriptor)
            face.person = Person(id=person_id, dist=dist)

    def process(self, data: InputOutput):
        if not data:
            return
        for face in data.faces:
            self._process(face)