import logging
import os.path
import random

import numpy
from pony import orm

from .constants import DATA_DIR
from .entities import DetectedFace, Person, db_transaction

logger = logging.getLogger(__name__)

THRESHOLD = 0.6


class Classifier:
    def __init__(self):
        self._initialize_random_name()

    def _initialize_random_name(self):
        def read_lines(filename):
            with open(os.path.join(os.path.dirname(__file__), filename), 'r') as file:
                return [line.strip() for line in file.readlines()]
        self._adjectives = read_lines('adjectives.txt')
        self._nouns = read_lines('nouns.txt')

    def _random_name(self):
        return random.choice(self._adjectives).title() + ' ' + random.choice(self._nouns).title()

    @db_transaction
    def recognize_person(self, new_face: DetectedFace):
        person = None
        is_new = False
        dist = 0.0

        new_descriptor = numpy.array(new_face.descriptor)

        # Compute distance from new_face to each of previously seen faces.
        # This is very brute-force - consider using scikit-learn KNN algorithms
        # if this proves to be slow.
        # See https://github.com/ageitgey/face_recognition/blob/master/examples/face_recognition_knn.py
        def generator():
            for seen_face in DetectedFace.select():
                if seen_face == new_face:
                    continue
                dist = numpy.linalg.norm(
                    numpy.array(seen_face.descriptor) - new_descriptor)
                yield (seen_face, dist)

        closest_seen_face, dist = sorted(
            generator(), key=lambda pair: pair[1])[0]
        if closest_seen_face and dist < THRESHOLD and closest_seen_face.person is not None:
            person = closest_seen_face.person
            new_face.person = person
            logger.debug(
                'match found: (name=%s, id=%s) is within dist=%.2f: ', person.name, person.id, dist)
        else:
            person = Person(name=self._random_name())
            is_new = True
            new_face.person = person
            logger.info('new person (name=%s, id=%s) created',
                        person.name, person.id)
        return (person, is_new, dist)
