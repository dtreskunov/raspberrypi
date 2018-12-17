import logging
import os.path
import random

import numpy
import rtree.core
import rtree.index
from pony import orm

from .constants import DATA_DIR
from .entities import Person

logger = logging.getLogger(__name__)

FACE_DESCRIPTOR_DIMENSIONS = 128
THRESHOLD = 0.6


class Classifier:
    def __init__(self):
        self._initialize_index()
        self._initialize_random_name()

    def __enter__(self):
        return self

    def __exit__(self, _, _):
        pass

    @orm.db_session
    def _initialize_index(self):
        def generator():
            for person in Person.select():
                yield (person.id, person.face_descriptor)
        is_empty = orm.count(c for c in Person) == 0
        properties = rtree.index.Property()
        properties.dimension = FACE_DESCRIPTOR_DIMENSIONS
        if is_empty:
            self._idx = rtree.index.Index(properties=properties)
        else:
            self._idx = rtree.index.Index(
                generator(), properties=properties)

    def _initialize_random_name(self):
        def read_lines(filename):
            with open(os.path.join(os.path.dirname(__file__), filename), 'r') as file:
                return [line.strip() for line in file.readlines()]
        self._adjectives = read_lines('adjectives.txt')
        self._nouns = read_lines('nouns.txt')

    def _random_name(self):
        return random.choice(self._adjectives).title() + ' ' + random.choice(self._nouns).title()

    def _insert(self, person):
        self._idx.insert(person.id, person.avg_face_descriptor)

    def _delete(self, person):
        try:
            self._idx.delete(person.id, self._idx.bounds)
        except rtree.core.RTreeError:
            # raised if index is empty - bounds are inverted in this case
            pass

    def _update(self, person):
        self._delete(person)
        self._insert(person)

    @orm.db_session
    def recognize_person(self, face_descriptor):
        person = None
        is_new = False
        dist = 0.0

        ids = self._idx.nearest(face_descriptor, 1)
        if ids:
            person = Person[ids[0]]
            dist = numpy.linalg.norm(
                numpy.array(person.avg_face_descriptor) - numpy.array(face_descriptor))
            if dist > THRESHOLD:
                person = None
            else:
                logger.debug(
                    'found a person within dist=%.2f: name=%s, id=%s', dist, person.name, person.id)
                person.avg_face_descriptor = numpy.average(
                    [person.avg_face_descriptor, face_descriptor],
                    weights=[person.n_samples, 1],
                    axis=0).tolist()
                person.n_samples += 1
                self._update(person)
        if not person:
            person = Person(
                avg_face_descriptor=face_descriptor,
                n_samples=1,
                name=self._random_name())
            self._insert(person)
            is_new = True
            logger.info('no matching identities found - a new one was created with name=%s, id=%s',
                        person.name, person.id)
        return (person, is_new, dist)
