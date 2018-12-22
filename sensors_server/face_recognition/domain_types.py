from typing import Any, Mapping, List, Tuple
from collections import namedtuple
from uuid import UUID
from .image import MyImage


class Region:
    def __init__(self, left: int, top: int, right: int, bottom: int):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def __str__(self):
        return 'Region(left={}, top={}, right={}, bottom={})'.format(self.left, self.top, self.right, self.bottom)
    __repr__ = __str__


class Person:
    def __init__(self, id: UUID, dist: float = None, name: str = None):
        self.id = id
        self.dist = dist
        self.name = name

    def __str__(self):
        return 'Person(id={}, dist={}, name={})'.format(str(self.id), self.dist, self.name)
    __repr__ = __str__


LabeledLandmarks = Mapping[str, List[Tuple[int, int]]]


class Face:
    def __init__(self, image_region: Region, raw_landmarks: Any = None,
                 labeled_landmarks: LabeledLandmarks = None, descriptor: List[float] = None,
                 face_score: float = None, joy_score: float = None, person: Person = None):
        self.image_region = image_region
        self.raw_landmarks = raw_landmarks
        self.labeled_landmarks = labeled_landmarks
        self.descriptor = descriptor
        self.face_score = face_score
        self.joy_score = joy_score
        self.person = person

    def __str__(self):
        return 'Face(image_region={}, raw_landmarks={}, labeled_landmarks={}, descriptor={}, face_score={}, joy_score={}, person={})'.format(
            self.image_region,
            '...' if self.raw_landmarks else 'None',
            '...' if self.labeled_landmarks else 'None',
            '...' if self.descriptor else 'None',
            self.face_score,
            self.joy_score,
            self.person)
    __repr__ = __str__


class InputOutput:
    def __init__(self, image: MyImage, faces: List[Face] = []):
        self.image = image
        self.faces = faces

    def __str__(self):
        return 'InputOutput(image={}, faces={})'.format(self.image, self.faces)

    __repr__ = __str__
