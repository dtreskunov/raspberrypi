import bz2
import functools
import io
import logging
import os.path
import urllib.request

import dlib
import util

from .constants import DATA_DIR
from .image import MyImage
from .domain_types import Region

logger = logging.getLogger(__name__)
stopwatch = util.make_stopwatch(logger)


class DlibWrapper:
    @staticmethod
    def _download_bz2(url, dest):
        'returns normalized destination path'
        dest = os.path.realpath(os.path.expanduser(dest))
        if not os.path.isfile(dest):
            with stopwatch('downloading from %s and extracting into %s' % (url, dest)):
                resp = urllib.request.urlopen(url)
                bz2_file = bz2.BZ2File(io.BytesIO(resp.read()))
                with open(dest, 'wb') as dest_file:
                    dest_file.write(bz2_file.read())
        return dest

    @staticmethod
    @functools.lru_cache(maxsize=2)
    def with_face_landmarks_model(face_landmarks_model):
        with stopwatch('downloading dlib shape predictor model'):
            shape_predictor_model_path = DlibWrapper._download_bz2(
                'http://dlib.net/files/{}.bz2'.format(face_landmarks_model),
                '{}/{}'.format(DATA_DIR, face_landmarks_model))
        with stopwatch('initializing dlib shape predictor'):
            shape_predictor = dlib.shape_predictor(shape_predictor_model_path)

        with stopwatch('downloading dlib face recognition model'):
            face_recognition_model_path = DlibWrapper._download_bz2(
                'http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2',
                '{}/dlib_face_recognition_resnet_model_v1.dat'.format(DATA_DIR))
        with stopwatch('initializing dlib face recognition model'):
            face_recognition_model = dlib.face_recognition_model_v1(
                face_recognition_model_path)

        def label_face_landmarks_func(face_landmarks):
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

        return DlibWrapper(shape_predictor, face_recognition_model, label_face_landmarks_func)

    def __init__(self, shape_predictor, face_recognition_model, label_face_landmarks_func):
        self._shape_predictor = shape_predictor
        self._face_recognition_model = face_recognition_model
        self._label_face_landmarks_func = label_face_landmarks_func

    @property
    def label_face_landmarks(self):
        return self._label_face_landmarks_func

    def get_face_landmarks(self, image: MyImage, region: Region):
        with stopwatch('get_face_landmarks'):
            return self._shape_predictor(
                image.numpy_array,
                dlib.rectangle(
                    left=region.left,
                    top=region.top,
                    right=region.right,
                    bottom=region.bottom))

    def get_face_descriptor(self, image: MyImage, face_landmarks):
        '''
        Compute the 128D vector that describes the face in image.
        In general, if two face descriptor vectors have a Euclidean
        distance between them less than 0.6 then they are from the same
        person, otherwise they are from different people.
        '''
        with stopwatch('get_face_descriptor'):
            face_descriptor = self._face_recognition_model.compute_face_descriptor(
                image.numpy_array, face_landmarks)
            return list(face_descriptor)
