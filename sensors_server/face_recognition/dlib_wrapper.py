import bz2
import io
import logging
import os.path
import urllib.request

import dlib
import util

from .constants import DATA_DIR
from .image import MyImage
from .domain_types import Face, Region

logger = logging.getLogger(__name__)
stopwatch = util.make_stopwatch(logger)


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


class DlibCnnFaceDetector:
    def __init__(self):
        with stopwatch('downloading dlib face detection model'):
            face_detection_model_path = _download_bz2(
                'http://dlib.net/files/mmod_human_face_detector.dat.bz2',
                '{}/mmod_human_face_detector.dat.bz2'.format(DATA_DIR))
        with stopwatch('initializing dlib CNN face detection model'):
            self._face_detection_model = dlib.cnn_face_detection_model_v1(
                face_detection_model_path)

    def get_faces(self, image: MyImage, region: Region):
        cropped_numpy_array = image.numpy_array[
            region.top:region.bottom,
            region.left:region.right,
            :]
        with stopwatch('get_faces (CNN)'):
            mmod_rects = self._face_detection_model(cropped_numpy_array)
        return [
            Face(
                image_region=Region(
                    left=region.left + mmod_rect.rect.left(),
                    top=region.top + mmod_rect.rect.top(),
                    right=region.left + mmod_rect.rect.right(),
                    bottom=region.top + mmod_rect.rect.bottom()
                ),
                face_score=mmod_rect.confidence,
            )
            for mmod_rect in mmod_rects]


class DlibHogFaceDetector:
    def __init__(self):
        self._face_detector = dlib.get_frontal_face_detector()

    def get_faces(self, image: MyImage, region: Region):
        cropped_numpy_array = image.numpy_array[
            region.top:region.bottom,
            region.left:region.right,
            :]
        with stopwatch('get_faces (HOG)'):
            rectangles = self._face_detector(cropped_numpy_array)
        return [
            Face(
                image_region=Region(
                    left=region.left + rect.left(),
                    top=region.top + rect.top(),
                    right=region.left + rect.right(),
                    bottom=region.top + rect.bottom()
                )
            )
            for rect in rectangles]


class DlibFaceLandmarksExtractor:
    def __init__(self, face_landmarks_model):
        self._face_landmarks_model = face_landmarks_model
        with stopwatch('downloading dlib shape predictor model'):
            shape_predictor_model_path = _download_bz2(
                'http://dlib.net/files/{}.bz2'.format(face_landmarks_model),
                '{}/{}'.format(DATA_DIR, face_landmarks_model))
        with stopwatch('initializing dlib shape predictor'):
            self._shape_predictor = dlib.shape_predictor(
                shape_predictor_model_path)

    def get_face_landmarks(self, image: MyImage, region: Region):
        with stopwatch('get_face_landmarks'):
            return self._shape_predictor(
                image.numpy_array,
                dlib.rectangle(
                    left=region.left,
                    top=region.top,
                    right=region.right,
                    bottom=region.bottom))

    def label_face_landmarks(self, face_landmarks):
        '''
        :return: A list of dicts of face feature locations (eyes, nose, etc)
        '''
        points = [(p.x, p.y) for p in face_landmarks.parts()]

        # For a definition of each point index, see https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
        if self._face_landmarks_model == 'shape_predictor_68_face_landmarks.dat':
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
        elif self._face_landmarks_model == 'shape_predictor_5_face_landmarks.dat':
            return {
                "nose_tip": [points[4]],
                "left_eye": points[2:4],
                "right_eye": points[0:2],
            }
        else:
            logger.warning(
                'unable to label face landmarks - unrecognized model %s', self._face_landmarks_model)
            return {}


class DlibFaceDescripitorExtractor:
    def __init__(self):
        with stopwatch('downloading dlib face recognition model'):
            face_recognition_model_path = _download_bz2(
                'http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2',
                '{}/dlib_face_recognition_resnet_model_v1.dat'.format(DATA_DIR))
        with stopwatch('initializing dlib face recognition model'):
            self._face_recognition_model = dlib.face_recognition_model_v1(
                face_recognition_model_path)

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
