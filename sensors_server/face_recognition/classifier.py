import contextlib
import logging
import os
import os.path
import pickle
import random

import numpy as np
import sklearn.exceptions
import sklearn.metrics
import sklearn.neighbors
import sklearn.preprocessing
from pony import orm

from .constants import DATA_DIR

logger = logging.getLogger(__name__)

THRESHOLD = 0.5


def _unpickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def _create_default_model():
    logger.info(
        "Initializing new KNeighborsClassifier(n_neighbors=1, metric='euclidean'")
    return sklearn.neighbors.KNeighborsClassifier(n_neighbors=1, metric='euclidean')


@contextlib.contextmanager
def pickled_classifier(pickle_file, create_default_model=_create_default_model):
    pickle_file = os.path.expanduser(pickle_file)
    classifier = None
    try:
        classifier = _unpickle(pickle_file)
    except Exception as e:
        logger.warning(
            'Unable to unpickle classifier from %s due to %s', pickle_file, e)
    if not classifier:
        logger.info('Creating empty classifier')
        classifier = Classifier(create_default_model(),
                                sklearn.preprocessing.LabelEncoder())
    try:
        yield classifier
    finally:
        try:
            _pickle(classifier, pickle_file)
        except Exception as e:
            logger.warning(
                'Unable to pickle classifier to %s due to %s', pickle_file, e)


class NotFittedError(RuntimeError):
    'Exception class to raise if classifier is used before fitting.'
    pass


class Classifier:
    def __init__(self, model, encoder):
        self._model = model
        self._encoder = encoder

    def fit(self, descriptor_person_id_pairs):
        if not descriptor_person_id_pairs:
            raise ValueError('No data provided')

        n = len(descriptor_person_id_pairs)
        random.shuffle(descriptor_person_id_pairs)

        # 128-dimensional face descriptor vectors
        X = np.array([pair[0] for pair in descriptor_person_id_pairs])

        # UUID ids from the Person table
        person_ids = np.array([pair[1] for pair in descriptor_person_id_pairs])
        # Numerical encoding of identities
        y = self._encoder.fit_transform(person_ids)

        # array([True, False, True, False, ...])
        train_idx = np.arange(n) % 2 == 0
        # array([False, True, False, True, ...])
        test_idx = np.arange(n) % 2 != 0

        X_train = X[train_idx]
        y_train = y[train_idx]

        self._model.fit(X_train, y_train)

        X_test = X[test_idx]
        y_test = y[test_idx]
        if X_test.shape[0] > 0:
            accuracy_score = sklearn.metrics.accuracy_score(
                y_test, self._model.predict(X_test))
            logger.info('Model has %.2f accuracy (%d training and %d test samples)',
                        accuracy_score,
                        X_train.shape[0], X_test.shape[0])
        else:
            logger.warning(
                'No test data - will be unable to calculate model accuracy')
            return

    def recognize_person(self, face_descriptor):
        try:
            distance = self._model.kneighbors([face_descriptor])[0][0][0]
            prediction = self._model.predict([face_descriptor])
            person_id = self._encoder.inverse_transform(prediction)[0]
        except sklearn.exceptions.NotFittedError as e:
            raise NotFittedError(e)

        if distance < THRESHOLD:
            logger.info('%s found at a distance of %.2f (threshold %.2f)',
                        person_id, distance, THRESHOLD)
            return person_id, distance
        else:
            logger.info('no match found within threshold of %.2f; nearest neighbor is %s at a distance of %.2f',
                        person_id, THRESHOLD, distance)
            return None, distance
