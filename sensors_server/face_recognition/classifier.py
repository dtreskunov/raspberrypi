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
from .entities import (DetectedFace, Person, db_transaction,
                       get_descriptor_person_id_pairs)

logger = logging.getLogger(__name__)

THRESHOLD = 0.5


def _unpickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _pickle(obj, path):
    os.makedirs(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def _create_default_model():
    logger.info(
        "Initializing new KNeighborsClassifier(n_neighbors=1, metric='euclidean'")
    return sklearn.neighbors.KNeighborsClassifier(n_neighbors=1, metric='euclidean')


@contextlib.contextmanager
def pickled_classifier(pickle_file, create_default_model=_create_default_model):
    pickle_file = os.path.expanduser(pickle_file)
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


class Classifier:
    def __init__(self, model, encoder):
        self._model = model
        self._encoder = encoder

    @db_transaction
    def fit(self):
        descriptor_person_id_pairs = get_descriptor_person_id_pairs()
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
        if X_train.shape[0] == 0:
            raise Exception(
                'Unable to fit model - identify some people first!')

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

    @db_transaction
    def recognize_person(self, face: DetectedFace):
        try:
            distance = self._model.kneighbors([face.descriptor])[0][0][0]
            prediction = self._model.predict([face.descriptor])
            person_id = self._encoder.inverse_transform(prediction)[0]
        except sklearn.exceptions.NotFittedError:
            return None

        person = Person[person_id]

        if distance < THRESHOLD:
            logger.info('%s found at a distance of %.2f (threshold %.2f)',
                        person, distance, THRESHOLD)
            face.person = person
            return person
        else:
            logger.info('no match found within threshold of %.2f; nearest neighbor is %s at a distance of %.2f',
                        person, THRESHOLD, distance)
            return None
