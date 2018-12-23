#!/usr/bin/env python3

# TODO: unable to run as script: ValueError: Attempted relative import in non-package
# WORKAROUND: run as `python3 -m face_recognition.main -h` from root Python directory

import argparse
import contextlib
import logging
import warnings

from util import make_stopwatch

from .classifier import Classifier
from .database import Person as DBPerson
from .database import db_connection, get_descriptor_person_id_pairs, get_or_create_person_by_name
from .dlib_wrapper import DlibWrapper
from .domain_types import Person
from .picamera_input import PiCameraInput
from .preview import Preview
from .processor import (ClassifierProcessor, DatabaseProcessor,
                        DescriptorProcessor, LandmarkProcessor,
                        PreviewProcessor, ProcessorChain)

logger = logging.getLogger(__name__)
stopwatch = make_stopwatch(logger)


def lookup_person(name):
    db_person = get_or_create_person_by_name(name)
    return Person(
        id=db_person.id,
        name=db_person.name,
        dist=0.0)


def main(args):
    logger.debug('entering main(%s)', args)
    with contextlib.ExitStack() as exit_stack:
        pi_camera_input = exit_stack.enter_context(PiCameraInput())

        processors = []
        if args.landmarks:
            processors.append(
                LandmarkProcessor(
                    DlibWrapper.with_face_landmarks_model(args.face_landmarks_model)))
        if args.descriptor:
            processors.append(
                DescriptorProcessor(
                    DlibWrapper.with_face_landmarks_model(args.face_landmarks_model)))
        if args.use_db:
            exit_stack.enter_context(
                db_connection(**args.db_connection_params))
            if args.name:
                pi_camera_input.person = lookup_person(args.name)
            if args.mode == 'recognition':
                classifier = Classifier()
                with stopwatch('fitting classifier'):
                    classifier.fit(get_descriptor_person_id_pairs())
                logger.info(
                    'classifier fit to recognize %d person(s)', classifier.person_count)
                processors.append(ClassifierProcessor(classifier))
            processors.append(DatabaseProcessor(mode=args.mode))
        if args.preview:
            processors.append(
                PreviewProcessor(camera=pi_camera_input.camera))

        processor_chain = exit_stack.enter_context(ProcessorChain(*processors))
        for data in pi_camera_input.iterator():
            with stopwatch('processor chain'):
                data = processor_chain.process(data)
                if data:
                    logger.info(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Face recognition CLI utility')
    parser.add_argument(
        '-d', '--debug', help='enable remote debugger compatible with VS Code', action='store_true')
    parser.add_argument(
        '--loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
    parser.add_argument(
        '--face-landmarks-model', help='model to download from http://dlib.net/files/ (sans .bz2 extension)',
        choices=['shape_predictor_5_face_landmarks.dat',
                 'shape_predictor_68_face_landmarks.dat'],
        default='shape_predictor_68_face_landmarks.dat')
    parser.add_argument(
        '--landmarks', help='extract facial landmarks', action='store_true', default=True)
    parser.add_argument(
        '--descriptor', help='extract facial descriptor (slow, but needed for recognition) (implies --landmarks)',
        action='store_true', default=True)
    parser.add_argument(
        '--name', help='name of person appearing in the input stream (implies --mode=training)')
    parser.add_argument(
        '--mode', help='"training" implies --use-db', choices=['recognition', 'training'], default='recognition')
    parser.add_argument(
        '--use-db', help='implies --descriptor', action='store_true', default=True)
    parser.add_argument(
        '--db-connection-params', default='provider=sqlite,filename=~/.face_recognition/data.sqlite')
    parser.add_argument(
        '--preview', help='overlay data on top of live camera feed', action='store_true', default=True)

    args = parser.parse_args()
    if args.debug:
        import ptvsd
        address = ('0.0.0.0', 5678)
        ptvsd.enable_attach(address=address)
        print('Waiting for debugger on {}...'.format(address))
        ptvsd.wait_for_attach()

    if args.name and args.mode != 'training':
        print('--mode=training implied by --name')
        args.mode = 'training'
    if args.mode == 'training' and not args.use_db:
        print('--use-db implied by --mode=training')
        args.use_db = True
    if args.use_db and not args.descriptor:
        print('--descriptor implied by --use-db')
        args.descriptor = True
    if args.descriptor and not args.landmarks:
        print('--landmarks implied by --descriptor')
        args.landmarks = True

    args.db_connection_params = dict(
        (kv.split('=') for kv in args.db_connection_params.split(',')))

    logging.basicConfig(level=getattr(logging, args.loglevel))
    logging.captureWarnings(True)
    warnings.simplefilter('default', DeprecationWarning)
    main(args)
