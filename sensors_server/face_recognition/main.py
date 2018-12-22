#!/usr/bin/env python3

# TODO: unable to run as script: ValueError: Attempted relative import in non-package
# WORKAROUND: run as `python3 -m face_recognition.main -h` from root Python directory

import argparse
import contextlib
import logging

from util import make_stopwatch

from .dlib_wrapper import DlibWrapper
from .picamera_input import PiCameraInput
from .preview import Preview
from .processor import (DescriptorProcessor, LandmarkProcessor,
                        PreviewProcessor, ProcessorChain, SkipIfBusyProcessor)

logger = logging.getLogger(__name__)
stopwatch = make_stopwatch(logger)

def main(args):
    with contextlib.ExitStack() as exit_stack:
        processors = []
        pi_camera_input = exit_stack.enter_context(PiCameraInput())
        if args.landmarks:
            processors.append(
                LandmarkProcessor(
                    DlibWrapper.with_face_landmarks_model(args.face_landmarks_model)))
        if args.descriptor:
            processors.append(
                SkipIfBusyProcessor(
                    DescriptorProcessor(
                        DlibWrapper.with_face_landmarks_model(args.face_landmarks_model))))
        if args.preview:
            processors.append(
                PreviewProcessor(camera=pi_camera_input.camera))

        processor_chain = exit_stack.enter_context(ProcessorChain(*processors))
        for data in pi_camera_input.iterator():
            with stopwatch('processor chain'):
                logger.info(processor_chain.process(data))


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
        '--descriptor', help='extract facial descriptor (slow, but needed for recognition)', action='store_true', default=True)
    parser.add_argument(
        '--preview', help='overlay data on top of live camera feed', action='store_true', default=True)

    args = parser.parse_args()
    if args.debug:
        import ptvsd
        address = ('0.0.0.0', 5678)
        ptvsd.enable_attach(address=address)
        print('Waiting for debugger on {}...'.format(address))
        ptvsd.wait_for_attach()

    logging.basicConfig(level=getattr(logging, args.loglevel))
    main(args)
