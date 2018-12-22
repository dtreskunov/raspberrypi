#!/usr/bin/env python3

# TODO: unable to run as script: ValueError: Attempted relative import in non-package
# WORKAROUND: run as `python3 -m face_recognition.main -h` from root Python directory

import argparse
import contextlib
import logging

from .picamera_input import PiCameraInput
from .preview import Preview
from .processor import PreviewProcessor


def main(args):
    with contextlib.ExitStack() as exit_stack:
        pi_camera_input = exit_stack.enter_context(PiCameraInput())
        preview_processor = exit_stack.enter_context(
            PreviewProcessor(camera=pi_camera_input.camera))
        for data in pi_camera_input.iterator():
            preview_processor.process(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Face recognition CLI utility')
    parser.add_argument(
        '-d', '--debug', help='enable remote debugger compatible with VS Code', action='store_true')
    parser.add_argument(
        '--loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')

    args = parser.parse_args()
    if args.debug:
        import ptvsd
        address = ('0.0.0.0', 5678)
        ptvsd.enable_attach(address=address)
        print('Waiting for debugger on {}...'.format(address))
        ptvsd.wait_for_attach()

    logging.basicConfig(level=getattr(logging, args.loglevel))
    main(args)
