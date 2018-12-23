import argparse
import contextlib
import logging
import time
import warnings

logger = logging.getLogger(__name__)


def make_stopwatch(logger):
    @contextlib.contextmanager
    def stopwatch(message):
        try:
            begin = time.monotonic()
            yield
        finally:
            end = time.monotonic()
            logger.debug('%s: done in %.3fs', message, end - begin)

    return stopwatch


def retry(remedy_func=None, exceptions=Exception, times=1):
    exceptions = exceptions if type(exceptions) == list else (exceptions,)

    def decorator(func):
        def wrapper(*args, **kwargs):
            _times = times
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if not _times > 0 or not isinstance(e, exceptions):
                        raise
                    logger.info(
                        'Caught exception "%s", will retry %d time(s)', e, _times)
                    _times -= 1
                    if remedy_func:
                        remedy_func()
        return wrapper

    return decorator


def lazy_getter(func):
    cache = []

    def getter():
        if not cache:
            cache.append(func())
        return cache[0]
    return getter


class CLI:
    def __init__(self, parser=None):
        if not parser:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser.add_argument(
                '-d', '--debug', help='enable remote debugger compatible with VS Code', action='store_true')
            parser.add_argument(
                '--loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
        self.parser = parser

    def run(self):
        args = self.parser.parse_args()
        if args.debug:
            import ptvsd
            address = ('0.0.0.0', 5678)
            ptvsd.enable_attach(address=address)
            print('Waiting for debugger on {}...'.format(address))
            ptvsd.wait_for_attach()
        logging.basicConfig(level=getattr(logging, args.loglevel))
        logging.captureWarnings(True)
        warnings.simplefilter('default', DeprecationWarning)
        self.main(args)

    def main(self, args):
        pass
