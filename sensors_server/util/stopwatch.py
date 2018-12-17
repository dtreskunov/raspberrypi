import time
from contextlib import contextmanager


def make_stopwatch(logger):
    @contextmanager
    def stopwatch(message):
        try:
            begin = time.monotonic()
            yield
        finally:
            end = time.monotonic()
            logger.debug('%s: done in %.3fs', message, end - begin)

    return stopwatch
