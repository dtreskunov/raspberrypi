import contextlib
import logging
import time

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
                    func(*args, **kwargs)
                    return
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
