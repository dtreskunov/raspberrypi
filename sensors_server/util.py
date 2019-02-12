import argparse
import asyncio
import asyncio.subprocess
import contextlib
import distutils.util
import functools
import importlib
import logging
import re
import sys
import threading
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


def serialize():
    '''
    Decorator that makes sure that the decorated function (regular or async) is executed
    in serial, i.e. there will be no concurrent execution.
    '''
    def decorator(func):
        func_is_async = asyncio.iscoroutinefunction(func)
        if func_is_async:
            lock = asyncio.Lock()
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                async with lock:
                    return await func(*args, **kwargs)
            return wrapper
        else:
            lock = threading.Lock()
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with lock:
                    return func(*args, **kwargs)
            return wrapper
    return decorator


def retry(remedy_func=None, exceptions=Exception, times=1, *remedy_func_args):
    exceptions = exceptions if type(exceptions) == list else (exceptions,)

    def decorator(func):
        remedy_func_is_async = asyncio.iscoroutinefunction(remedy_func)
        func_is_async = asyncio.iscoroutinefunction(func)
        if remedy_func_is_async != func_is_async:
            raise ValueError(
                'Wrapped and remedy functions must both be either plain or async')
        if func_is_async:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                _times = times
                while True:
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        if not _times > 0 or not isinstance(e, exceptions):
                            raise
                        logger.debug(
                            'Caught %s, will retry %d time(s)', repr(e), _times)
                        _times -= 1
                        if remedy_func:
                            await remedy_func(*remedy_func_args)
            return wrapper
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                _times = times
                while True:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if not _times > 0 or not isinstance(e, exceptions):
                            raise
                        logger.debug(
                            'Caught %s, will retry %d time(s)', repr(e), _times)
                        _times -= 1
                        if remedy_func:
                            remedy_func(*remedy_func_args)
            return wrapper

    return decorator


def lazy_getter(func):
    cache = []

    def getter():
        if not cache:
            cache.append(func())
        return cache[0]
    return getter


@serialize()
async def pip_install(pip_package):
    log_prefix = 'pip install --user {}'.format(pip_package)
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        '-m', 'pip', 'install', '--user', pip_package,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    while True:
        data = await process.stdout.readline()
        if len(data) == 0:
            break
        line = data.decode().rstrip()
        logger.info('%s: %s', log_prefix, line)
    await process.wait()
    if process.returncode != 0:
        raise RuntimeError('{} exited with returncode {}'.format(
            log_prefix, process.returncode))


async def do_imports(pip_package, *modules):
    @retry(pip_install, ImportError, 1, pip_package)
    async def _do_imports():
        for module in modules:
            importlib.import_module(module)
    await _do_imports()


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

    def _setup(self):
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
        return args

    def run(self):
        self.main(self._setup())

    def main(self, args):
        pass

    @staticmethod
    def bool(v):
        return bool(distutils.util.strtobool(v))
