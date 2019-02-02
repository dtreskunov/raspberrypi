import asyncio
import contextlib
import logging
import os
import sys

from util import retry

logger = logging.getLogger(__name__)

_shutdown_env_key = __name__ + 'SHUTDOWN_SIGNALLED'
_shutdown_env_val = 'abracadabra'

def shutdown_signalled():
    return os.environ.get(_shutdown_env_key) == _shutdown_env_val

def _signal_shutdown():
    os.environ[_shutdown_env_key] = _shutdown_env_val

def _shutdown_handler(loop=None):
    if loop is None:
        loop = asyncio.get_event_loop()

    def get_pending_tasks():
        return [
            task for task in asyncio.Task.all_tasks(loop)
            if not task.cancelled()]

    async def wait_for_tasks(tasks):
        logger.debug('Waiting for %d tasks...', len(tasks))
        await asyncio.gather(*tasks)
        logger.debug('All awaited tasks finished, whew!')

    _signal_shutdown()

    for task in get_pending_tasks():
        logger.debug('Cancelling %s', repr(task))
        task.cancel()
    
    @retry(exceptions=(asyncio.CancelledError, RuntimeError), times=10)
    def run_until_pending_tasks_are_complete():
        loop.run_until_complete(wait_for_tasks(get_pending_tasks()))
    run_until_pending_tasks_are_complete()

def _monkey_patch_epoll_selector():
    '''
    File "/usr/local/lib/python3.5/selectors.py", line 436, in select
        timeout = math.ceil(timeout * 1e3) * 1e-3
    OverflowError: cannot convert float infinity to integer
    '''
    import selectors
    try:
        EpollSelector = selectors.EpollSelector
    except AttributeError:
        # no epoll on win32
        return
    import math
    orig_select = EpollSelector.select
    def select(self, timeout=None):
        if timeout == math.inf:
            logger.debug('Avoiding OverflowError: cannot convert float infinity to integer')
            timeout = None
        return orig_select(self, timeout)
    EpollSelector.select = select

def main(async_main):
    future = None
    if sys.platform == 'win32':
        # need to use ProactorEventLoop to support asyncio.subprocess_exec
        # https://docs.python.org/3/library/asyncio-eventloops.html#windows
        loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(loop)

        # https://stackoverflow.com/questions/27480967/why-does-the-asyncios-event-loop-suppress-the-keyboardinterrupt-on-windows
        async def wakeup():
            while True:
                await asyncio.sleep(1)
        loop.create_task(wakeup())
    else:
        loop = asyncio.get_event_loop()
        _monkey_patch_epoll_selector()

        def signal_handler(s):
            logger.info('Got %s, shutting down', s)
            loop.stop()

        import signal
        for s in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(s, lambda s=s: signal_handler(s))

    status = 0
    future = asyncio.ensure_future(async_main(), loop=loop)
    try:
        loop.run_until_complete(future)
    except KeyboardInterrupt:
        logger.info('Interrupted')
    except Exception as e:
        if str(e) != 'Event loop stopped before Future completed.':
            logger.error('Uncaught error, exiting: %s', e, exc_info=e)
            status = 1
    finally:
        _shutdown_handler(loop)
        # workaround for deadlock in Thread._wait_for_tstate_lock
        os._exit(status)
