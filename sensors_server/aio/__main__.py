import asyncio
import itertools
import logging
import math

import aioconsole

from .modules import (AIOContextManager, AIOFilterModule, AIOLogger, AIOModule,
                      AIOOutput, AIOProducerModule)
from .utils import main

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)-15s %(levelname)s [%(filename)s:%(lineno)d][%(process)d:%(threadName)s] %(message)s')
logger = logging.getLogger(__name__)


class TestGen(AIOProducerModule):
    def __init__(self, interval=1):
        AIOProducerModule.__init__(self)
        self._interval = interval
        self._count = itertools.count()

    @property
    def interval(self):
        return self._interval
    
    @interval.setter
    def interval(self, interval):
        self._interval = interval
    
    async def on_start(self):
        logger.info('TestGen: starting...')
        await asyncio.sleep(3)
        logger.info('TestGen: started')

    async def on_stop(self):
        logger.info('TestGen: stopping...')
        await asyncio.sleep(3)
        logger.info('TestGen: stopped')

    async def produce(self):
        await asyncio.sleep(self.interval)
        n = next(self._count)
        logger.info('TestGen: generated %d', n)
        return n

class TestFilter(AIOFilterModule):
    async def on_start(self):
        logger.info('TestFilter starting...')
        await asyncio.sleep(0.5)
        logger.info('TestFilter started')
    async def on_stop(self):
        logger.info('TestFilter stopping...')
        await asyncio.sleep(0.5)
        logger.info('TestFilter stopped')
    async def filter(self, item):
        await asyncio.sleep(0.1)
        if item % 2 == 0:
            logger.info('Got even number %s, squaring it', item)
            return item ** 2
        else:
            logger.info('Got odd number %s, skipping it', item)

class TestDriver(AIOModule):
    def __init__(self):
        AIOModule.__init__(self)

    async def run(self):
        gen = TestGen().start()
        log1 = AIOLogger('log1').start()
        log1.add_input(gen.add_output())
        log2 = AIOLogger('log2').start()
        log2.add_input(gen.add_output())
        log3 = AIOLogger('log3').start()
        fil = TestFilter()
        fil.add_input(gen.add_output())
        log3.add_input(fil.add_output())

        async def control():
            async def print_help():
                await aioconsole.aprint('''
                    Enter a number to set TestGen interval,
                    <Ret> to stop/start TestGen,
                    <r> to restart TestGen,
                    <f> to stop/start TestFilter,
                    <l> to stop/start log1
                ''')
            await print_help()
            while True:
                s = await aioconsole.ainput()
                if len(s) == 0:
                    if gen.running:
                        logger.info('TestDriver: calling gen.stop()')
                        gen.stop()
                    else:
                        logger.info('TestDriver: calling gen.start()')
                        gen.start()
                elif s == 'r':
                    logger.info('TestDriver: calling gen.restart()')
                    await gen.restart()
                elif s == 'f':
                    if fil.running:
                        logger.info('TestDriver: calling fil.stop()')
                        fil.stop()
                    else:
                        logger.info('TestDriver: calling fil.start()')
                        fil.start()
                elif s == 'l':
                    if log1.running:
                        logger.info('TestDriver: calling log1.stop()')
                        log1.stop()
                    else:
                        logger.info('TestDriver: calling log1.start()')
                        log1.start()
                else:
                    try:
                        interval = float(s)
                        gen.interval = interval
                    except ValueError:
                        await print_help()
        asyncio.ensure_future(control())
        await asyncio.sleep(math.inf)


async def async_main():
    await TestDriver().start().wait()

if __name__=='__main__':
    main(async_main)
