import asyncio
import itertools
import logging
import math

import aioconsole

from .modules import (AIOContextManager, AIOFilterModule, AIOLogger, AIOModule,
                      AIOOutput)
from .utils import main

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)-15s %(levelname)s [%(filename)s:%(lineno)d][%(process)d:%(threadName)s] %(message)s')
logger = logging.getLogger(__name__)


class TestGen(AIOModule, AIOOutput):
    def __init__(self, interval=1):
        AIOModule.__init__(self)
        AIOOutput.__init__(self)
        self._interval = interval
        self._count = itertools.count()

    @property
    def interval(self):
        return self._interval
    
    @interval.setter
    def interval(self, interval):
        self._interval = interval
    
    async def on_start(self):
        logger.info('%s: starting...', self.name)
        await asyncio.sleep(3)
        logger.info('%s: started', self.name)

    async def on_stop(self):
        logger.info('%s: stopping...', self.name)
        await asyncio.sleep(3)
        logger.info('%s: stopped', self.name)

    async def run(self):
        async with AIOContextManager(self.on_start, self.on_stop):
            while True:
                await asyncio.sleep(self.interval)
                n = next(self._count)
                logger.info('%s: generated %d', self.name, n)
                await self.output(n)

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

    async def control(self, gen):
        async def print_help():
            await aioconsole.aprint('Enter a number to control TestGen interval (empty will pause/resume)')
        await print_help()
        while True:
            s = await aioconsole.ainput()
            if len(s) == 0:
                if gen.running:
                    logger.info('%s: calling gen.stop()', self.name)
                    gen.stop()
                else:
                    logger.info('%s: calling gen.start()', self.name)
                    gen.start()
            elif s == 'r':
                logger.info('%s: calling gen.restart()', self.name)
                await gen.restart()
            else:
                try:
                    interval = float(s)
                    gen.interval = interval
                except ValueError:
                    await print_help()

    async def run(self):
        gen = TestGen()
        log1 = AIOLogger()
        log1.name = 'log1'
        log1.add_input(gen.add_output())
        log2 = AIOLogger()
        log2.name = 'log2'
        log2.add_input(gen.add_output())
        log3 = AIOLogger()
        log3.name = 'log3'
        log3.add_input(log2.add_output())
        log4 = AIOLogger()
        log4.name = 'log4'
        fil = TestFilter()
        fil.add_input(gen.add_output())
        log4.add_input(fil.add_output())
        log1.start()
        log2.start()
        log3.start()
        gen.start()

        asyncio.ensure_future(self.control(gen))
        await asyncio.sleep(math.inf)


async def async_main():
    driver = TestDriver()
    driver.start()
    await driver.wait()

if __name__=='__main__':
    main(async_main)
