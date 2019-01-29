import asyncio
import itertools
import logging
import math

import aioconsole

from .modules import AIOLogger, AIOModule, AIOProducerModule
from .utils import main

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)-15s %(levelname)s [%(filename)s:%(lineno)d][%(process)d:%(threadName)s] %(message)s')
logger = logging.getLogger(__name__)


class TestGen(AIOProducerModule):
    def __init__(self, config={'interval': 1}):
        super(TestGen, self).__init__(config=config)
        self._count = itertools.count()

    async def on_start(self):
        logger.info('%s: starting...', self.name)
        await asyncio.sleep(3)
        logger.info('%s: started', self.name)

    async def on_stop(self):
        logger.info('%s: stopping...', self.name)
        await asyncio.sleep(3)
        logger.info('%s: stopped', self.name)

    async def item(self):
        await asyncio.sleep(self.config['interval'])
        n = next(self._count)
        logger.info('%s: generated %d', self.name, n)
        return n


class TestDriver(AIOModule):
    async def control(self, gen):
        async def print_help():
            await aioconsole.aprint('Enter a number to control TestGen interval (empty will pause/resume)')
        await print_help()
        while True:
            s = await aioconsole.ainput()
            if len(s) == 0:
                gen.running = not gen.running
            else:
                try:
                    interval = float(s)
                    gen.config['interval'] = interval
                except ValueError:
                    await print_help()

    async def run(self):
        gen = TestGen()
        log1 = AIOLogger()
        log1.name = 'log1'
        log1.sources = [gen.mkchannel()]
        log2 = AIOLogger()
        log2.name = 'log2'
        log2.sources = [gen.mkchannel()]
        log3 = AIOLogger()
        log3.name = 'log3'
        log3.sources = [log2.mkchannel()]
        asyncio.ensure_future(gen.run())
        asyncio.ensure_future(log1.run())
        asyncio.ensure_future(log2.run())
        asyncio.ensure_future(log3.run())
        asyncio.ensure_future(self.control(gen))
        await asyncio.sleep(math.inf)


async def async_main():
    driver = TestDriver()
    await asyncio.ensure_future(driver.run())

if __name__=='__main__':
    main(async_main)
