import abc
import asyncio
import logging

import aiochannel

logger = logging.getLogger(__name__)


class AIOShutdownContextManager:
    def __init__(self, *coros):
        self._coros = coros

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc, tb):
        async def shutdown_coro():
            await asyncio.gather(*self._coros)
        loop = asyncio.get_event_loop()
        loop.create_task(shutdown_coro())


class AIOModule(abc.ABC):
    def __init__(self, name=None, config=None):
        self._name = name or type(self).__name__
        self._config = config

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = config

    @abc.abstractmethod
    async def run(self):
        pass


class AIOProducerModule(AIOModule):
    def __init__(self, name=None, config=None):
        super(AIOProducerModule, self).__init__(name, config)
        self._running = asyncio.Event()
        self._running.set()
        self._channels = []

    @property
    def running(self):
        return self._running.is_set()

    @running.setter
    def running(self, running):
        logger.debug('Changing running state of %s to %s', self.name, running)
        if running:
            self._running.set()
        else:
            self._running.clear()

    def mkchannel(self, maxsize=1):
        channel = aiochannel.Channel(maxsize=maxsize)
        self._channels.append(channel)
        return channel

    async def shutdown(self):
        pass

    async def _put_item_into_channel(self, item, channel):
        await channel.put(item)

    async def run(self):
        with AIOShutdownContextManager(self.shutdown()):
            while True:
                await self._running.wait()
                item = await self.item()
                await asyncio.gather(*[
                    self._put_item_into_channel(item, channel)
                    for channel in self._channels
                ])

    @abc.abstractmethod
    async def item(self):
        pass


class AIOConsumerModule(AIOProducerModule):
    'Consumes from multiple channels and produces filtered output'
    def __init__(self, name=None, config=None):
        super(AIOConsumerModule, self).__init__(name, config)
        self._sources = []
        self._sources_combined = aiochannel.Channel(maxsize=1)
        self._putters = []

    @property
    def sources(self):
        return self._sources

    @sources.setter
    def sources(self, sources):
        self._sources = sources
        for putter in self._putters:
            putter.cancel()

        async def put_into_sources_combined(source):
            async for item in source:
                await self._sources_combined.put(item)
        self._putters = [asyncio.ensure_future(put_into_sources_combined(source))
                         for source in self._sources]

    async def item(self):
        source_item = await self._sources_combined.get()
        consumed_item = await self.consume(source_item)
        return consumed_item

    @abc.abstractmethod
    async def consume(self, item):
        pass


class AIOLogger(AIOConsumerModule):
    async def consume(self, item):
        logger.info('%s: %s', self.name, item)
        return item
