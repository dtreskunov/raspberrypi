import abc
import asyncio
import logging
import typing

import aiochannel

logger = logging.getLogger(__name__)


class AIOContextManager:
    def __init__(self,
                 enter_coro: typing.Optional[typing.Coroutine] = None,
                 exit_coro: typing.Optional[typing.Coroutine] = None):
        self._enter_coro = enter_coro
        self._exit_coro = exit_coro

    async def __aenter__(self):
        if self._enter_coro:
            await self._enter_coro

    async def __aexit__(self, exc_type, exc, tb):
        if self._exit_coro:
            # need to create a separate task, since the current one may have
            # been cancelled already
            loop = asyncio.get_event_loop()
            task = loop.create_task(self._exit_coro)
            await task


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
        self._channels = []

    def mkchannel(self, maxsize=1):
        channel = aiochannel.Channel(maxsize=maxsize)
        self._channels.append(channel)
        return channel

    async def output(self, item):
        async def _put_item_into_channel(item, channel):
            await channel.put(item)
        await asyncio.gather(*[
            _put_item_into_channel(item, channel)
            for channel in self._channels
        ])


class AIOConsumerModule(AIOProducerModule):
    'Consumes from multiple channels and produces filtered output'

    def __init__(self, name=None, config=None):
        super(AIOConsumerModule, self).__init__(name, config)
        self._sources = []
        self._input = aiochannel.Channel(maxsize=1)
        self._putters = []

    @property
    def sources(self):
        return self._sources

    @sources.setter
    def sources(self, sources):
        self._sources = sources
        for putter in self._putters:
            putter.cancel()

        async def put_into_input(source):
            async for item in source:
                await self._input.put(item)
        self._putters = [asyncio.ensure_future(put_into_input(source))
                         for source in self._sources]

    async def on_start(self):
        pass

    async def on_stop(self):
        pass

    async def run(self):
        async with AIOContextManager(self.on_start(), self.on_stop()):
            while True:
                source_item = await self._input.get()
                consumed_item = await self.consume(source_item)
                await self.output(consumed_item)

    @abc.abstractmethod
    async def consume(self, item):
        pass


class AIOLogger(AIOConsumerModule):
    async def consume(self, item):
        logger.info('%s: %s', self.name, item)
        return item
