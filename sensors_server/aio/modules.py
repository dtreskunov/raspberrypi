import abc
import asyncio
import logging
import typing

import aiochannel
from .utils import shutdown_signalled

logger = logging.getLogger(__name__)


class AIOContextManager:
    def __init__(self, on_start=None, on_stop=None):
        self._on_start = on_start
        self._on_stop = on_stop

    async def __aenter__(self):
        if self._on_start:
            await self._on_start()

    async def __aexit__(self, exc_type, exc, tb):
        if self._on_stop:
            # need to create a separate task, since the current one may have
            # been cancelled already
            loop = asyncio.get_event_loop()
            task = loop.create_task(self._on_stop())
            await task


class AIOModule(abc.ABC):
    def __init__(self):
        self._task = None

    @property
    def running(self):
        return not (self._task is None or self._task.cancelled())

    async def wait(self):
        try:
            return await self._task
        except asyncio.CancelledError as e:
            return e

    def start(self):
        self._task = asyncio.ensure_future(self.run())
        return self # chainable

    def stop(self):
        if self._task is not None:
            self._task.cancel()
        return self # chainable
    
    async def restart(self):
        self.stop()
        await self.wait()
        self.start()

    @abc.abstractmethod
    async def run(self):
        pass


class AIOOutput:
    def __init__(self):
        self._outputs = []

    def add_output(self, channel=None):
        if channel is None:
            channel = aiochannel.Channel(maxsize=1)
        if not channel in self._outputs:
            self._outputs.append(channel)
        return channel

    def remove_output(self, channel):
        try:
            self._outputs.remove(channel)
        except ValueError:
            pass

    async def output(self, item):
        async def _put_item_into_channel(item, channel):
            await channel.put(item)
        await asyncio.gather(*[
            _put_item_into_channel(item, channel)
            for channel in self._outputs
        ])


class AIOInput:
    'Consumes from multiple channels and produces filtered output'

    def __init__(self):
        self._input_to_putter = {}
        self._combined_input = aiochannel.Channel(maxsize=1)

    def add_input(self, channel):
        if channel in self._input_to_putter:
            return

        async def put_into_input():
            async for item in channel:
                await self._combined_input.put(item)
        self._input_to_putter[channel] = asyncio.ensure_future(
            put_into_input())

    def remove_input(self, channel):
        putter = self._input_to_putter.pop(channel, None)
        if putter is None:
            return
        putter.cancel()

    async def input(self):
        return await self._combined_input.get()


class AIOEnableableModule(AIOModule):
    def __init__(self):
        AIOModule.__init__(self)
        self._noop_task = None
        self._start_noop_task()
    
    @abc.abstractmethod
    async def noop_loop(self):
        pass
    
    @abc.abstractmethod
    async def real_loop(self):
        pass

    async def on_start(self):
        pass
    
    async def on_stop(self):
        pass
    
    async def run(self):
        async def on_start():
            await self.on_start()
            self._stop_noop_task()
        async def on_stop():
            if not shutdown_signalled():
                self._start_noop_task()
            await self.on_stop()
        async with AIOContextManager(on_start=on_start, on_stop=on_stop):
            await self.real_loop()

    def _start_noop_task(self):
        self._noop_task = asyncio.ensure_future(self.noop_loop())
    
    def _stop_noop_task(self):
        if self._noop_task is not None:
            self._noop_task.cancel()


class AIOProducerModule(AIOOutput, AIOEnableableModule):
    def __init__(self):
        AIOOutput.__init__(self)
        AIOEnableableModule.__init__(self)
    
    @abc.abstractmethod
    async def produce(self):
        pass

    async def noop_loop(self):
        pass
    
    async def real_loop(self):
        while True:
            await self.output(await self.produce())


class AIOConsumerModule(AIOInput, AIOEnableableModule):
    def __init__(self):
        AIOInput.__init__(self)
        AIOEnableableModule.__init__(self)
    
    @abc.abstractmethod
    async def consume(self, item):
        pass

    async def noop_loop(self):
        while True:
            await self.input()
    
    async def real_loop(self):
        while True:
            await self.consume(await self.input())


class AIOFilterModule(AIOInput, AIOOutput, AIOEnableableModule):
    def __init__(self):
        AIOInput.__init__(self)
        AIOOutput.__init__(self)
        AIOEnableableModule.__init__(self)

    @abc.abstractmethod
    async def filter(self, item):
        pass
    
    async def noop_loop(self):
        while True:
            await self.output(await self.input())

    async def real_loop(self):
        while True:    
            item = await self.input()
            filtered_item = await self.filter(item)
            if filtered_item is not None:
                await self.output(filtered_item)


class AIOLogger(AIOConsumerModule):
    def __init__(self, name):
        AIOConsumerModule.__init__(self)
        self._name = name

    async def consume(self, item):
        logger.info('%s: %s', self._name, item)
