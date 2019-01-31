import abc
import asyncio
import logging
import typing

import aiochannel

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
    def __init__(self, name=None):
        self._name = name or type(self).__name__
        self._task = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

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

    def stop(self):
        if self._task is not None:
            self._task.cancel()
    
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


class AIOFilterModule(AIOInput, AIOOutput, AIOModule):
    def __init__(self):
        AIOInput.__init__(self)
        AIOOutput.__init__(self)
        AIOModule.__init__(self)
        self._passthru_task = None
        self._start_passthru_task()
    
    async def _passthru(self):
        while True:
            await self.output(await self.input())
    
    def _start_passthru_task(self):
        self._passthru_task = asyncio.ensure_future(self._passthru())
    
    def _stop_passthru_task(self):
        if self._passthru_task is not None:
            self._passthru_task.cancel()
    
    async def run(self):
        async def on_start():
            await self.on_start()
            self._stop_passthru_task()
        async def on_stop():
            self._start_passthru_task()
            await self.on_stop()
        async with AIOContextManager(on_start=on_start, on_stop=on_stop):
            while True:
                item = await self.input()
                filtered_item = await self.filter(item)
                if filtered_item is not None:
                    await self.output(filtered_item)
    
    async def on_start(self):
        pass
    
    async def on_stop(self):
        pass

    @abc.abstractmethod
    async def filter(self, item):
        pass

class AIOLogger(AIOModule, AIOInput, AIOOutput):
    def __init__(self):
        AIOModule.__init__(self)
        AIOInput.__init__(self)
        AIOOutput.__init__(self)

    async def run(self):
        while True:
            item = await self.input()
            logger.info('%s: %s', self.name, item)
            await self.output(item)
