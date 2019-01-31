import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import aiochannel
import pip._internal

from aio.modules import AIOInput
from config.items import Configurable, TypedConfigItem
from util import retry


class EnableableConsumerModule(AIOInput):
    def __init__(self, wrapped, restart_on_config_change=True):
        if not hasattr(wrapped, 'config'):
            raise ValueError('.config needed')
        if not hasattr(wrapped.config, 'apply'):
            raise ValueError('.config.apply needed')
        if not hasattr(wrapped.config, 'enabled'):
            raise ValueError('.config.enabled needed')
        if not hasattr(wrapped.config.enabled, 'value'):
            raise ValueError('.config.enabled.value needed')
        self._wrapped = wrapped
        self._to_wrapped = aiochannel.Channel(maxsize=1)
        self._from_wrapped = aiochannel.Channel(maxsize=1)
        self._wrapped.add_input(self._to_wrapped)
        self._wrapped.add_output(self._from_wrapped)
        self._wrapped_run_task = None
    
    @property
    def enabled(self):
        return self._wrapped.config.enabled.value
    
    @property
    def wrapped(self):
        return self._wrapped

    async def apply_config(self, config_values):
        updated = self._wrapped.config.apply(config_values)
        if updated:
            if self._wrapped_run_task:
                self._wrapped_run_task.cancel()
                await asyncio.wait([self._wrapped_run_task])
            if self._wrapped.config.enabled.value:
                self._wrapped_run_task = asyncio.ensure_future(self._wrapped.run())

    async def consume(self, item):
        if self.enabled:
            if not self._wrapped_run_task or self._wrapped_run_task.cancelled():
                raise RuntimeError('Wrapper is enabled, but the wrapped module is not running')
            await self._to_wrapped.put(item)
        else:
            return item

    


class MQTTConsumer(Configurable, AIOModule, AIOInput):        
    def __init__(self):
        Configurable.__init__(self,
            TypedConfigItem(
                    name='enabled',
                    display_name='Enabled',
                    description=None,
                    default_value=False,
                    item_type=bool),
            TypedConfigItem(
                    name='host',
                    display_name='Host',
                    description='MQTT broker hostname',
                    default_value='localhost',
                    item_type=str),
            TypedConfigItem(
                    name='port',
                    display_name='Port',
                    description='MQTT broker port number',
                    default_value=1883,
                    item_type=int))
    
    async def on_start(self):
        do_imports()
    async def on_stop(self):
        pass
    async def run(self):
        pass

def pip_install(*modules):
    try:
        pip._internal.main(['install', '--user'] + modules)
    except SystemExit as e:
        raise RuntimeError('Unable to pip install {}: {}'.format(modules, e))

@util.retry(partial(pip_install, 'paho-mqtt'), ImportError)
def do_imports():
    import paho.mqtt.client as mqtt
