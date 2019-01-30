import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pip._internal

from aio.modules import AIOConsumerModule
from config.items import Configurable, TypedConfigItem
from util import retry


class MQTTConsumer(Configurable, AIOConsumerModule):        
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
        self._task = None
        self._executor = ThreadPoolExecutor(1)
    
    async def apply_config(self, config_values):
        updated = self.config.apply(config_values)
        if updated:
            if self._task:
                self._task.cancel()
                await asyncio.wait([self._task])
            if self.config.enabled.value:
                self._task = asyncio.ensure_future(self.run())

    async def on_start(self):
        do_imports()
    async def on_stop(self):
        pass
    async def consume(self, item):
        pass

def pip_install(*modules):
    try:
        pip._internal.main(['install', '--user'] + modules)
    except SystemExit as e:
        raise RuntimeError('Unable to pip install {}: {}'.format(modules, e))

@util.retry(partial(pip_install, 'paho-mqtt'), ImportError)
def do_imports():
    import paho.mqtt.client as mqtt
