import logging

from aio.modules import AIOConsumerModule
from config.items import Configurable, TypedConfigItem
from util import do_imports

logger = logging.getLogger(__name__)


class MQTTPublisher(AIOConsumerModule, Configurable):
    def __init__(self):
        AIOConsumerModule.__init__(self)
        Configurable.__init__(self,
                              TypedConfigItem(
                                  name='enabled',
                                  display_name='Enabled',
                                  description=None,
                                  default_value=False,
                                  item_type=bool),
                              TypedConfigItem(
                                  name='uri',
                                  display_name='URI',
                                  description='MQTT broker URI (mqtt[s]://[username][:password]@host.domain[:port])',
                                  default_value='mqtt://iot.eclipse.org',
                                  item_type=str))
        self._client = None

    async def on_start(self):
        # https://hbmqtt.readthedocs.io/en/latest
        await do_imports('hbmqtt', 'hbmqtt.client')
        import hbmqtt.client
        self._client = hbmqtt.client.MQTTClient()
        logging.info('Connecting to %s', self.config.uri.value)
        await self._client.connect(self.config.uri.value)
        logging.info('Connected to %s', self.config.uri.value)

    async def on_stop(self):
        logging.info('Disconnecting from %s', self.config.uri.value)
        await self._client.disconnect()
        logging.info('Disconnected from %s', self.config.uri.value)

    async def consume(self, item):
        try:
            topic = item['topic']
        except (TypeError, KeyError):
            logger.debug('Item `%s` had no topic', item)
            return
        try:
            payload = item['payload']
        except (TypeError, KeyError):
            logger.debug('Item `%s` had no payload', item)
            return
        await self._client.publish(topic, payload)
        logger.debug('Published to topic `%s` payload `%s`', topic, payload)


###
if __name__ == '__main__':
    # import ptvsd
    # address = ('0.0.0.0', 5678)
    # ptvsd.enable_attach(address=address)
    # print('Waiting for debugger on {}...'.format(address))
    # ptvsd.wait_for_attach()

    import aio.utils
    import aiochannel
    import asyncio

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)-15s %(levelname)s [%(filename)s:%(lineno)d][%(process)d:%(threadName)s] %(message)s')

    async def async_main():
        logger.info(''''
            Sleeping for 5 seconds... Use the following command to receive test messages:

            hbmqtt_sub --url mqtt://test.mosquitto.org -t dtreskunov
        ''')
        await asyncio.sleep(5)

        channel = aiochannel.Channel(maxsize=1)
        mqtt = MQTTPublisher().start()
        mqtt.add_input(channel)
        while True:
            await channel.put({'topic': 'dtreskunov', 'payload': b'Hello, world!'})
            await asyncio.sleep(1)

    aio.utils.main(async_main)
