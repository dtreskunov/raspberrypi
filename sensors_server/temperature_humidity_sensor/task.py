import asyncio
import logging

import Adafruit_DHT
import aiy.pins

logger = logging.getLogger(__name__)


async def temperature_humidity_sensor_task(callback):
    logger.info('starting temperature_humidity_sensor_task')
    while True:
        # Try to grab a sensor reading.  Use the read_retry method which will retry up
        # to 15 times to get a sensor reading (waiting 2 seconds between each retry).
        humidity, temperature = Adafruit_DHT.read_retry(
            Adafruit_DHT.DHT11, aiy.pins.PIN_B.gpio_spec.pin)
        if humidity is not None:
            callback({'humidity': humidity})
        if temperature is not None:
            callback({'temperature': temperature})
        await asyncio.sleep(10)
