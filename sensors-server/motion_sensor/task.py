import asyncio
import logging

import aiy.pins
from .aiy_vision_hat_motion_sensor import AIYVisionHatMotionSensor

logger = logging.getLogger(__name__)


async def motion_sensor_task(callback):
    logger.info('starting motion_sensor_task')
    sensor = AIYVisionHatMotionSensor(aiy.pins.PIN_A)
    sensor.when_motion = lambda: callback('motion_detected')
    while True:
        await asyncio.sleep(1)
