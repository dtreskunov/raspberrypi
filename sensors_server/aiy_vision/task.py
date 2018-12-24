import asyncio
import logging

from aiy.vision.inference import InferenceEngine

logger = logging.getLogger(__name__)


async def aiy_vision_task(callback):
    logger.info('starting aiy_vision_task')
    with InferenceEngine() as engine:
        while True:
            inference_state = engine.get_inference_state()
            camera_state = engine.get_camera_state()
            system_info = engine.get_system_info()
            firmware_info = engine.get_firmware_info()
            callback({
                'inference_state': {
                    'loaded_models': inference_state.loaded_models,
                    'processing_models': inference_state.processing_models,
                },
                'camera_state': {
                    'running': camera_state.running,
                    'width': camera_state.width,
                    'height': camera_state.height,
                },
                'system_info': {
                    'uptime_seconds': system_info.uptime_seconds,
                    'temperature_celsius': system_info.temperature_celsius,
                },
                'firmware_info': {
                    'version': firmware_info,
                }
            })
            await asyncio.sleep(60)
