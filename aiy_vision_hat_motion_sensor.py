from gpiozero import MotionSensor

class AIYVisionHatMotionSensor(MotionSensor):
    """
    Inverts MotionSensor from gpiozero and hard-codes pull_up to True
    """
    def __init__(
            self, pin=None, queue_len=1, sample_rate=10, threshold=0.5,
            partial=False, pin_factory=None):
        super(AIYVisionHatMotionSensor, self).__init__(
            pin=pin, queue_len=queue_len, sample_rate=sample_rate, threshold=threshold,
            partial=partial, pin_factory=pin_factory,
            pull_up=True # AIY doesn't support False
        )


def _negate(func):
    return lambda *args, **kwds: not func(*args, **kwds)


AIYVisionHatMotionSensor.motion_detected = _negate(MotionSensor.motion_detected)
AIYVisionHatMotionSensor.when_motion = MotionSensor.when_no_motion
AIYVisionHatMotionSensor.when_no_motion = MotionSensor.when_motion
AIYVisionHatMotionSensor.wait_for_motion = MotionSensor.wait_for_no_motion
AIYVisionHatMotionSensor.wait_for_no_motion = MotionSensor.wait_for_motion
