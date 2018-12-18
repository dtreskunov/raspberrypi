from PIL import Image, ImageDraw

import picamera


def _round_to_bit(value, power):
    """Rounds the given value to the next multiple of 2^power.

    Args:
      value: int to be rounded.
      power: power of two which the value should be rounded up to.
    Returns:
      the result of value rounded to the next multiple 2^power.
    """
    return (((value - 1) >> power) + 1) << power


def _round_buffer_dims(dims):
    """Appropriately rounds the given dimensions for image overlaying.

    The overlay buffer must be rounded the next multiple of 32 for the hight, and
    the next multiple of 16 for the width."""
    return (_round_to_bit(dims[0], 5), _round_to_bit(dims[1], 4))


class Preview:
    '''Utility for managing annotations on the camera preview
    access to underlying PIL.ImageDraw'''

    def __init__(self, camera: picamera.PiCamera, bg_color=(0, 0, 0, 0), dimensions=None):
        _monkey_patch_picamera()
        self._camera = camera
        self._bg_color = bg_color
        self._dims = dimensions if dimensions else camera.resolution
        self._buffer_dims = _round_buffer_dims(self._dims)
        self._buffer = Image.new('RGBA', self._buffer_dims)
        self._draw = ImageDraw.Draw(self._buffer)
        self._overlay = None

    def __enter__(self):
        self._camera.start_preview()
        self._overlay = self._camera.add_overlay(
            self._buffer.tobytes(), format='rgba', layer=3, size=self._buffer_dims)
        return self

    def __exit__(self, *args, **kwds):
        self._camera.remove_overlay(self._overlay)
        self._camera.stop_preview()

    # MMALPort has a bug in enable.wrapper, where it always calls
    # self._pool.send_buffer(block=False) regardless of the port direction.
    # This is in contrast to setup time when it only calls
    # self._pool.send_all_buffers(block=False)
    # if self._port[0].type == mmal.MMAL_PORT_TYPE_OUTPUT.
    # Because of this bug updating an overlay once will log a MMAL_EAGAIN
    # error every update. This is safe to ignore as we the user is driving
    # the renderer input port with calls to update() that dequeue buffers
    # and sends them to the input port (so queue is empty on when
    # send_all_buffers(block=False) is called from wrapper).
    # As a workaround, monkey patch MMALPortPool.send_buffer and
    # silence the "error" if thrown by our overlay instance.
    def _monkey_patch_picamera(self):
        original_send_buffer = picamera.mmalobj.MMALPortPool.send_buffer

        def silent_send_buffer(zelf, *args, **kwargs):
            try:
                original_send_buffer(zelf, *args, **kwargs)
            except picamera.exc.PiCameraMMALError as error:
                # Only silence MMAL_EAGAIN for our target instance.
                our_target = self._overlay.renderer.inputs[0].pool == zelf
                if not our_target or error.status != 14:
                    raise error

        picamera.mmalobj.MMALPortPool.send_buffer = silent_send_buffer

    @property
    def draw(self):
        ':return PIL.ImageDraw '
        return self._draw

    def update(self):
        'Updates the contents of the overlay.'
        self._overlay.update(self._buffer.tobytes())

    def clear(self):
        'Clears the contents of the overlay - leaving only the plain background.'
        self._draw.rectangle((0, 0) + self._dims, fill=self._bg_color)
