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

    def __init__(self, camera: picamera.Camera, bg_color=(0, 0, 0, 0), dimensions=None):
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
