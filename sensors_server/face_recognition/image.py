import base64
import io
import logging
import os
import os.path

import numpy
import PIL.Image

import picamera
from util.stopwatch import make_stopwatch

JPEG_QUALITY = 75
logger = logging.getLogger(__name__)
stopwatch = make_stopwatch(logger)


class MyImage():
    'Provides efficient conversion between bytes, numpy.array, and PIL.Image representations'
    
    @staticmethod
    def capture(camera, **kwds):
        'Captures camera images directly into a numpy.array'
        kwds['format'] = 'rgb'
        with stopwatch('MyImage.capture'):
            with picamera.array.PiRGBArray(camera) as stream:
                camera.capture(stream, **kwds)
                return MyImage(numpy_array=stream.array)

    def __init__(self, _bytes=None, pil_image=None, numpy_array=None, jpeg_quality=JPEG_QUALITY):
        if sum((_bytes is not None, pil_image is not None, numpy_array is not None)) != 1:
            raise ValueError(
                'Exactly one of _bytes, pil_image and numpy_array must be specified!')
        self._bytes = _bytes
        self._pil_image = pil_image
        self._numpy_array = numpy_array
        self._jpeg_quality = jpeg_quality

    @property
    def bytes(self):
        if not self._bytes:
            with stopwatch('PIL.Image.save'):
                stream = io.BytesIO()
                self.pil_image.save(stream, format='jpeg',
                                    quality=self._jpeg_quality)
                self._bytes = stream.getvalue()
        return self._bytes

    @property
    def width(self):
        if self._pil_image is not None:
            return self._pil_image.width
        if self._numpy_array is not None:
            return self._numpy_array.shape[1]
        return self.pil_image.width

    @property
    def height(self):
        if self._pil_image is not None:
            return self._pil_image.height
        if self._numpy_array is not None:
            return self._numpy_array.shape[0]
        return self.pil_image.height

    @property
    def pil_image(self):
        if self._pil_image is None:
            if self._numpy_array is not None:
                with stopwatch('PIL.Image.fromarray'):
                    self._pil_image = PIL.Image.fromarray(
                        self._numpy_array.astype('uint8'), 'RGB')
            elif self._bytes is not None:
                with stopwatch('PIL.Image.open'):
                    stream = io.BytesIO(self._bytes)
                    self._pil_image = PIL.Image.open(stream)
            else:
                raise Exception('No data')
        return self._pil_image

    @property
    def numpy_array(self):
        if self._numpy_array is None:
            if self._pil_image is None:
                raise Exception('No data')
            with stopwatch('numpy.array'):
                self._numpy_array = numpy.array(self._pil_image)
        return self._numpy_array

    @property
    def data_uri(self):
        return 'data:image/jpeg;base64,{}'.format(
            base64.b64encode(self.bytes).decode())

    def save(self, file):
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, 'wb') as fp:
            fp.write(self.bytes)
        logger.debug('saved %s', file)

    def crop(self, region):
        'returns a new instance cropped to the given region (left, top, right, bottom)'
        pil_image = self.pil_image.crop(region)
        pil_image.load()
        return MyImage(pil_image=pil_image)
