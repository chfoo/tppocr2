import asyncio
import logging
from typing import List, Tuple
import math

import PIL.Image
import PIL.ImageMath
import PIL.ImageStat
import tesserocr

from .region import Region
from .stream import Stream

_logger = logging.getLogger(__name__)

class OCR:
    def __init__(self, stream: Stream, timestamp_region: Region, regions: List[Region]):
        super().__init__()
        self._stream = stream
        self._timestamp_region = timestamp_region
        self._regions = regions

        if not tesserocr.PyTessBaseAPI.Version().startswith('4.'):
            raise Exception('Designed for Tesseract 4 only')

        self._ocr_api = None
        self._reset_ocr()

    def _reset_ocr(self):
        if self._ocr_api:
            self._ocr_api.End()

        self._ocr_api = tesserocr.PyTessBaseAPI(
            path='../tessdata_best/',
            lang='eng+jpn+chi_sim+chi_tra+kor+spa+deu+ita',
            psm=tesserocr.PSM.SINGLE_BLOCK)
        self._ocr_api.SetVariable("classify_enable_learning", "0")
        self._ocr_api.SetVariable("user_defined_dpi", "90")

    async def run(self):
        stream_task = asyncio.create_task(self._stream.run())

        _logger.info('Starting OCR loop')
        frame_count = 0
        done_task = asyncio.create_task(self._stream.done.wait())

        while not self._stream.done.is_set():
            get_task = asyncio.create_task(self._stream.queue.get())
            done, pending = await asyncio.wait((done_task, get_task), return_when=asyncio.FIRST_COMPLETED)

            if get_task in done:
                frame_data = await get_task
                self._process_frame_data(frame_data)
                frame_count += 1
            else:
                await done_task

            if frame_count % 100 == 0:
                _logger.info('Processed %d frame(s)', frame_count)

        _logger.info('Stopped OCR loop')

    def _process_frame_data(self, frame_data: bytes):
        image = PIL.Image.frombytes('RGB', self._stream.frame_size, frame_data)

        for region in self._regions:
            region_score = self._get_score(image, region)

            if region_score >= 0.8:
                text, score = self._get_image_text(image, region)

                if score >= 0.6:
                    timestamp_text, timestamp_score = self._get_image_text(image, self._timestamp_region)
                    print(timestamp_text, region.name, score, text)

                    break

    def _get_score(self, image: PIL.Image, region:Region) -> float:
        if region.points:
            return self._check_points(image, region)
        elif region.transparent_window:
            return self._check_transparent_window(image, region)
        else:
            raise Exception("Shouldn't reach here")

    def _check_points(self, image: PIL.Image, region: Region) -> float:
        total_difference = 0

        for point in region.points:
            pixel = image.getpixel((point.x, point.y))

            total_difference += math.sqrt(
                (pixel[0] - point.r) ** 2 +
                (pixel[1] - point.g) ** 2 +
                (pixel[2] - point.b) ** 2
            )

        max_difference = 10.0

        score = max(0.0, (max_difference - total_difference / len(region.points)) / max_difference)
        return score

    def _check_transparent_window(self, image: PIL.Image, region: Region) -> float:
        clear_image = image.crop(region.transparent_window.clear_box()).convert('HSV')
        trans_image = image.crop(region.transparent_window.trans_box()).convert('HSV')

        clear_stats = PIL.ImageStat.Stat(clear_image)
        trans_stats = PIL.ImageStat.Stat(trans_image)

        transition_difference = (clear_stats.mean[2] - trans_stats.mean[2]) / 255.0
        difference = abs(transition_difference - region.transparent_window.difference)
        max_difference = 0.05
        score = min(1.0, max(0.0, (max_difference - difference) / max_difference))

        return score

    def _get_image_text(self, image: PIL.Image, region: Region) -> Tuple[str, float]:
        target_image = image.crop(region.box())

        if region.name == 'cutscene_dialog':
            # https://stackoverflow.com/a/39173605
            #  NV = Min(255, Max(0, (V - L) * 255 / (H - L)))
            target_image = PIL.ImageMath.eval('(convert(a, "L") - 127) * 255 / (255 - 127)', a=target_image).convert('RGB')

        if region.scale != 1.0:
            target_image = target_image.resize(
                (int(region.width * region.scale), int(region.height * region.scale)),
                PIL.Image.NEAREST)

        self._ocr_api.ClearAdaptiveClassifier()
        self._ocr_api.SetImage(target_image)
        text = self._ocr_api.GetUTF8Text().strip()

        # ocr_image: PIL.Image = self._ocr_api.GetThresholdedImage()
        # ocr_image.save('/tmp/ocr_debug.png')

        score = self._ocr_api.MeanTextConf() / 100

        return (text, score)
