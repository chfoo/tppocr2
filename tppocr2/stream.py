import asyncio
import logging
import os
import subprocess
import time
from typing import Tuple

_logger = logging.getLogger(__name__)

class Stream:
    def __init__(self, url: str, frame_size: Tuple[int, int],
            real_time: bool = False, processing_fps: int = 60):
        super().__init__()
        self._url = url
        self._frame_size = frame_size
        self._real_time = real_time
        self._processing_fps = processing_fps
        self._queue = asyncio.Queue(120)
        self._done = asyncio.Event()

    @property
    def queue(self) -> asyncio.Queue:
        return self._queue

    @property
    def frame_size(self) -> Tuple[int, int]:
        return self._frame_size

    @property
    def done(self) -> asyncio.Event:
        return self._done

    async def run(self):
        args = [
            '-i', self._url,
            '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo',
            '-r', str(self._processing_fps),
            '-nostats', '-v', 'error', '-nostdin', '-'
        ]

        if self._real_time:
            args.insert(1, '-re')

        env = os.environ.copy()
        env['AV_LOG_FORCE_NOCOLOR'] = '1'

        _logger.info('Starting ffmpeg')

        process = await asyncio.create_subprocess_exec(
            'ffmpeg',
            *args,
            stdout=subprocess.PIPE,
            env=env
        )

        frame_data_length = self._frame_size[0] * self._frame_size[1] * 3
        log_cooldown_timestamp = 0

        while process.returncode is None:
            try:
                frame_data = await process.stdout.readexactly(frame_data_length)
            except asyncio.streams.IncompleteReadError:
                break

            try:
                self._queue.put_nowait(frame_data)
            except asyncio.IncompleteReadError:
                time_now = time.monotonic()

                if time_now - log_cooldown_timestamp > 60:
                    _logger.warning('Queue full. You may need to lower '
                                    'settings or increase CPU power.')
                    log_cooldown_timestamp = time_now

        _logger.info('Waiting for ffmpeg to exit')
        self._done.set()

        await process.wait()

        _logger.info('ffmpeg exited')
