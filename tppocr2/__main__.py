import argparse
import asyncio
import logging

from .region import parse_regions, parse_timestamp_region
from .ocr import OCR
from .stream import Stream

def main():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command', required=True)

    ocr_parser = subparser.add_parser('ocr')
    ocr_parser.add_argument('config')
    ocr_parser.add_argument('url')
    ocr_parser.add_argument('--real-time', help='Process stream as live stream')
    ocr_parser.add_argument('--processing-fps', help='FPS at which frames are sampled and processed', type=int, default=60)
    ocr_parser.add_argument('--stream-width', help='Width of input frames in pixels', type=int, default=1280)
    ocr_parser.add_argument('--stream-height', help='Height of input frames in pixels', type=int, default=720)
    ocr_parser.set_defaults(func=run_ocr)

    csv_parser = subparser.add_parser('csv')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    args.func(args)

def run_ocr(args):
    config_filename: str = args.config
    timestamp_region = parse_timestamp_region(config_filename)
    regions = parse_regions(config_filename)

    stream = Stream(args.url, (args.stream_width, args.stream_height), args.real_time, args.processing_fps)
    ocr = OCR(stream, timestamp_region, regions)

    asyncio.get_event_loop().run_until_complete(ocr.run())


if __name__ == '__main__':
    main()
