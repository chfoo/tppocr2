from typing import List, Tuple, NamedTuple

import toml

class ColorPoint(NamedTuple):
    x: int
    y: int
    color: int
    r: int
    g: int
    b: int


class Region:
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__()
        self.name = ''
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.invert = False
        self.scale = 1.0
        self.x2 = 0
        self.y2 = 0
        self.width2 = 0
        self.height2 = 0
        self.invert2 = False
        self.points: List[ColorPoint] = []
        self.transparent_window: TransparentWindow = None

    def box(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

class TransparentWindow:
    def __init__(self):
        super().__init__()
        self.clear_x = 0
        self.clear_y = 0
        self.clear_width = 0
        self.clear_height = 0
        self.trans_x = 0
        self.trans_y = 0
        self.trans_width = 0
        self.trans_height = 0
        self.difference = 0.0

    def clear_box(self) -> Tuple[int, int, int, int]:
        return (self.clear_x, self.clear_y, self.clear_x + self.clear_width, self.clear_y + self.clear_height)

    def trans_box(self) -> Tuple[int, int, int, int]:
        return (self.trans_x, self.trans_y, self.trans_x + self.trans_width, self.trans_y + self.trans_height)

def parse_timestamp_region(filename: str) -> Region:
    with open(filename) as file:
        doc = toml.load(file)

        return toml_table_to_region(doc['timestamp-region'])


def parse_regions(filename: str) -> List[Region]:
    regions = []
    region_map = {}

    with open(filename) as file:
        doc = toml.load(file)

        for item in doc['region']:
            region_map[item['name']] = toml_table_to_region(item)

    for name in doc['region_order']:
        regions.append(region_map[name])

    return regions


def toml_table_to_region(table) -> Region:
    region = Region(table['x'], table['y'], table['width'], table['height'])

    props = ('name', 'invert', 'scale', 'x2', 'y2', 'width2', 'height2', 'invert2')

    for prop in props:
        if prop in table:
            setattr(region, prop, table[prop])

    if 'points' in table:
        points = list(table['points'])

        while points:
            x = points.pop(0)
            y = points.pop(0)
            color = points.pop(0)
            r = (color >> 16) & 0xff
            g = (color >> 8) & 0xff
            b = color & 0xff

            region.points.append(ColorPoint(x, y, color, r, g, b))

    if 'transparent_window' in table:
        window_table = table['transparent_window']
        transparent_window = TransparentWindow()
        props = ('clear_x', 'clear_y', 'clear_width', 'clear_height',
            'trans_x', 'trans_y', 'trans_width', 'trans_height',
            'difference')

        for prop in props:
            setattr(transparent_window, prop, window_table[prop])

        region.transparent_window = transparent_window


    return region

