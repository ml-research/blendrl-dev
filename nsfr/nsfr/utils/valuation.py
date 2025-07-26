import re
from typing import Optional

import torch
from nsfr.fol.logic import Const


def fuzzy_position(pos1, pos2, keyword):
    x = pos2[:, 0] - pos1[:, 0]
    y = pos2[:, 1] - pos1[:, 1]
    tan = torch.atan2(y, x)
    degree = tan[:] / torch.pi * 180
    if keyword == 'top':
        probs = 1 - abs(degree[:] - 90) / 90
        result = torch.where((180 >= degree) & (degree >= 0), probs * 0.9, 0.)
    elif keyword == 'left':
        probs = (abs(degree[:]) - 90) / 90
        result = torch.where((degree <= -90) | (degree >= 90), probs * 0.9, 0.)
    elif keyword == 'bottom':
        probs = 1 - abs(degree[:] + 90) / 90
        result = torch.where((0 >= degree) & (degree >= -180), probs * 0.9, 0.)
    elif keyword == 'right':
        probs = 1 - abs(degree[:]) / 90
        result = torch.where((90 >= degree) & (degree >= -90), probs * 0.9, 0.)

    return result

def get_object_index_of_const(const: Const) -> Optional[int]:
    result = re.match("obj([1-9][0-9]*)", const.name)
    if result is not None:
        # The constant is an object constant
        obj_id = result[1]
        obj_index = int(obj_id) - 1
        return obj_index

    return None
