from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from tractian_cm.part1.schemas import Wave
from tractian_cm.io.metadata_part2 import validate_orientation
from tractian_cm.io.loaders import RawTriAxial


class OrientationError(ValueError):
    pass


def to_hva_waves(raw: RawTriAxial, orientation: Dict[str, str]) -> Tuple[Wave, Wave, Wave]:
    """
    Convert axisX/axisY/axisZ waveforms into horizontal/vertical/axial Wave objects
    using the orientation mapping from metadata.

    orientation example:
      {"axisX": "horizontal", "axisY": "axial", "axisZ": "vertical"}
    """
    validate_orientation(orientation)

    axis_map = {
        "axisX": raw.axisX,
        "axisY": raw.axisY,
        "axisZ": raw.axisZ,
    }

    # invert mapping: orientation -> axis key
    inv = {v: k for k, v in orientation.items()}
    if set(inv.keys()) != {"horizontal", "vertical", "axial"}:
        raise OrientationError(f"Invalid orientation mapping (missing H/V/A): {orientation}")

    t_list = raw.t.tolist()

    wave_hor = Wave(time=t_list, signal=axis_map[inv["horizontal"]].tolist())
    wave_ver = Wave(time=t_list, signal=axis_map[inv["vertical"]].tolist())
    wave_axi = Wave(time=t_list, signal=axis_map[inv["axial"]].tolist())

    return wave_hor, wave_ver, wave_axi
