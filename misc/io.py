import os
import re
from pathlib import Path
from typing import List, Tuple


def get_all_checkpoints(checkpoints_dir: Path, sorted: bool = True) -> List[Tuple[int, Path]]:
    checkpoints = os.listdir(checkpoints_dir)
    result = []
    pattern = re.compile("[0-9]+")
    for i, c in enumerate(checkpoints):
        match = pattern.search(c)
        if match is not None:
            step = int(match.group())
            path = checkpoints_dir / c
            result.append((step, path))

    if sorted:
        result.sort()

    return result


def get_most_recent_checkpoint(checkpoints_dir: Path) -> Tuple[int, Path]:
    checkpoints = get_all_checkpoints(checkpoints_dir, sorted=True)

    return checkpoints[-1]
