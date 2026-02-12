from typing import Dict, Any

from ocatari.ram.skiing import _detect_objects_skiing_raw

def reward_function(self) -> float:
    if not hasattr(self, '_reward_fn_state'):
        self._reward_fn_state: Dict[str, Any] = {
            "prev_score": None
        }

    info = dict()
    _detect_objects_skiing_raw(info, self._env.env.unwrapped.ale.getRAM().tolist())
    score = info['score']

    passed_flags = False
    prev_score = self._reward_fn_state["prev_score"]
    if prev_score is not None and score != prev_score:
        passed_flags = True

    self._reward_fn_state["prev_score"] = score
    self.set_custom_info("passed_flags", passed_flags)

    if passed_flags:
        return 1.0

    return 0.0