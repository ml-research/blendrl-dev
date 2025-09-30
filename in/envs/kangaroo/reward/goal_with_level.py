from typing import Dict, Any


def reward_function(self) -> float:
    for obj in self.objects:
        if 'player' in str(obj).lower():
            player = obj
            break

    levels = [148, 100, 52, 4]
    current_level_idx = 0
    for level_idx in range(1, len(levels)):
        if player.y <= levels[level_idx]:
            current_level_idx = level_idx
        else:
            break

    current_frame_number = self.ale.getEpisodeFrameNumber()

    if not hasattr(self, '_reward_fn_state'):
        self._reward_fn_state: Dict[str, Any] = {
            "level_idx": current_level_idx,
            "frame_number": current_frame_number,
        }

    prev_frame_number = self._reward_fn_state["frame_number"]
    new_ep = prev_frame_number > current_frame_number

    if new_ep or self._reward_fn_state["level_idx"] is None:
        self._reward_fn_state["level_idx"] = current_level_idx

    prev_level_idx = self._reward_fn_state["level_idx"]

    reward = 0.0
    self.set_custom_info("reached_child", False)
    self.set_custom_info("reached_next_level", False)

    # if player moved one level up, give reward and update reward fn state
    if not new_ep and current_level_idx > prev_level_idx:
        reward += 0.5
        self.set_custom_info("reached_next_level", True)

        # if player reached child, give additional reward and reset reward fn state
        if current_level_idx == len(levels) - 1:
            reward += 0.5
            self._reward_fn_state["level_idx"] = None
            self.set_custom_info("reached_child", True)
        else:
            self._reward_fn_state["level_idx"] = current_level_idx

    self._reward_fn_state["frame_number"] = current_frame_number

    return reward