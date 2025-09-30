from typing import Dict


def reward_function(self) -> float:
    for obj in self.objects:
        if 'player' in str(obj).lower():
            player = obj
            break

    collected_divers = self.objects[-6:]
    num_collected_divers = sum([d.category == "CollectedDiver" for d in collected_divers])

    current_frame_number = self.ale.getEpisodeFrameNumber()

    if not hasattr(self, '_reward_fn_state'):
        self._reward_fn_state: Dict[str, int] = {
            "num_collected_divers": num_collected_divers,
            "frame_number": current_frame_number,
        }

    prev_frame_number = self._reward_fn_state["frame_number"]
    new_ep = prev_frame_number > current_frame_number

    if new_ep:
        self._reward_fn_state["num_collected_divers"] = num_collected_divers

    prev_num_collected_divers = self._reward_fn_state["num_collected_divers"]

    reward = 0.0
    self.set_custom_info("rescued_divers", False)

    if self.org_reward == 1.0 and player.y == 46:
        # when rescued 6 divers
        self.set_custom_info("rescued_divers", True)
        reward = 1.0
    elif num_collected_divers > prev_num_collected_divers:
        reward = (num_collected_divers - prev_num_collected_divers) * 0.5

    self._reward_fn_state["num_collected_divers"] = num_collected_divers
    self._reward_fn_state["frame_number"] = current_frame_number

    return reward