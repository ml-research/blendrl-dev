from typing import Dict, Any

from hackatari import HackAtari

def reward_function(self: HackAtari) -> float:
    for obj in self.objects:
        if 'player' in str(obj).lower():
            player = obj
            break

    current_lives = self.ale.lives()

    if not hasattr(self, '_reward_fn_state'):
        self._reward_fn_state: Dict[str, Any] = {
            "lives": current_lives,
        }

    prev_lives = self._reward_fn_state["lives"]

    reward = 0.0
    self.set_custom_info("reached_child", False)

    # if player has lost a life, give penalty and update reward fn state
    if current_lives < prev_lives:
        reward -= 1.0

    if player.y == 4 and player.prev_y != 4:
        reward += 20.0
        self.set_custom_info("reached_child", True)
    elif self.org_reward == 1.0 and player.prev_y != 4:
        reward += 1.0

    self._reward_fn_state["lives"] = current_lives

    return reward