def reward_function(self) -> float:
    for obj in self.objects:
        if 'player' in str(obj).lower():
            player = obj
            break

    # BUG ↓ with multi envs, rewards collected repeatedly
    if self.org_reward == 1.0 and player.y != 46:
        # e.g. eliminate a shark
        reward = 0.5
    elif self.org_reward == 1.0 and player.y == 46:
        # when rescued 6 divers
        reward = 1.0
    else:
        reward = 0.0
    return reward