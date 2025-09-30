def reward_function(self) -> float:
    for obj in self.objects:
        if 'player' in str(obj).lower():
            player = obj
            break

    self.set_custom_info("rescued_divers", False)

    if self.org_reward == 1.0 and player.y == 46:
        # when rescued 6 divers
        reward = 1.0
        self.set_custom_info("rescued_divers", True)
    else:
        reward = 0.0
    return reward