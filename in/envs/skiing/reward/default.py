def reward_function(self) -> float:

    reward = 0.0
    if self.org_reward != 0.0:
        reward = 1.0

    return reward