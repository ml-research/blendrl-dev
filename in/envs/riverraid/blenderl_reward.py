import numpy as np


def reward_function(self) -> float:
    player = next((obj for obj in self.objects if "player" in str(obj).lower()), None)
    if player is None:
        return 0.0  # No player, no reward

    reward = 0.0

    # Reward for destroying enemies
    if self.org_reward > 0:
        reward += self.org_reward * 10.0  # High reward for destruction

    # Reward for fuel collection (increased)
    for obj in self.objects:
        if "fueldepot" in str(obj).lower():
            distance = np.linalg.norm(np.array(player.xy) - np.array(obj.xy))
            if distance < 20:
                reward += 1.5  # Increase fuel reward
                break

    # Survival reward (encourage lasting longer)
    reward += 0.05

    return reward

