import numpy as np


def reward_function(self) -> float:
    # Find the player object
    player = None
    for obj in self.objects:
        if "player" in str(obj).lower():
            player = obj
            break

    # Ensure player exists
    if player is None:
        return 0.0  # No reward if player is missing

    # Base reward
    reward = 0.0

    # Reward for destroying enemies
    if self.org_reward > 0:
        reward = self.org_reward * 10.0  # High reward for destruction

    # Reward for forward movement
    elif player.x > player.prev_x:
        reward = 1.0

    # Slight reward for being close to a fuel depot
    for obj in self.objects:
        if "fueldepot" in str(obj).lower():
            distance = np.linalg.norm(np.array(player.xy) - np.array(obj.xy))
            if distance < 20:
                reward += 0.5
                break

    return reward
