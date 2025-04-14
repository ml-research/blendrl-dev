import numpy as np


def reward_function(self) -> float:
    player = next((obj for obj in self.objects if "player" in str(obj).lower()), None)
    if player is None:
        return 0.0 

    reward = 0.0

    # Reward for destroying enemies
    if self.org_reward > 0:
        reward += self.org_reward * 10.0 

    # Reward for fuel collection
    for obj in self.objects:
        if "fueldepot" in str(obj).lower():
            distance = np.linalg.norm(np.array(player.xy) - np.array(obj.xy))
            if distance < 20:
                reward += 1.5
                break

    # Survival reward
    reward += 0.05

    # Center position reward
    x = player.xy[0]
    screen_center = 96
    max_x = 192

     # Encourage staying near the center while avoiding edges
    distance_from_center = abs(x - screen_center)
    center_reward = 1.0 - (distance_from_center / screen_center) ** 1.5 
    edge_penalty = max(0, (abs(x - max_x / 2) / (max_x / 2)) ** 3) 

    reward += center_reward - edge_penalty 

    return reward