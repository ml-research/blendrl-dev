import numpy as np


def reward_function(self) -> float:
    for obj in self.objects:
        if 'player' in str(obj).lower():
            player = obj
        if 'door' in str(obj).lower():
            door_exists = True
        else:
            door_exists = False

    reward = 0.0

    #surviving
    reward += 0.05

    #door_exists
    if door_exists:
        reward += 0.5
    
    return reward