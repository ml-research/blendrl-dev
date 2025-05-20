def reward_function(self) -> float:
    for obj in self.objects:
        if 'player' in str(obj).lower():
            player = obj
            break

    if player.y == 4 and player.prev_y != 4:
        reward = 20.0
    else:
        reward = 0.0

    return reward