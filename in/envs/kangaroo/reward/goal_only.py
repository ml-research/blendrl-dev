def reward_function(self) -> float:
    for obj in self.objects:
        if 'player' in str(obj).lower():
            player = obj
            break

    self.set_custom_info("reached_child", False)

    if player.y == 4 and player.prev_y != 4:
        reward = 1.0
        self.set_custom_info("reached_child", True)
    else:
        reward = 0.0

    return reward