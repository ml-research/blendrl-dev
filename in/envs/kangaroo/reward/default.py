from hackatari import HackAtari

def reward_function(self: HackAtari) -> float:
    for obj in self.objects:
        if 'player' in str(obj).lower():
            player = obj
            break

    # got reward and previous step was on the top platform -> reached the child
    # if game_reward == 1.0 and player.prev_y == 4:
    #    reward = 10.0
    # x = 129
    # if player.y == 4:
        # reward = 0.2
    # BUG ↓ with multi envs, rewards collected repeatedly
    self.set_custom_info("reached_child", False)

    if player.y == 4 and player.prev_y != 4:
        reward = 20.0
        self.set_custom_info("reached_child", True)
    elif self.org_reward == 1.0 and player.prev_y != 4:
        reward = 1.0
    else:
        reward = 0.0

    return reward