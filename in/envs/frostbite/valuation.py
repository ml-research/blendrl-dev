import torch as th

from nsfr.utils.common import bool_to_probs

def enemy_left(player: th.Tensor, enemy: th.Tensor) -> th.Tensor:
    """Goes towards 1 if the player is left of the enemy and the enemy is close."""
    player_x = player[..., 1]
    player_y = player[..., 2]
    enemy_x = enemy[..., 1]
    enemy_y = enemy[..., 2]
    player_is_left = player_x < enemy_x
    enemy_on_same_y = abs(player_y - enemy_y) < 64
    if (player_is_left&enemy_on_same_y).all():
        result = th.clip(player_is_left * (64 - (enemy_x - player_x)) / 64, 0, 1)
        return result
    return bool_to_probs(th.tensor([False]))

def enemy_right(player: th.Tensor, enemy: th.Tensor) -> th.Tensor:
    """Goes towards 1 if the player is right of the enemy and the enemy is close."""
    player_x = player[..., 1]
    player_y = player[..., 2]
    enemy_x = enemy[..., 1]
    enemy_y = enemy[..., 2]
    player_is_right = enemy_x < player_x 
    enemy_on_same_y = abs(player_y - enemy_y) < 64
    if (player_is_right&enemy_on_same_y).all():
        result = th.clip(player_is_right * (64 - (player_x - enemy_x)) / 64, 0, 1)
        return result
    return bool_to_probs(th.tensor([False]))

def no_enemy_left(player: th.Tensor, enemy: th.Tensor) -> th.Tensor:
    """Goes towards 1 if there is no enemy left of the player."""
    enemy_on_left = enemy_left(player, enemy)
    result = th.ones_like(enemy_on_left) - enemy_on_left
    return result

def no_enemy_right(player: th.Tensor, enemy: th.Tensor) -> th.Tensor:
    """Goes towards 1 if there is no enemy right of the player."""
    enemy_on_right = enemy_right(player, enemy)
    result = th.ones_like(enemy_on_right) - enemy_on_right
    return result

def no_enemy_above(player: th.Tensor, enemy: th.Tensor) -> th.Tensor:
    """True iff there is an enemy right above the player"""
    player_x = player[..., 1]
    player_y = player[..., 2]
    enemy_x = enemy[..., 1]
    enemy_y = enemy[..., 2]
    y_distance = player_y - enemy_y
    return bool_to_probs((y_distance > 0) & (y_distance <= 40) & (abs(player_x - enemy_x) < 20))

def no_enemy_below(player: th.Tensor, enemy: th.Tensor) -> th.Tensor:
    """True iff there is a block below the player"""
    player_x = player[..., 1]
    player_y = player[..., 2]
    enemy_x = enemy[..., 1]
    enemy_y = enemy[..., 2]
    y_distance = enemy_y - player_y
    return bool_to_probs((y_distance > 0) & (y_distance <= 40) & (abs(player_x - enemy_x) < 20))

def left_of_door(player: th.Tensor, door: th.Tensor) -> th.Tensor:
    """True iff the player is 'left of' the door."""
    player_x = player[..., 1]
    door_x = door[..., 1]
    door_prob = door[:, 0]
    return bool_to_probs(player_x < door_x) * door_prob

def right_of_door(player: th.Tensor, door: th.Tensor) -> th.Tensor:
    """True iff the player is 'right of' the door."""
    player_x = player[..., 1]
    door_x = door[..., 1]
    door_prob = door[:, 0]
    return bool_to_probs(player_x > door_x) * door_prob

def door_exists(door: th.Tensor) -> th.Tensor:
    result = door[..., 0] == 1
    return bool_to_probs(result)

def no_door_exists(door: th.Tensor) -> th.Tensor:
    result = door[..., 0] != 1
    return bool_to_probs(result)

def block_above(player: th.Tensor, block: th.Tensor) -> th.Tensor:
    """True iff there is a block right above the player"""
    player_x = player[..., 1]
    player_y = player[..., 2]
    block_x = block[..., 1]
    block_y = block[..., 2]
    y_distance = player_y - block_y
    return bool_to_probs((y_distance > 0) & (y_distance <= 40) & (abs(player_x - block_x) < 20))

def block_below(player: th.Tensor, block: th.Tensor) -> th.Tensor:
    """True iff there is a block below the player"""
    player_x = player[..., 1]
    player_y = player[..., 2]
    block_x = block[..., 1]
    block_y = block[..., 2]
    y_distance = block_y - player_y
    return bool_to_probs((y_distance > 0) & (y_distance <= 40) & (abs(player_x - block_x) < 20))

def no_block_above(player: th.Tensor, block: th.Tensor) -> th.Tensor:
    """Goes towards 1 iff there is no block above the player"""
    block_is_above = block_above(player, block)
    result = th.ones_like(block_is_above) - block_is_above
    return result

def no_block_below(player: th.Tensor, block: th.Tensor) -> th.Tensor:
    """Goes towards 1 iff there is no block above the player"""
    block_is_below = block_below(player, block)
    result = th.ones_like(block_is_below) - block_is_below
    return result

def far_right(player:th.Tensor) -> th.Tensor:
    player_x = player[..., 1]
    result = th.clip((player_x - 162) / 30, 0, 1)
    return result

def far_left(player:th.Tensor) -> th.Tensor:
    player_x = player[..., 1]
    result = th.clip((30 - player_x) / 30, 0, 1)
    return result

def neural_agent_value(player:th.Tensor) -> th.Tensor:
    return bool_to_probs(th.tensor([False]))

def logic_agent_value(player:th.Tensor) -> th.Tensor:
    return bool_to_probs(th.tensor([True]))
