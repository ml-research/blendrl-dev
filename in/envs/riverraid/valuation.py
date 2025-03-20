import torch as th
from nsfr.utils.common import bool_to_probs

def on_river(player: th.Tensor, river: th.Tensor) -> th.Tensor:
    result = player[..., 1] == river[..., 1]
    return bool_to_probs(result)

def right_of_river(player: th.Tensor, river: th.Tensor) -> th.Tensor:
    player_x = player[..., 1]
    river_x = river[..., 1]
    return bool_to_probs(player_x > river_x)

def left_of_river(player: th.Tensor, river: th.Tensor) -> th.Tensor:
    player_x = player[..., 1]
    river_x = river[..., 1]
    return bool_to_probs(player_x < river_x)

def false_predicate(player: th.Tensor) -> th.Tensor:
    return bool_to_probs(th.tensor([False]))

def close_by_fuel(player: th.Tensor, fuel: th.Tensor) -> th.Tensor:
    return _close_by(player, fuel)

def close_by_enemy_ship(player: th.Tensor, enemy_ship: th.Tensor) -> th.Tensor:
    return _close_by(player, enemy_ship)

def close_by_helicopter(player: th.Tensor, helicopter: th.Tensor) -> th.Tensor:
    return _close_by(player, helicopter)

def close_by_enemy_base(player: th.Tensor, enemy_base: th.Tensor) -> th.Tensor:
    return _close_by(player, enemy_base)

def close_by_bridge(player: th.Tensor, bridge: th.Tensor) -> th.Tensor:
    return _close_by(player, bridge)

def nothing_around(player: th.Tensor) -> th.Tensor:
    return bool_to_probs(th.tensor([True]))

def same_level_river(player: th.Tensor, river: th.Tensor) -> th.Tensor:
    return bool_to_probs(player[..., 2] == river[..., 2])

def higher_than_enemy(player: th.Tensor, enemy_ship: th.Tensor) -> th.Tensor:
    return bool_to_probs(player[..., 2] < enemy_ship[..., 2] - 4)

def lower_than_enemy(player: th.Tensor, enemy_ship: th.Tensor) -> th.Tensor:
    return bool_to_probs(player[..., 2] > enemy_ship[..., 2] + 4)

def higher_than_helicopter(player: th.Tensor, helicopter: th.Tensor) -> th.Tensor:
    return bool_to_probs(player[..., 2] < helicopter[..., 2] - 4)

def lower_than_helicopter(player: th.Tensor, helicopter: th.Tensor) -> th.Tensor:
    return bool_to_probs(player[..., 2] > helicopter[..., 2] + 4)

def _close_by(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    th_val = 48
    player_x = player[..., 1]
    player_y = player[..., 2]
    obj_x = obj[..., 1]
    obj_y = obj[..., 2]
    obj_prob = obj[:, 0]
    dist = (player_x - obj_x).pow(2) + (player_y - obj_y).pow(2)
    return bool_to_probs(dist.sqrt() < th_val) * obj_prob

def _not_close_by(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_x = player[..., 1]
    player_y = player[..., 2]
    obj_x = obj[..., 1]
    obj_y = obj[..., 2]
    result = th.clip((abs(player_x - obj_x) + abs(player_y - obj_y) - 64) / 64, 0, 1)
    return result
