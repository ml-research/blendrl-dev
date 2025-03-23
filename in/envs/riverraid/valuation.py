import torch as th
from nsfr.utils.common import bool_to_probs


def on_river(player: th.Tensor, river: th.Tensor) -> th.Tensor:
    player_x, player_y = player[..., 1], player[..., 2]
    river_x, river_y = river[..., 1], river[..., 2]
    river_prob = river[:, 0]
    return bool_to_probs(abs(player_x - river_x) < 15) * river_prob


def right_of_river(player: th.Tensor, river: th.Tensor) -> th.Tensor:
    player_x = player[..., 1]
    river_x = river[..., 1]
    river_prob = river[:, 0]
    return bool_to_probs(player_x > river_x) * river_prob


def left_of_river(player: th.Tensor, river: th.Tensor) -> th.Tensor:
    player_x = player[..., 1]
    river_x = river[..., 1]
    river_prob = river[:, 0]
    return bool_to_probs(player_x < river_x) * river_prob


'''def left_of_river(player: th.Tensor, river: th.Tensor) -> th.Tensor:
    player_y = player[..., 2]
    river_y = river[..., 2]
    river_prob = river[:, 0]
    return bool_to_probs(player_y < river_y) * river_prob

def right_of_river(player: th.Tensor, river: th.Tensor) -> th.Tensor:
    player_y = player[..., 2]
    river_y = river[..., 2]
    river_prob = river[:, 0]
    return bool_to_probs(player_y > river_y) * river_prob'''


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


def nothing_around(objs: th.Tensor) -> th.Tensor:
    enemies = th.cat([objs[:, 5:10], objs[:, 19:22]], dim=1)  # Enemy ships & helicopters
    near_enemies = th.sum(enemies[:, :, 0], dim=1) == 0
    return bool_to_probs(near_enemies)


def same_level_river(player: th.Tensor, river: th.Tensor) -> th.Tensor:
    player_x = player[..., 2]
    river_x = river[..., 2]
    river_prob = river[:, 0]
    return bool_to_probs(abs(player_x - river_x) < 5) * river_prob

def same_level_enemy(player: th.Tensor, enemy: th.Tensor) -> th.Tensor:
    player_x = player[..., 2]
    enemy_x = enemy[..., 2]
    enemy_prob = enemy[:, 0]
    return bool_to_probs(abs(player_x - enemy_x) < 5) * enemy_prob

def same_level_helicopter(player: th.Tensor, helicopter: th.Tensor) -> th.Tensor:
    player_x = player[..., 2]
    helicopter_x = helicopter[..., 2]
    helicopter_prob = helicopter[:, 0]
    return bool_to_probs(abs(player_x - helicopter_x) < 5) * helicopter_prob


def higher_than_enemy(player: th.Tensor, enemy: th.Tensor) -> th.Tensor:
    return _higher_than(player, enemy)


def lower_than_enemy(player: th.Tensor, enemy: th.Tensor) -> th.Tensor:
    return _lower_than(player, enemy)


def higher_than_helicopter(player: th.Tensor, helicopter: th.Tensor) -> th.Tensor:
    return _higher_than(player, helicopter)


def lower_than_helicopter(player: th.Tensor, helicopter: th.Tensor) -> th.Tensor:
    return _lower_than(player, helicopter)


def _close_by(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    threshold = 32
    player_x, player_y = player[..., 1], player[..., 2]
    obj_x, obj_y = obj[..., 1], obj[..., 2]
    obj_prob = obj[:, 0]
    distance = ((player_x - obj_x) ** 2 + (player_y - obj_y) ** 2).sqrt()
    return bool_to_probs(distance < threshold) * obj_prob


def _higher_than(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    obj_prob = obj[:, 0]
    return bool_to_probs(player_y < obj_y - 4) * obj_prob


def _lower_than(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    obj_prob = obj[:, 0]
    return bool_to_probs(player_y > obj_y + 4) * obj_prob


def test_predicate_global(global_state: th.Tensor) -> th.Tensor:
    result = global_state[..., 0, 2] < 100
    return bool_to_probs(result)


def test_predicate_object(agent: th.Tensor) -> th.Tensor:
    result = agent[..., 2] < 100
    return bool_to_probs(result)


def true_predicate(agent: th.Tensor) -> th.Tensor:
    return bool_to_probs(th.tensor([True]))


def false_predicate(agent: th.Tensor) -> th.Tensor:
    return bool_to_probs(th.tensor([False]))
