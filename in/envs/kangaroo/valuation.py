import torch as th

from nsfr.utils.common import bool_to_probs


def nothing_around(objs: th.Tensor) -> th.Tensor:
    # target objects: fruit, bell, monkey, fallingcoconut, throwncoconut
    fruits = objs[:, 2:5]
    bell = objs[:, 5].unsqueeze(1)
    monkey = objs[:, 32:36]
    falling_coconut = objs[:, 36].unsqueeze(1)
    thrown_coconut = objs[:, 37:40]
    # target_objs = th.cat([fruits, bell, monkey, falling_coconut, thrown_coconut], dim=1)
    target_objs = th.cat([monkey, falling_coconut, thrown_coconut], dim=1)
    players = objs[:, 0].unsqueeze(1).expand(-1, target_objs.size(1), -1)

    # batch_size * num_target_objs
    probs = th.stack([_close_by(players[:, i, :], target_objs[:, i, :]) for i in range(target_objs.size(1))], dim=1)

    max_closeby_prob, _ = probs.max(dim=1)
    result = (1.0 - max_closeby_prob).float()
    return result


def _in_field(obj: th.Tensor) -> th.Tensor:
    x = obj[..., 1]
    prob = obj[:, 0]
    return bool_to_probs(th.logical_and(16 <= x, x <= 16 + 128)) * prob


def _on_platform(obj1: th.Tensor, obj2: th.Tensor) -> th.Tensor:
    """True iff obj1 is 'on' obj2."""
    obj1_y = obj1[..., 2]
    obj2_y = obj2[..., 2]
    obj1_prob = obj1[:, 0]
    obj2_prob = obj2[:, 0]
    result = th.logical_and(12 < obj2_y - obj1_y, obj2_y - obj1_y < 60)
    return bool_to_probs(result) * obj1_prob * obj2_prob


def on_pl_player(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _on_platform(player, obj)


def on_pl_ladder(ladder: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _on_platform(ladder, obj)


def on_pl_fruit(fruit: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _on_platform(fruit, obj)


def on_pl_bell(bell: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _on_platform(bell, obj)


def on_ladder(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_x = player[..., 1]
    obj_x = obj[..., 1]
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    obj_prob = obj[:, 0]
    x_prob = bool_to_probs(abs(player_x - obj_x) < 4)
    return x_prob
    # y_prob = bool_to_probs(obj_y > player_y - 8)
    # return  x_prob * y_prob * obj_prob * same_level_ladder(player, obj)


def left_of_ladder(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is 'left of' the object."""
    player_x = player[..., 1]
    obj_x = obj[..., 1]
    obj_prob = obj[:, 0]
    return bool_to_probs(3 < obj_x - player_x) * obj_prob * same_level_ladder(player, obj)


def right_of_ladder(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is 'right of' the object."""
    player_x = player[..., 1]
    obj_x = obj[..., 1]
    obj_prob = obj[:, 0]
    return bool_to_probs(3 < player_x - obj_x) * obj_prob * same_level_ladder(player, obj)


def same_depth_diver(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    obj_prob = obj[:, 0]
    return bool_to_probs(abs(player_y - obj_y) < 6) * obj_prob


def same_depth_missile(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    obj_prob = obj[:, 0]
    return bool_to_probs(abs(player_y - obj_y) < 6) * obj_prob


def deeper_than_enemy(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is (significantly) 'deeper than' the object."""
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    obj_prob = obj[:, 0]
    return bool_to_probs(player_y > obj_y + 4) * obj_prob


def deeper_than_diver(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is (significantly) 'deeper than' the object."""
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    obj_prob = obj[:, 0]
    return bool_to_probs(player_y > obj_y + 4) * obj_prob


def higher_than_enemy(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is (significantly) 'higher than' the object."""
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    obj_prob = obj[:, 0]
    return bool_to_probs(player_y < obj_y - 4) * obj_prob


def higher_than_diver(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is (significantly) 'higher than' the object."""
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    obj_prob = obj[:, 0]
    return bool_to_probs(player_y < obj_y - 4) * obj_prob


def same_level_fruit(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _same_level(player, obj)


def same_level_bell(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _same_level(player, obj)


def same_level_ladder(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    obj1_y = player[..., 2] + 10
    obj2_y = obj[..., 2]

    is_3rd_level = th.logical_and(th.logical_and(28 < obj1_y, obj1_y < 76), th.logical_and(28 < obj2_y, obj2_y < 76))
    is_2nd_level = th.logical_and(th.logical_and(76 < obj1_y, obj1_y < 124), th.logical_and(76 < obj2_y, obj2_y < 124))
    is_1st_level = th.logical_and(th.logical_and(124 < obj1_y, obj1_y < 172),
                                  th.logical_and(124 < obj2_y, obj2_y < 172))

    is_same_level = th.logical_or(is_3rd_level, th.logical_or(is_2nd_level, is_1st_level))
    return bool_to_probs(is_same_level)


def _same_level(obj1: th.Tensor, obj2: th.Tensor) -> th.Tensor:
    obj1_y = obj1[..., 2]
    obj2_y = obj2[..., 2]

    is_3rd_level = th.logical_and(th.logical_and(28 < obj1_y, obj1_y < 76), th.logical_and(28 < obj2_y, obj2_y < 76))
    is_2nd_level = th.logical_and(th.logical_and(76 < obj1_y, obj1_y < 124), th.logical_and(76 < obj2_y, obj2_y < 124))
    is_1st_level = th.logical_and(th.logical_and(124 < obj1_y, obj1_y < 172),
                                  th.logical_and(124 < obj2_y, obj2_y < 172))

    is_same_level = th.logical_or(is_3rd_level, th.logical_or(is_2nd_level, is_1st_level))
    return bool_to_probs(is_same_level)


def close_by_fruit(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _close_by(player, obj) * _same_level(player, obj)


def close_by_bell(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _close_by(player, obj) * _same_level(player, obj)


def close_by_throwncoconut(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _close_by(player, obj)


def close_by_fallingcoconut(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _close_by(player, obj)


def close_by_monkey(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _close_by(player, obj)


def close_by_coconut(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _close_by(player, obj)


def close_by_diver(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _close_by(player, obj)


def _close_by(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    th = 32
    player_x = player[:, 1]
    player_y = player[:, 2]
    obj_x = obj[:, 1]
    obj_y = obj[:, 2]
    obj_prob = obj[:, 0]
    x_dist = (player_x - obj_x).pow(2)
    y_dist = (player_y - obj_y).pow(2)
    dist = (x_dist + y_dist).sqrt()
    # dist = (player[:, 1:2] - obj[:, 1:2]).pow(2).sum(1).sqrt()
    return bool_to_probs(dist < th) * obj_prob * _in_field(obj)
    # result = th.clip((128 - abs(player_x - obj_x) - abs(player_y - obj_y)) / 128, 0, 1) * obj_prob
    # return result


def _not_close_by(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_x = player[..., 1]
    player_y = player[..., 2]
    obj_x = obj[..., 1]
    obj_y = obj[..., 2]
    result = th.clip((abs(player_x - obj_x) + abs(player_y - obj_y) - 64) / 64, 0, 1)
    return result


def not_close_by_missile(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _not_close_by(player, obj)


def not_close_by_enemy(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _not_close_by(player, obj)


def left_of_diver(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is 'left of' the object."""
    player_x = player[..., 1]
    obj_x = obj[..., 1]
    obj_prob = obj[:, 0]
    return bool_to_probs(player_x < obj_x) * obj_prob


def right_of_diver(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is 'right of' the object."""
    player_x = player[..., 1]
    obj_x = obj[..., 1]
    obj_prob = obj[:, 0]
    return bool_to_probs(player_x > obj_x) * obj_prob


def oxygen_low(oxygen_bar: th.Tensor) -> th.Tensor:
    """True iff oxygen bar is below 16/64."""
    # result = oxygen_bar[..., 1] < 16
    result = oxygen_bar[..., 1] < 24
    return bool_to_probs(result)


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
