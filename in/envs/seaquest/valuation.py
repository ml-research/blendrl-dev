import torch as th

from nsfr.utils.common import bool_to_probs


def full_divers(objs: th.Tensor) -> th.Tensor:
    # objs: batch_size, 43, 4
    # result = th.tensor(False)
    divers_vs = objs[:, -7:-1]
    num_collected_divers = th.sum(divers_vs[:,:,0], dim=1)
    result = num_collected_divers == 6
    return bool_to_probs(result)

def not_full_divers(objs: th.Tensor) -> th.Tensor:
    divers_vs = objs[:, -7:-1]
    num_collected_divers = th.sum(divers_vs[:,:,0], dim=1)
    result = num_collected_divers < 6
    return bool_to_probs(result)

def many_enemies(objs: th.Tensor) -> th.Tensor:
    # enemies_vs = objs[:, ]
    enemies_vs = th.cat([objs[:, 5:30], objs[:, 31:35]], dim=1)
    num_enemies = th.sum(enemies_vs[:,:,0], dim=1)
    result = num_enemies >= 4
    return bool_to_probs(result)


def visible_missile(obj: th.Tensor) -> th.Tensor:
    result = obj[..., 0] == 1
    return bool_to_probs(result)


def visible_enemy(obj: th.Tensor) -> th.Tensor:
    result = obj[..., 0] == 1
    return bool_to_probs(result)


def visible_diver(obj: th.Tensor) -> th.Tensor:
    result = obj[..., 0] == 1
    return bool_to_probs(result)


def facing_left(player: th.Tensor) -> th.Tensor:
    result = player[..., 3] == 12
    return bool_to_probs(result)


def facing_right(player: th.Tensor) -> th.Tensor:
    result = player[..., 3] == 4
    return bool_to_probs(result)


def same_depth_enemy(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    obj_prob = obj[:, 0]
    return bool_to_probs(abs(player_y - obj_y) < 6) * obj_prob


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


def close_by_missile(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _close_by(player, obj)


def close_by_enemy(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _close_by(player, obj)


def close_by_diver(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _close_by(player, obj)


def _close_by(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    th = 48
    player_x = player[..., 1]
    player_y = player[..., 2]
    obj_x = obj[..., 1]
    obj_y = obj[..., 2]
    obj_prob = obj[:, 0]
    dist = (player[:, 1:2] - obj[:, 1:2]).pow(2).sum(1).sqrt()
    return bool_to_probs(dist < th) * obj_prob
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


def left_of_enemy(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is 'left of' the object."""
    player_x = player[..., 1]
    obj_x = obj[..., 1]
    obj_prob = obj[:, 0]
    return bool_to_probs(player_x < obj_x) * obj_prob


def left_of_diver(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is 'left of' the object."""
    player_x = player[..., 1]
    obj_x = obj[..., 1]
    obj_prob = obj[:, 0]
    return bool_to_probs(player_x < obj_x) * obj_prob


def right_of_enemy(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is 'right of' the object."""
    player_x = player[..., 1]
    obj_x = obj[..., 1]
    obj_prob = obj[:, 0]
    return bool_to_probs(player_x > obj_x) * obj_prob


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
