import torch as th

from nsfr.utils.common import bool_to_probs


def _in_field(obj: th.Tensor) -> th.Tensor:
    x = obj[..., 1]
    field_left, field_right = 16, 144
    inside_field = sigmoid_smoothing((x >= field_left) & (x <= field_right), 6.0)
    return inside_field


def on_ladder(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_x = player[..., 1]
    obj_x = obj[..., 1]
    is_on_ladder = sigmoid_smoothing(abs(player_x - obj_x) < 4, temperature=5.0)
    return is_on_ladder


def same_level_ladder(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    obj1_y = player[..., 2] + 8
    obj2_y = obj[..., 2]
    z = 5  # floor thickness
    # eps = 12

    is_1st_level = th.logical_and(th.logical_and(176 - 5 < obj1_y, obj1_y < 193 + 8),
                                  th.logical_and(176 < obj2_y, obj2_y < 193))
    is_2nd_level = th.logical_and(th.logical_and(148 - 5 < obj1_y, obj1_y < 165 + 8),
                                  th.logical_and(148 < obj2_y, obj2_y < 165))
    is_3rd_level = th.logical_and(th.logical_and(120 - 5 < obj1_y, obj1_y < 137 + 8),
                                  th.logical_and(120 < obj2_y, obj2_y < 137))
    is_4th_level = th.logical_and(th.logical_and(92 - 5 < obj1_y, obj1_y < 109 + 8),
                                  th.logical_and(92 < obj2_y, obj2_y < 109))
    is_5th_level = th.logical_and(th.logical_and(64 - 5 < obj1_y, obj1_y < 81 + 8),
                                  th.logical_and(64 < obj2_y, obj2_y < 81))
    is_6th_level = th.logical_and(th.logical_and(40 - 5 < obj1_y, obj1_y < 57 + 8),
                                  th.logical_and(40 < obj2_y, obj2_y < 57))

    is_same_level_1 = th.logical_or(is_3rd_level, th.logical_or(is_2nd_level, is_1st_level))
    is_same_level_2 = th.logical_or(is_4th_level, th.logical_or(is_5th_level, is_6th_level))
    is_same_level = th.logical_or(is_same_level_1, is_same_level_2)
    return sigmoid_smoothing(is_same_level, temperature=6.0)


def obj_above_ladder(obj: th.Tensor, ladder: th.Tensor, obj_h: int) -> th.Tensor:
    LADDER_HEIGHT = 17
    ladder_x, ladder_y = ladder[..., 1], ladder[..., 2]
    ladder_top_y = ladder_y + (LADDER_HEIGHT / 2) + 5
    obj_x, obj_y = obj[..., 1], obj[..., 2]
    obj_down_y = obj_y - (obj_h / 2)
    x_dist = (ladder_x - obj_x).pow(2)
    y_dist = (ladder_top_y - obj_down_y).pow(2)
    dist = (x_dist + y_dist).sqrt()
    return sigmoid_smoothing((dist < 16), temperature=8.0)


def no_obj_above(obj: th.Tensor, obj_h: int, player: th.Tensor) -> th.Tensor:
    PLAYER_HEIGHT = 17
    player_x, player_y = player[..., 1], player[..., 2]
    obj_x, obj_y = obj[..., 1], obj[..., 2]
    player_top_y = player_y + (PLAYER_HEIGHT / 2)
    obj_down_y = obj_y - (obj_h / 2)
    x_dist = (player_x - obj_x).pow(2)
    y_dist = (player_top_y - obj_down_y).pow(2)
    dist = (x_dist + y_dist).sqrt()
    return sigmoid_smoothing((dist > 24), temperature=8.0)


def safe_to_climb(barrel: th.Tensor, donkeykong: th.Tensor, player: th.Tensor) -> th.Tensor:
    barrel_h = 8
    donkeykong_h = 19
    no_hazard_barrel = no_obj_above(barrel, barrel_h, player)
    no_hazard_dk = no_obj_above(donkeykong, donkeykong_h, player)
    safe_condition = th.logical_or(no_hazard_barrel > 0.6, no_hazard_dk > 0.6)
    safe_probability = sigmoid_smoothing(safe_condition, temperature=7.0)
    return safe_probability


def _close_by(player: th.Tensor, obj: th.Tensor, th: int = 32) -> th.Tensor:
    player_x = player[:, 1]
    player_y = player[:, 2]
    obj_x = obj[:, 1]
    obj_y = obj[:, 2]
    x_dist = (player_x - obj_x).pow(2)
    y_dist = (player_y - obj_y).pow(2)
    dist = (x_dist + y_dist).sqrt()
    prob = bool_to_probs(dist < th)
    return prob


def _not_close_by(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_x = player[..., 1]
    player_y = player[..., 2]
    obj_x = obj[..., 1]
    obj_y = obj[..., 2]
    result = th.clip((abs(player_x - obj_x) + abs(player_y - obj_y) - 64) / 64, 0, 1)
    return result


def on_left(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """
    Check if the object is to the left of the player.
    """
    player_x = player[:, 1]
    obj_x = obj[:, 1]
    left = player_x - obj_x
    return sigmoid_smoothing(left > 0, temperature=6.0)


def on_right(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """
    Check if the object is to the right of the player.
    """
    player_x = player[:, 1]
    obj_x = obj[:, 1]
    right = obj_x - player_x
    prob = sigmoid_smoothing(right > 0, temperature=6.0)
    return prob


def above(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """
    Compute probability that an object is above the player using sigmoid smoothing,
    scaled to prevent overpowering `close_by()`
    """
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    above_prob = th.sigmoid(6.0 * (obj_y > player_y) * 0.5)
    return above_prob


def close_by_barrel(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _close_by(player, obj, th=25) * same_level(player, obj)


def close_by_hammer(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _close_by(player, obj, th=32) * same_level(player, obj)


def has_hammer(player: th.Tensor, hammer: th.Tensor) -> th.Tensor:
    """
    Dynamically check if the player has collected the hammer based on proximity.
    """
    # Compute half dimensions and sum them to get the thresholds
    horizontal_threshold = (8 / 2) + (4 / 2)  # 4 + 2 = 6
    vertical_threshold = (17 / 2) + (7 / 2)  # 8.5 + 3.5 = 12
    player_x, player_y = player[..., 1], player[..., 2]
    hammer_x, hammer_y = hammer[..., 1], hammer[..., 2]
    horizontal_overlap = th.abs(player_x - hammer_x) < horizontal_threshold
    vertical_overlap = th.abs(player_y - hammer_y) < vertical_threshold
    collision = th.logical_and(horizontal_overlap, vertical_overlap)
    collected_prob = sigmoid_smoothing(collision, temperature=8.0)
    return collected_prob


def has_no_hammer(player: th.Tensor, hammer: th.Tensor) -> th.Tensor:
    no_hammer = 1 - has_hammer(player, hammer)
    return no_hammer


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


def sigmoid_smoothing(bool_tensor: th.Tensor, temperature: float = 5.0) -> th.Tensor:
    """
    Apply sigmoid smoothing to a boolean tensor, converting True/False into soft probabilities.

    :param bool_tensor: Boolean tensor indicating condition
    :param temperature: Controls softness of probability conversion (higher = more binary-like).
    :return: Soft probability tensor (0.0 to 1.0).
    """
    return th.sigmoid(temperature * (bool_tensor.float() - 0.5))


def same_level(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    obj1_y = player[..., 2]
    obj2_y = obj[..., 2]
    z = 5
    is_1st_level = th.logical_and(th.logical_and(176 - 5 < obj1_y, obj1_y < 193 + 8),
                                  th.logical_and(176 < obj2_y, obj2_y < 193))
    is_2nd_level = th.logical_and(th.logical_and(148 - 5 < obj1_y, obj1_y < 165 + 8),
                                  th.logical_and(148 < obj2_y, obj2_y < 165))
    is_3rd_level = th.logical_and(th.logical_and(120 - 5 < obj1_y, obj1_y < 137 + 8),
                                  th.logical_and(120 < obj2_y, obj2_y < 137))
    is_4th_level = th.logical_and(th.logical_and(92 - 5 < obj1_y, obj1_y < 109 + 8),
                                  th.logical_and(92 < obj2_y, obj2_y < 109))
    is_5th_level = th.logical_and(th.logical_and(64 - 5 < obj1_y, obj1_y < 81 + 8),
                                  th.logical_and(64 < obj2_y, obj2_y < 81))
    is_6th_level = th.logical_and(th.logical_and(40 - 5 < obj1_y, obj1_y < 57 + 8),
                                  th.logical_and(40 < obj2_y, obj2_y < 57))

    is_same_level_1 = th.logical_or(is_3rd_level, th.logical_or(is_2nd_level, is_1st_level))
    is_same_level_2 = th.logical_or(is_4th_level, th.logical_or(is_5th_level, is_6th_level))
    is_same_level = th.logical_or(is_same_level_1, is_same_level_2)
    return sigmoid_smoothing(is_same_level, temperature=6.0)
