import torch as th

from nsfr.utils.common import bool_to_probs

""" in ocatari/ram/kangaroo.py :
        MAX_ESSENTIAL_OBJECTS = {
            'Player': 1, (0)
            'Child': 1, (1)
            'Fruit': 3, (2)
            'Bell': 1, (5)
            'Platform': 20,
            'Ladder': 6,
            'Monkey': 4,
            'FallingCoconut': 1,
            'ThrownCoconut': 3,
            'Life': 8,
            'Time': 1,}       
"""


# def nothing_around(objs: th.Tensor) -> th.Tensor:
#     # target objects: fruit, bell, monkey, fallingcoconut, throwncoconut
#     fruits = objs[:, 2:5]
#     bell = objs[:, 5].unsqueeze(1)
#     monkey = objs[:, 32:36]
#     falling_coconut = objs[:, 36].unsqueeze(1)
#     thrown_coconut = objs[:, 37:40]
#     # target_objs = th.cat([fruits, bell, monkey, falling_coconut, thrown_coconut], dim=1)
#     target_objs = th.cat([monkey, falling_coconut, thrown_coconut], dim=1)
#     players = objs[:, 0].unsqueeze(1).expand(-1, target_objs.size(1), -1)
#
#     # batch_size * num_target_objs
#     probs = th.stack([_close_by(players[:, i, :], target_objs[:, i, :], ) for i in range(target_objs.size(1))], dim=1)
#
#     max_closeby_prob, _ = probs.max(dim=1)
#     result = (1.0 - max_closeby_prob).float()
#     return result


def _in_field(obj: th.Tensor) -> th.Tensor:
    x = obj[..., 1]
    field_left, field_right = 16, 144
    inside_field = sigmoid_smoothing((x >= field_left) & (x <= field_right), 6.0)
    return inside_field



# ---------------------------------------------------------------------------------
# platform logic
def _on_platform(obj1: th.Tensor, obj2: th.Tensor) -> th.Tensor:
    """True iff obj1 is 'on' obj2."""
    obj1_y = obj1[..., 2]
    obj2_y = obj2[..., 2]
    obj1_prob = obj1[:, 0]
    obj2_prob = obj2[:, 0]
    result = th.logical_and(12 < obj2_y - obj1_y, obj2_y - obj1_y < 60)
    return bool_to_probs(result) * obj1_prob * obj2_prob


def is_player_on_ladder(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """
    Compute the probability that the player is on a ladder while considering jumping behavior.
    :param temperature: Controls probability softness.
    :return: Probability Tensor (0.0 to 1.0).
    """
    PLAYER_WIDTH = 8
    PLAYER_HEIGHT_STANDING = 24
    LADDER_WIDTH = 8
    LADDER_HEIGHT = 35
    temperature = 6.0
    player_x, player_y = player[..., 1], player[..., 2]
    obj_x, obj_y = obj[..., 1], obj[..., 2]
    # Probability stays high unless the player climbs too high**
    # The player should still be on the ladder as long as their top isn't above the ladder's top
    player_top_y = player_y - (PLAYER_HEIGHT_STANDING / 2)
    ladder_top_y = obj_y + (LADDER_HEIGHT / 2)
    y_overlap = th.sigmoid(temperature * (ladder_top_y - player_top_y))  # Higher when below ladder top
    # for initial alignment, not for climbing**
    x_tolerance = (PLAYER_WIDTH / 2) + (LADDER_WIDTH / 2)
    x_alignment = th.sigmoid(-temperature * (th.abs(player_x - obj_x) - x_tolerance))
    # Use X only if the player is not yet climbing**
    climbing_mask = y_overlap > 0.8  # If already on ladder, ignore X alignment
    final_prob = th.where(climbing_mask, y_overlap, y_overlap * x_alignment)
    return final_prob


def same_level_ladder(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    obj1_y = player[..., 2] + 10
    obj2_y = obj[..., 2]

    is_3rd_level = (28 <= obj1_y) & (obj1_y <= 76) & (28 <= obj2_y) & (obj2_y <= 76)
    is_2nd_level = (76 <= obj1_y) & (obj1_y <= 124) & (76 <= obj2_y) & (obj2_y <= 124)
    is_1st_level = (124 <= obj1_y) & (obj1_y <= 172) & (124 <= obj2_y) & (obj2_y <= 172)

    # Determine if they are on the same level
    is_same_level = is_3rd_level | is_2nd_level | is_1st_level  # Bitwise OR for efficiency

    return sigmoid_smoothing(is_same_level, temperature=6.0)  # Smooth probability output


# ---------------------------------------------------------------------------------
# fruit logic
def close_by_fruit(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _close_by(player, obj) * same_level(player, obj)



# ---------------------------------------------------------------------------------
# bell logic
def close_by_bell(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _close_by(player, obj) * same_level(player, obj)


# ---------------------------------------------------------------------------------
# monkey logic
def close_by_monkey(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _close_by(player, obj) * same_level(player, obj)


# ---------------------------------------------------------------------------------
# coconut logic
def close_by_coconut_combi(player: th.Tensor, *objects: th.Tensor) -> th.Tensor:
    results = []
    for obj in objects:
        results.append(_close_by(player, obj))
    # Stack the results to combine the probabilities across different objects.
    stacked = th.stack(results)  # Shape: [number_of_objects, ...]
    # Compute the union probability assuming independence:
    return 1 - th.prod(1 - stacked, dim=0)


def close_by_coconut(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """
    Compute probability that a player is close to a coconut using sigmoid smoothing.
    :param temperature: Controls probability smoothness.
    :return: Probability Tensor (0.0 to 1.0).
    """
    return _close_by(player, obj)


# ---------------------------------------------------------------------------------
# general methods
def _close_by(player: th.Tensor, obj: th.Tensor, temperature: float = 6.0) -> th.Tensor:
    """
    Compute probability that a player is close to an object using sigmoid smoothing.
    :param temperature: Controls probability smoothness.
    :return: Probability Tensor (0.0 to 1.0).
    """
    th = 32
    player_x = player[:, 1]
    player_y = player[:, 2]
    obj_x = obj[:, 1]
    obj_y = obj[:, 2]
    x_dist = (player_x - obj_x).pow(2)
    y_dist = (player_y - obj_y).pow(2)
    dist = (x_dist + y_dist).sqrt()
    return sigmoid_smoothing(dist < th, temperature) * _in_field(obj)



#def _close_by(player: th.Tensor, obj: th.Tensor, temperature: float = 5.0) -> th.Tensor:
    """
    Compute probability that a player is close to an object using sigmoid smoothing.
    :param temperature: Controls probability smoothness.
    :return: Probability Tensor (0.0 to 1.0).
    """
    # PLAYER_WIDTH, PLAYER_HEIGHT = 8, 24
    # player_x, player_y = player[:, 1], player[:, 2]
    # obj_x, obj_y = obj[:, 1], obj[:, 2]
    # min_dist_x = th.tensor((PLAYER_WIDTH / 2) + (obj_w / 2), dtype=th.float32)
    # min_dist_y = th.tensor((PLAYER_HEIGHT / 2) + (obj_h / 2), dtype=th.float32)
    # min_dist = (min_dist_x ** 2 + min_dist_y ** 2).sqrt()
    # dist = ((player_x - obj_x).pow(2) + (player_y - obj_y).pow(2)).sqrt()
    # close_prob = th.sigmoid(temperature * (min_dist * 1.2 - dist))
    # return close_prob


def same_level(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    obj1_y = player[..., 2]
    obj2_y = obj[..., 2]

    is_3rd_level = (28 <= obj1_y) & (obj1_y <= 76) & (28 <= obj2_y) & (obj2_y <= 76)
    is_2nd_level = (76 <= obj1_y) & (obj1_y <= 124) & (76 <= obj2_y) & (obj2_y <= 124)
    is_1st_level = (124 <= obj1_y) & (obj1_y <= 172) & (124 <= obj2_y) & (obj2_y <= 172)

    # Determine if they are on the same level
    is_same_level = is_3rd_level | is_2nd_level | is_1st_level  # Bitwise OR for efficiency

    return sigmoid_smoothing(is_same_level, temperature=6.0)  # Smooth probability output


def is_lower(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """
    Check if one or more objects are horizontally lower than the player.
    """
    player_y = player[:, 2]
    obj_y = obj[:, 2]
    return sigmoid_smoothing(obj_y < player_y, temperature=6.0)


def is_lower_combi(player: th.Tensor, *objects: th.Tensor) -> th.Tensor:
    """
    Check if one or more objects are horizontally lower than the player using union probability.
    """
    results = []
    # Compute individual probabilities using the is_lower function.
    for obj in objects:
        results.append(is_lower(player, obj))
    # Stack the individual probability tensors.
    stacked = th.stack(results)  # Shape: [number_of_objects, ...]
    # Compute the union probability assuming independence:
    # P(at least one) = 1 - (1 - p1) * (1 - p2) * ... * (1 - p_n)
    combined_prob = 1 - th.prod(1 - stacked, dim=0)
    return combined_prob


def too_high_to_jump(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """
    Check if one or more objects are horizontally on the same level or slightly higher than the player.
    """
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    is_high = obj_y >= player_y
    return sigmoid_smoothing(is_high, temperature=6.0) * same_level(player, obj)


def too_high_to_jump_combi(player: th.Tensor, *objects: th.Tensor) -> th.Tensor:
    """
    Check if one or more objects are on the same level or slightly higher than the player
    (i.e., too high to jump) using union probability to combine individual probabilities.
    """
    results = []
    for obj in objects:
        results.append(too_high_to_jump(player, obj))
    stacked = th.stack(results)  # Shape: [number_of_objects, ...]
    # Combine the probabilities using the union probability formula:
    # P(at least one) = 1 - ∏ (1 - p_i)
    combined_prob = 1 - th.prod(1 - stacked, dim=0)
    return combined_prob


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
    return sigmoid_smoothing(obj_x < player_x, temperature=6.0)


def on_right(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """
    Check if the object is to the right of the player.
    """
    player_x = player[:, 1]
    obj_x = obj[:, 1]
    return sigmoid_smoothing(obj_x > player_x, temperature=6.0)


def on_right_combi(player: th.Tensor, *objects: th.Tensor) -> th.Tensor:
    """
    Check if one or more objects are to the right of the player.
    """
    temperature = 6.0
    player_x = player[..., 1]
    results = []
    for obj in objects:
        obj_x = obj[..., 1]
        is_on_right = sigmoid_smoothing(obj_x > player_x, temperature)
        results.append(is_on_right)
    return th.stack(results).max(dim=0).values


def on_left_combi(player: th.Tensor, *objects: th.Tensor) -> th.Tensor:
    """
    Check if one or more objects are to the left of the player.
    """
    temperature = 6.0
    player_x = player[..., 1]
    results = []
    for obj in objects:
        obj_x = obj[..., 1]
        is_on_left = sigmoid_smoothing(obj_x < player_x, temperature)
        results.append(is_on_left)
    return th.stack(results).max(dim=0).values


def above(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """
    Compute probability that an object is above the player using sigmoid smoothing,
    scaled to prevent overpowering `close_by()`
    """
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    above_prob = th.sigmoid(6.0 * (obj_y > player_y) * 0.5)
    return above_prob


def above_combi(player: th.Tensor, *objects: th.Tensor, vertical_tolerance: float = 60) -> th.Tensor:
    """
    Check if one or more objects are above the player.
    """
    temperature = 6.0
    player_y = player[..., 2]
    results = []
    for obj in objects:
        obj_y = obj[..., 2]
        # Compute the probability that the object is above the player.
        is_above = sigmoid_smoothing(obj_y > player_y, temperature)
        results.append(is_above)
    # Stack the probabilities into a single tensor along a new dimension.
    stacked = th.stack(results)  # Shape: [number_of_objects, ...]
    # Probability that at least one object is above = 1 - (1 - p1)*(1 - p2)*...*(1 - p_n)
    return 1 - th.prod(1 - stacked, dim=0)

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

    :param bool_tensor: Boolean tensor indicating condition (True = overlap, False = no overlap).
    :param temperature: Controls softness of probability conversion (higher = more binary-like).
    :return: Soft probability tensor (0.0 to 1.0).
    """
    return th.sigmoid(temperature * (bool_tensor.float() - 0.5))  # Adaptive smoothing
