import torch as th
from nsfr.utils.common import bool_to_probs

""" in ocatari/ram/freeway.py :
    MAX_NB_OBJECTS = {'Chicken': 2, 'Car': 10}
    
"""

# def type(z: th.Tensor, a: th.Tensor) -> th.Tensor:
#     z_type = z[:, 0:2]  # [1, 0, 0, 0] * [1.0, 0, 0, 0] .sum = 0.0  type(obj1, key):0.0
#     prob = (a * z_type).sum(dim=1)
#     return prob


def closeby(car: th.Tensor, player: th.Tensor) -> th.Tensor:
    """Determines if a car is close to the player.

    Args:
        car (th.Tensor): Tensor representing the car's position data
        player (th.Tensor): Tensor representing the player's position data

    Returns:
        th.Tensor: Probability tensor indicating if objects are close (True) or not (False)
    """
    car_xy = car[:, -2:]
    player_xy = player[:, -2:]

    dis_x = abs(car_xy[:, 0] - player_xy[:, 0]) / 171
    dis_y = abs(car_xy[:, 1] - player_xy[:, 1]) / 171

    result = bool_to_probs((dis_x < 2.5) & (dis_y <= 0.1))

    return result


def on_left(car: th.Tensor, player: th.Tensor):
    """Determines if a car is to the left of the player.

    Args:
        car (th.Tensor): Tensor representing the car's position data
        player (th.Tensor): Tensor representing the player's position data

    Returns:
        th.Tensor: Probability tensor indicating if the player is on the left of the car (True) or not (False)
    """
    car_x = car[:, -2]
    player_x = player[:, -2]
    diff = player_x - car_x
    result = bool_to_probs(diff > 0)
    return result


def on_right(car: th.Tensor, player: th.Tensor):
    """Determines if a car is to the right of the player.

    Args:
        car (th.Tensor): Tensor representing the car's position data
        player (th.Tensor): Tensor representing the player's position data

    Returns:
        th.Tensor: Probability tensor indicating if the player is on the right of the car (True) or not (False)
    """
    car_x = car[:, -2]
    player_x = player[:, -2]
    diff = player_x - car_x
    result = bool_to_probs(diff < 0)
    return result


def same_row(car: th.Tensor, player: th.Tensor):
    """Determines if the player is in approximately the same row as the car.

    Args:
        car (th.Tensor): Tensor representing the car's position data
        player (th.Tensor): Tensor representing the player's position data

    Returns:
        th.Tensor: Probability tensor indicating if objects are in same row (True) or not (False)
    """
    car_y = car[:, -1]
    player_y = player[:, -1]
    diff = abs(player_y - car_y)
    result = bool_to_probs(diff < 6)
    return result


def above_row(car: th.Tensor, player: th.Tensor):
    """Determines if a car is in a row above the player within a specific range.

    Args:
        car (th.Tensor): Tensor representing the car's position data
        player (th.Tensor): Tensor representing the player's position data

    Returns:
        th.Tensor: Probability tensor indicating if the player is above the car (True) or not (False)
    """
    car_y = car[:, -1]
    player_y = player[:, -1]
    diff = player_y - car_y
    result1 = bool_to_probs(diff < 23)
    result2 = bool_to_probs(diff > 4)
    return result1 * result2


def top5car(car: th.Tensor):
    """Determines if a car is in the top 5 rows of the game space.
    Important because the cars in the top 5 rows only drive right to left.

    Args:
        car (th.Tensor): Tensor representing the car's position data

    Returns:
        th.Tensor: Probability tensor indicating if car is in top 5 rows (True) or not (False)
    """
    car_y = car[:, -1]
    result = bool_to_probs(car_y > 100)
    return result


def bottom5car(car: th.Tensor):
    """Determines if a car is in the bottom 5 rows of the game space.
    Important because the cars in the bottom 5 rows only drive left to right.

    Args:
        car (th.Tensor): Tensor representing the car's position data

    Returns:
        th.Tensor: Probability tensor indicating if car is in bottom 5 rows (True) or not (False)
    """
    car_y = car[:, -1]
    result = bool_to_probs(car_y < 100)
    return result
