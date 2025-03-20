import torch as th
from nsfr.utils.common import bool_to_probs

""" in ocatari/ram/spaceinvaders.py :
    MAX_NB_OBJECTS = {'Player': 1, 'Shield': 3,
                  'Bullet': 3, 'Satellite': 1, 'Alien': 36}
    
"""

def is_player_under_bullet(player: th.Tensor, bullet: th.Tensor):
    player_x = player[:, -2]
    bullet_x = bullet[:, -2]
    diff = abs(bullet_x - player_x)
    result = bool_to_probs(diff < 4)
    return result

def is_player_under_alien(player: th.Tensor, alien: th.Tensor):
    player_x = player[:, -2]
    alien_x = alien[:, -2]
    diff = abs(alien_x - player_x)
    result = bool_to_probs(diff < 4)
    return result

def is_player_under_shield(player: th.Tensor, shield: th.Tensor):
    player_x = player[:, -2]
    shield_x = shield[:, -2]
    diff = abs(shield_x - player_x)
    result = bool_to_probs(diff < 10)
    return result

def is_player_not_under_shield(player: th.Tensor, shield: th.Tensor):
    player_x = player[:, -2]
    shield_x = shield[:, -2]
    diff = abs(shield_x - player_x)
    result = 1 - bool_to_probs(diff < 10)
    return result

def is_player_under_satellite(player: th.Tensor, satellite: th.Tensor) -> th.Tensor:
    player_x = player[:, -2]
    satellite_x = satellite[:, -2]
    diff = abs(satellite_x - player_x)
    result = bool_to_probs(diff < 8)
    return result

def close_to_bullet(player: th.Tensor, bullet: th.Tensor) -> th.Tensor:
    player_xy = player[:, -2:]
    bullet_xy = bullet[:, -2:]

    dis_x = abs(player_xy[:, 0] - bullet_xy[:, 0])
    dis_y = abs(player_xy[:, 1] - bullet_xy[:, 1]) 

    result = bool_to_probs((dis_x < 6) & (dis_y < 20))

    return result

def on_left_shield(player: th.Tensor, shield: th.Tensor) -> th.Tensor:
    """
    Check if the object is to the left of the player.
    """
    player_x = player[:, -2]
    shield_x = shield[:, -2]
    diff = shield_x - player_x
    result = bool_to_probs(diff > 0)
    return result

def on_right_shield(player: th.Tensor, shield: th.Tensor) -> th.Tensor:
    """
    Check if the object is to the right of the player.
    """
    player_x = player[:, -2]
    shield_x = shield[:, -2]
    diff = shield_x - player_x
    result = bool_to_probs(diff < 0)
    return result

def on_left_bullet(player: th.Tensor, bullet: th.Tensor) -> th.Tensor:
    """
    Check if the object is to the left of the player.
    """
    player_x = player[:, -2]
    bullet_x = bullet[:, -2]
    diff = bullet_x - player_x
    result = bool_to_probs(diff < 0)
    return result

def on_right_bullet(player: th.Tensor, bullet: th.Tensor) -> th.Tensor:
    """
    Check if the object is to the right of the player.
    """
    player_x = player[:, -2]
    bullet_x = bullet[:, -2]
    diff = bullet_x - player_x
    result = bool_to_probs(diff > 0)
    return result


def sigmoid_smoothing(bool_tensor: th.Tensor, temperature: float = 5.0) -> th.Tensor:
    """
    Apply sigmoid smoothing to a boolean tensor, converting True/False into soft probabilities.

    :param bool_tensor: Boolean tensor indicating condition (True = overlap, False = no overlap).
    :param temperature: Controls softness of probability conversion (higher = more binary-like).
    :return: Soft probability tensor (0.0 to 1.0).
    """
    return th.sigmoid(temperature * (bool_tensor.float() - 0.5))  # Adaptive smoothing