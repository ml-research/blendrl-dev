import torch as th
from nsfr.utils.common import bool_to_probs

""" in ocatari/ram/spaceinvaders.py :
    MAX_NB_OBJECTS = {'Player': 1, 'Shield': 3,
                  'Bullet': 3, 'Satellite': 1, 'Alien': 36}
    
"""

def is_player_under_bullet(player: th.Tensor, bullet: th.Tensor) -> th.Tensor:
    """Determines if the player is positioned under a bullet.

    Args:
        player (th.Tensor): Tensor containing player position data, where the second-to-last
            element represents the x-coordinate.
        bullet (th.Tensor): Tensor containing bullet position data, where the second-to-last
            element represents the x-coordinate.

    Returns:
        th.Tensor: Probability tensor (0 to 1) indicating whether the player is under the bullet,
            where values closer to 1 indicate higher probability of alignment.
    """
    player_x = player[:, -2]
    bullet_x = bullet[:, -2]
    diff = abs(bullet_x - player_x)
    result = bool_to_probs(diff < 4)
    return result

def is_player_under_alien(player: th.Tensor, alien: th.Tensor) -> th.Tensor:
    """Determines if the player is positioned under an alien.

    Args:
        player (th.Tensor): Tensor containing player position data, where the second-to-last
            element represents the x-coordinate.
        alien (th.Tensor): Tensor containing alien position data, where the second-to-last
            element represents the x-coordinate.

    Returns:
        th.Tensor: Probability tensor (0 to 1) indicating whether the player is under the alien,
            where values closer to 1 indicate higher probability of alignment.
    """
    player_x = player[:, -2]
    alien_x = alien[:, -2]
    diff = abs(alien_x - player_x)
    result = bool_to_probs(diff < 4)
    return result

def is_player_under_shield(player: th.Tensor, shield: th.Tensor) -> th.Tensor:
    """Determines if the player is positioned under a shield.

    Args:
        player (th.Tensor): Tensor containing player position data, where the second-to-last
            element represents the x-coordinate.
        shield (th.Tensor): Tensor containing shield position data, where the second-to-last
            element represents the x-coordinate.

    Returns:
        th.Tensor: Probability tensor (0 to 1) indicating whether the player is under the shield,
            where values closer to 1 indicate higher probability of alignment.
    """
    player_x = player[:, -2]
    shield_x = shield[:, -2]
    diff = abs(shield_x - player_x)
    result = bool_to_probs(diff < 10)
    return result

def is_player_not_under_shield(player: th.Tensor, shield: th.Tensor) -> th.Tensor:
    """Determines if the player is NOT positioned under a shield.

    Args:
        player (th.Tensor): Tensor containing player position data, where the second-to-last
            element represents the x-coordinate.
        shield (th.Tensor): Tensor containing shield position data, where the second-to-last
            element represents the x-coordinate.

    Returns:
        th.Tensor: Probability tensor (0 to 1) indicating whether the player is NOT under the shield,
            where values closer to 1 indicate higher probability of no alignment.
    """
    player_x = player[:, -2]
    shield_x = shield[:, -2]
    diff = abs(shield_x - player_x)
    result = 1 - bool_to_probs(diff < 10)
    return result

def is_player_under_satellite(player: th.Tensor, satellite: th.Tensor) -> th.Tensor:
    """Determines if the player is positioned under a satellite.

    Args:
        player (th.Tensor): Tensor containing player position data, where the second-to-last
            element represents the x-coordinate.
        satellite (th.Tensor): Tensor containing satellite position data, where the second-to-last
            element represents the x-coordinate.

    Returns:
        th.Tensor: Probability tensor (0 to 1) indicating whether the player is under the satellite,
            where values closer to 1 indicate higher probability of alignment.
    """
    player_x = player[:, -2]
    satellite_x = satellite[:, -2]
    diff = abs(satellite_x - player_x)
    result = bool_to_probs(diff < 8)
    return result

def close_to_bullet(player: th.Tensor, bullet: th.Tensor) -> th.Tensor:
    """Determines if the player is in close proximity to a bullet.

    Args:
        player (th.Tensor): Tensor containing player position data, with last two elements
            representing (x, y) coordinates.
        bullet (th.Tensor): Tensor containing bullet position data, with last two elements
            representing (x, y) coordinates.

    Returns:
        th.Tensor: Probability tensor (0 to 1) indicating whether the player is close to the bullet,
            where values closer to 1 indicate higher probability of proximity.
    """
    player_xy = player[:, -2:]
    bullet_xy = bullet[:, -2:]

    dis_x = abs(player_xy[:, 0] - bullet_xy[:, 0])
    dis_y = abs(player_xy[:, 1] - bullet_xy[:, 1]) 

    result = bool_to_probs((dis_x < 6) & (dis_y < 20))

    return result

def on_left_shield(player: th.Tensor, shield: th.Tensor) -> th.Tensor:
    """Determines if the shield is to the left of the player.

    Args:
        player (th.Tensor): Tensor containing player position data, where the second-to-last
            element represents the x-coordinate.
        shield (th.Tensor): Tensor containing shield position data, where the second-to-last
            element represents the x-coordinate.

    Returns:
        th.Tensor: Probability tensor (0 to 1) indicating whether the shield is to the left
            of the player, where values closer to 1 indicate higher probability.
    """
    player_x = player[:, -2]
    shield_x = shield[:, -2]
    diff = shield_x - player_x
    result = bool_to_probs(diff > 0)
    return result

def on_right_shield(player: th.Tensor, shield: th.Tensor) -> th.Tensor:
    """Determines if the shield is to the right of the player.

    Args:
        player (th.Tensor): Tensor containing player position data, where the second-to-last
            element represents the x-coordinate.
        shield (th.Tensor): Tensor containing shield position data, where the second-to-last
            element represents the x-coordinate.

    Returns:
        th.Tensor: Probability tensor (0 to 1) indicating whether the shield is to the right
            of the player, where values closer to 1 indicate higher probability.
    """
    player_x = player[:, -2]
    shield_x = shield[:, -2]
    diff = shield_x - player_x
    result = bool_to_probs(diff < 0)
    return result

def on_left_bullet(player: th.Tensor, bullet: th.Tensor) -> th.Tensor:
    """Determines if the bullet is to the left of the player.

    Args:
        player (th.Tensor): Tensor containing player position data, where the second-to-last
            element represents the x-coordinate.
        bullet (th.Tensor): Tensor containing bullet position data, where the second-to-last
            element represents the x-coordinate.

    Returns:
        th.Tensor: Probability tensor (0 to 1) indicating whether the bullet is to the left
            of the player, where values closer to 1 indicate higher probability.
    """
    player_x = player[:, -2]
    bullet_x = bullet[:, -2]
    diff = bullet_x - player_x
    result = bool_to_probs(diff < 0)
    return result

def on_right_bullet(player: th.Tensor, bullet: th.Tensor) -> th.Tensor:
    """Determines if the bullet is to the right of the player.

    Args:
        player (th.Tensor): Tensor containing player position data, where the second-to-last
            element represents the x-coordinate.
        bullet (th.Tensor): Tensor containing bullet position data, where the second-to-last
            element represents the x-coordinate.

    Returns:
        th.Tensor: Probability tensor (0 to 1) indicating whether the bullet is to the right
            of the player, where values closer to 1 indicate higher probability.
    """
    player_x = player[:, -2]
    bullet_x = bullet[:, -2]
    diff = bullet_x - player_x
    result = bool_to_probs(diff > 0)
    return result
