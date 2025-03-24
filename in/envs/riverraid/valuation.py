import torch as th
from nsfr.utils.common import bool_to_probs

def on_river(player: th.Tensor, grass: th.Tensor) -> th.Tensor:
    """Returns True if the player is within river boundaries (far from grass)."""
    player_x = player[..., 1]
    grass_x = grass[..., 1]
    grass_prob = grass[:, 0]
    return bool_to_probs(abs(player_x - grass_x) > 15) * grass_prob

def left_edge_river(player: th.Tensor, grass: th.Tensor) -> th.Tensor:
    """Returns True if the player is too close to the left-side grass."""
    player_x = player[..., 1]
    grass_x = grass[..., 1]
    grass_prob = grass[:, 0]
    return bool_to_probs(player_x < grass_x + 10) * grass_prob

def right_edge_river(player: th.Tensor, grass: th.Tensor) -> th.Tensor:
    """Returns True if the player is too close to the right-side grass."""
    player_x = player[..., 1]
    grass_x = grass[..., 1]
    grass_prob = grass[:, 0]
    return bool_to_probs(player_x > grass_x - 10) * grass_prob

def same_level_river(player: th.Tensor, grass: th.Tensor) -> th.Tensor:
    """Returns True if there is grass at the same height as the player (blocking path)."""
    player_y = player[..., 2]
    grass_y = grass[..., 2]
    grass_prob = grass[:, 0]
    return bool_to_probs(abs(player_y - grass_y) < 5) * grass_prob

def close_by_fuel(player: th.Tensor, fuel: th.Tensor) -> th.Tensor:
    return _close_by(player, fuel, threshold=25)

def close_by_enemy_ship(player: th.Tensor, enemy_ship: th.Tensor) -> th.Tensor:
    return _close_by(player, enemy_ship, threshold=40)

def close_by_helicopter(player: th.Tensor, helicopter: th.Tensor) -> th.Tensor:
    return _close_by(player, helicopter, threshold=35)

def close_by_enemy_base(player: th.Tensor, enemy_base: th.Tensor) -> th.Tensor:
    return _close_by(player, enemy_base, threshold=30)

def close_by_bridge(player: th.Tensor, bridge: th.Tensor) -> th.Tensor:
    return _close_by(player, bridge, threshold=40)

def nothing_around(objs: th.Tensor) -> th.Tensor:
    """Returns True if there are no enemy objects around."""
    enemies = th.cat([objs[:, 5:10], objs[:, 19:22]], dim=1)  # Enemy ships & helicopters
    near_enemies = th.sum(enemies[:, :, 0], dim=1) == 0
    return bool_to_probs(near_enemies)

def same_level_enemy_ship(player: th.Tensor, enemy: th.Tensor) -> th.Tensor:
    return _same_level(player, enemy)

def same_level_helicopter(player: th.Tensor, helicopter: th.Tensor) -> th.Tensor:
    return _same_level(player, helicopter)

def same_level_enemy_base(player: th.Tensor, enemy: th.Tensor) -> th.Tensor:
    return _same_level(player, enemy)

def same_level_bridge(player: th.Tensor, bridge: th.Tensor) -> th.Tensor:
    return _same_level(player, bridge)

def right_of_enemy_ship(player: th.Tensor, enemy: th.Tensor) -> th.Tensor:
    return _higher_than(player, enemy)

def left_of_enemy_ship(player: th.Tensor, enemy: th.Tensor) -> th.Tensor:
    return _lower_than(player, enemy)

def right_of_helicopter(player: th.Tensor, helicopter: th.Tensor) -> th.Tensor:
    return _higher_than(player, helicopter)

def left_of_helicopter(player: th.Tensor, helicopter: th.Tensor) -> th.Tensor:
    return _lower_than(player, helicopter)

def right_of_enemy_base(player: th.Tensor, enemy: th.Tensor) -> th.Tensor:
    return _higher_than(player, enemy)

def left_of_enemy_base(player: th.Tensor, enemy: th.Tensor) -> th.Tensor:
    return _lower_than(player, enemy)

def right_of_bridge(player: th.Tensor, bridge: th.Tensor) -> th.Tensor:
    return _higher_than(player, bridge)

def left_of_bridge(player: th.Tensor, bridge: th.Tensor) -> th.Tensor:
    return _lower_than(player, bridge)

def _close_by(player: th.Tensor, obj: th.Tensor, threshold=32) -> th.Tensor:
    """Returns True if player is within a certain distance of an object."""
    player_x, player_y = player[..., 1], player[..., 2]
    obj_x, obj_y = obj[..., 1], obj[..., 2]
    obj_prob = obj[:, 0]
    distance = ((player_x - obj_x) ** 2 + (player_y - obj_y) ** 2).sqrt()
    return bool_to_probs(distance < threshold) * obj_prob

def _higher_than(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """Returns True if the player is above the object."""
    return bool_to_probs(player[..., 2] < obj[..., 2] - 4) * obj[:, 0]

def _lower_than(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """Returns True if the player is below the object."""
    return bool_to_probs(player[..., 2] > obj[..., 2] + 4) * obj[:, 0]

def _same_level(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """Returns True if the player and the object are at the same height."""
    return bool_to_probs(abs(player[..., 2] - obj[..., 2]) < 5) * obj[:, 0]

def test_predicate_global(global_state: th.Tensor) -> th.Tensor:
    return bool_to_probs(global_state[..., 0, 2] < 100)

def test_predicate_object(agent: th.Tensor) -> th.Tensor:
    return bool_to_probs(agent[..., 2] < 100)

def true_predicate(agent: th.Tensor) -> th.Tensor:
    return bool_to_probs(th.tensor([True]))

def false_predicate(agent: th.Tensor) -> th.Tensor:
    return bool_to_probs(th.tensor([False]))
