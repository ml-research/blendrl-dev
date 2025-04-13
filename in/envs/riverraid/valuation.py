import torch as th
from nsfr.utils.common import bool_to_probs

LEFT_EDGE = 50
RIGHT_EDGE = 192 - LEFT_EDGE


def on_river(player: th.Tensor) -> th.Tensor:
    """Returns True if the player is within river boundaries."""
    player_x = player[..., 1]
    return bool_to_probs((player_x > LEFT_EDGE + 15) & (player_x < RIGHT_EDGE - 15))


def left_edge_river(player: th.Tensor) -> th.Tensor:
    """Returns True if the player is too close to the left-side boundary."""
    player_x = player[..., 1]
    return bool_to_probs(player_x < LEFT_EDGE + 10)


def right_edge_river(player: th.Tensor) -> th.Tensor:
    """Returns True if the player is too close to the right-side boundary."""
    player_x = player[..., 1]
    return bool_to_probs(player_x > RIGHT_EDGE - 10)


# Close by objects
def close_by_fuel(player: th.Tensor, fuel: th.Tensor) -> th.Tensor:
    return _close_by(player, fuel, threshold=25)


def close_by_enemy_ship(player: th.Tensor, enemy_ship: th.Tensor) -> th.Tensor:
    return _close_by(player, enemy_ship, threshold=40)


def close_by_helicopter(player: th.Tensor, helicopter: th.Tensor) -> th.Tensor:
    return _close_by(player, helicopter, threshold=35)


def close_by_enemy_jet(player: th.Tensor, enemy_jet: th.Tensor) -> th.Tensor:
    return _close_by(player, enemy_jet, threshold=30)


def close_by_bridge(player: th.Tensor, bridge: th.Tensor) -> th.Tensor:
    return _close_by(player, bridge, threshold=40)


def close_by_enemy(
    player: th.Tensor,
    enemy_ship: th.Tensor,
    helicopter: th.Tensor,
    enemy_jet: th.Tensor,
) -> th.Tensor:
    return th.maximum(
        _close_by(player, enemy_ship, threshold=40),
        th.maximum(
            _close_by(player, helicopter, threshold=35),
            _close_by(player, enemy_jet, threshold=30),
        ),
    )


# Far from objects
def far_from_fuel(player: th.Tensor, fuel: th.Tensor) -> th.Tensor:
    return _far_from(player, fuel, threshold=25)


def far_from_enemy_ship(player: th.Tensor, enemy_ship: th.Tensor) -> th.Tensor:
    return _far_from(player, enemy_ship, threshold=40)


def far_from_helicopter(player: th.Tensor, helicopter: th.Tensor) -> th.Tensor:
    return _far_from(player, helicopter, threshold=35)


def far_from_enemy_jet(player: th.Tensor, enemy_jet: th.Tensor) -> th.Tensor:
    return _far_from(player, enemy_jet, threshold=30)


def far_from_bridge(player: th.Tensor, bridge: th.Tensor) -> th.Tensor:
    return _far_from(player, bridge, threshold=40)


def far_from_enemy(
    player: th.Tensor,
    enemy_ship: th.Tensor,
    helicopter: th.Tensor,
    enemy_jet: th.Tensor,
) -> th.Tensor:
    return th.maximum(
        _far_from(player, enemy_ship, threshold=40),
        th.maximum(
            _far_from(player, helicopter, threshold=35),
            _far_from(player, enemy_jet, threshold=30),
        ),
    )


def nothing_around(objs: th.Tensor) -> th.Tensor:
    """Returns True if there are no enemy objects around."""
    enemies = th.cat(
        [objs[:, 5:10], objs[:, 19:22]], dim=1
    )  # Enemy ships & helicopters
    near_enemies = th.sum(enemies[:, :, 0], dim=1) == 0
    return bool_to_probs(near_enemies)


def same_level_enemy_ship(player: th.Tensor, enemy: th.Tensor) -> th.Tensor:
    return _same_level(player, enemy)


def same_level_helicopter(player: th.Tensor, helicopter: th.Tensor) -> th.Tensor:
    return _same_level(player, helicopter)


def same_level_enemy_jet(player: th.Tensor, enemy: th.Tensor) -> th.Tensor:
    return _same_level(player, enemy)


def same_level_bridge(player: th.Tensor, bridge: th.Tensor) -> th.Tensor:
    return _same_level(player, bridge)


def right_of_enemy_ship(player: th.Tensor, enemy: th.Tensor) -> th.Tensor:
    return _right_of(player, enemy)


def left_of_enemy_ship(player: th.Tensor, enemy: th.Tensor) -> th.Tensor:
    return _left_of(player, enemy)


def right_of_fuel(player: th.Tensor, fuel: th.Tensor) -> th.Tensor:
    return _right_of(player, fuel)


def left_of_fuel(player: th.Tensor, fuel: th.Tensor) -> th.Tensor:
    return _left_of(player, fuel)


def right_of_helicopter(player: th.Tensor, helicopter: th.Tensor) -> th.Tensor:
    return _right_of(player, helicopter)


def left_of_helicopter(player: th.Tensor, helicopter: th.Tensor) -> th.Tensor:
    return _left_of(player, helicopter)


def right_of_enemy_jet(player: th.Tensor, enemy: th.Tensor) -> th.Tensor:
    return _right_of(player, enemy)


def left_of_enemy_jet(player: th.Tensor, enemy: th.Tensor) -> th.Tensor:
    return _left_of(player, enemy)


def right_of_bridge(player: th.Tensor, bridge: th.Tensor) -> th.Tensor:
    return _right_of(player, bridge)


def left_of_bridge(player: th.Tensor, bridge: th.Tensor) -> th.Tensor:
    return _left_of(player, bridge)


def _close_by(player: th.Tensor, obj: th.Tensor, threshold=32) -> th.Tensor:
    """Returns True if player is within a certain distance of an object."""
    player_x, player_y = player[..., 1], player[..., 2]
    obj_x, obj_y = obj[..., 1], obj[..., 2]
    obj_prob = obj[:, 0]
    distance = ((player_x - obj_x) ** 2 + (player_y - obj_y) ** 2).sqrt()
    return bool_to_probs(distance < threshold) * obj_prob


def _far_from(player: th.Tensor, obj: th.Tensor, threshold=32) -> th.Tensor:
    """Returns True if player is within a certain distance of an object."""
    player_x, player_y = player[..., 1], player[..., 2]
    obj_x, obj_y = obj[..., 1], obj[..., 2]
    obj_prob = obj[:, 0]
    distance = ((player_x - obj_x) ** 2 + (player_y - obj_y) ** 2).sqrt()
    return bool_to_probs(distance > threshold) * obj_prob


def visible_bridge(obj: th.Tensor) -> th.Tensor:
    result = obj[..., 0] == 1
    return bool_to_probs(result)


def _right_of(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """Returns True if the player is above the object."""
    return bool_to_probs(player[..., 1] > obj[..., 1] - 4) * obj[:, 0]


def _left_of(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """Returns True if the player is below the object."""
    return bool_to_probs(player[..., 1] < obj[..., 1] + 4) * obj[:, 0]


def _same_level(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """Returns True if the player and the object are at the same height."""
    return bool_to_probs(abs(player[..., 1] - obj[..., 1]) < 5) * obj[:, 0]


def test_predicate_global(global_state: th.Tensor) -> th.Tensor:
    return bool_to_probs(global_state[..., 0, 2] < 100)


def test_predicate_object(agent: th.Tensor) -> th.Tensor:
    return bool_to_probs(agent[..., 2] < 100)


def true_predicate(agent: th.Tensor) -> th.Tensor:
    return bool_to_probs(th.tensor([True]))


def false_predicate(agent: th.Tensor) -> th.Tensor:
    return bool_to_probs(th.tensor([False]))
