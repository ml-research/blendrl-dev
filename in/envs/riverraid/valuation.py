import torch as th

from nsfr.utils.common import bool_to_probs


def on_river(player: th.Tensor, river: th.Tensor) -> th.Tensor:
    return bool_to_probs(player[..., 0] == 1) * river[..., 0]


def right_of_river(player: th.Tensor, river: th.Tensor) -> th.Tensor:
    return bool_to_probs(player[..., 1] > river[..., 1]) * river[..., 0]


def left_of_river(player: th.Tensor, river: th.Tensor) -> th.Tensor:
    return bool_to_probs(player[..., 1] < river[..., 1]) * river[..., 0]


def false_predicate(player: th.Tensor) -> th.Tensor:
    return bool_to_probs(th.tensor([False]))


def on_bridge_player(player: th.Tensor, bridge: th.Tensor) -> th.Tensor:
    return (
        bool_to_probs(
            (player[..., 1] > bridge[..., 1])
            & (player[..., 1] < bridge[..., 1] + bridge[..., 3])
        )
        * bridge[..., 0]
    )


def on_bridge_river(bridge: th.Tensor, river: th.Tensor) -> th.Tensor:
    return (
        bool_to_probs(
            (bridge[..., 1] > river[..., 1])
            & (bridge[..., 1] < river[..., 1] + river[..., 3])
        )
        * river[..., 0]
    )


def close_by_fuel(player: th.Tensor, fuel: th.Tensor) -> th.Tensor:
    return _close_by(player, fuel)


def close_by_enemy_ship(player: th.Tensor, enemy_ship: th.Tensor) -> th.Tensor:
    return _close_by(player, enemy_ship)


def close_by_helicopter(player: th.Tensor, helicopter: th.Tensor) -> th.Tensor:
    return _close_by(player, helicopter)


def close_by_bullet(player: th.Tensor, bullet: th.Tensor) -> th.Tensor:
    return _close_by(player, bullet)


def close_by_enemy_base(player: th.Tensor, enemy_base: th.Tensor) -> th.Tensor:
    return _close_by(player, enemy_base)


def _close_by(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    threshold = 48
    dist = (player[:, 1:3] - obj[:, 1:3]).pow(2).sum(1).sqrt()
    return bool_to_probs(dist < threshold) * obj[..., 0]


def nothing_around(image: th.Tensor) -> th.Tensor:
    return bool_to_probs(th.all(image == 0))


def same_level_river(player: th.Tensor, river: th.Tensor) -> th.Tensor:
    return bool_to_probs(abs(player[..., 2] - river[..., 2]) < 6) * river[..., 0]


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
