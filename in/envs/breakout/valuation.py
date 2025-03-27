import torch as th
from nsfr.utils.common import bool_to_probs

def player_left_of_ball(player: th.Tensor, ball: th.Tensor) -> th.Tensor:
    """Determines if the player is to the left of the ball.

    Args:
        player (th.Tensor): Tensor representing the player's current and old position
        ball (th.Tensor): Tensor representing the ball's current and old position

    Returns:
        th.Tensor: A probability tensor indicating if the player is to the left of the ball (True) or not (False)
    """
    player_x = player[..., 0]
    ball_x = ball[..., 0]
    dist = player_x - ball_x
    return sigmoid_smoothing(dist < 8, temperature=7.0)
    # return bool_to_probs(dist < 8)


def player_right_of_ball(player: th.Tensor, ball: th.Tensor) -> th.Tensor:
    """Determines if the player is to the right of the ball.

    Args:
        player (th.Tensor): Tensor representing the player's current and old position
        ball (th.Tensor): Tensor representing the ball's current and old position

    Returns:
        th.Tensor: A probability tensor indicating if the player is to the right of the ball (True) or not (False)
    """
    player_x = player[..., 0]
    ball_x = ball[..., 0]
    dist = player_x - ball_x
    return sigmoid_smoothing(dist > 8, temperature=7.0)
    # return bool_to_probs(dist > 8)


def ball_closeto_player(player: th.Tensor, ball: th.Tensor) -> th.Tensor:
    """Determines if the player is close to the ball.

    Args:
        player (th.Tensor): Tensor representing the player's current and old position
        ball (th.Tensor): Tensor representing the ball's current and old position

    Returns:
        th.Tensor: A probability tensor indicating if the player is close to the ball (True) or not (False)
    """
    player_x = player[..., 0]
    ball_x = ball[..., 0]
    dist= abs(player_x - ball_x)
    return sigmoid_smoothing(dist < 10, temperature=9.0)
    # return bool_to_probs(dist < 10)


def ball_goto_enemy(ball: th.Tensor) -> th.Tensor:
    """Determines is moving away from the player (up).

    Args:
        ball (th.Tensor): Tensor representing the ball's current and old position

    Returns:
        th.Tensor: A probability tensor indicating if ball is moving away from the player (up) (True) or not (False)
    """
    ball_y = ball[..., 1]
    ball_y2 = ball[..., 3]
    dir = ball_y - ball_y2
    return sigmoid_smoothing(dir < 0,temperature=8.0)
    # return bool_to_probs(dir < 0)


def ball_comingto_player(ball: th.Tensor) -> th.Tensor:
    """Determines is moving toward the player (down).

    Args:
        ball (th.Tensor): Tensor representing the ball's current and old position

    Returns:
        th.Tensor: A probability tensor indicating if ball is moving toward from the player (down) (True) or not (False)
    """
    ball_y = ball[..., 1]
    ball_y2 = ball[..., 3]
    dir = ball_y -ball_y2
    return sigmoid_smoothing(dir > 0, temperature=8.0)
    # return bool_to_probs(dir > 0)

def no_ball(ball: th.Tensor) -> th.Tensor:
    """Determines if the ball is in play

    Args:
        ball (th.Tensor): Tensor representing the ball's current and old position, -1 if no ball

    Returns:
        th.Tensor: A probability tensor indicating if the ball is in play (True) or not (False)
    """
    no_ball = ball[..., 0] == -1
    return sigmoid_smoothing(no_ball, temperature=9.0)
    # return bool_to_probs(no_ball)

def sigmoid_smoothing(bool_tensor: th.Tensor, temperature: float = 5.0) -> th.Tensor:
    """
    Apply sigmoid smoothing to a boolean tensor, converting True/False into soft probabilities.

    :param bool_tensor: Boolean tensor indicating condition (True = overlap, False = no overlap).
    :param temperature: Controls softness of probability conversion (higher = more binary-like).
    :return: Soft probability tensor (0.0 to 1.0).
    """
    return th.sigmoid(temperature * (bool_tensor.float() - 0.5))  # Adaptive smoothing