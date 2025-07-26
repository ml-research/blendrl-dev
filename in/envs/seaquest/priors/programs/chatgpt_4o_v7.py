import torch

def left_of_diver(x: torch.Tensor) -> torch.Tensor:
    # x is a tensor of size [batch_size, 2]
    # Return 1 if player is left of diver, else 0
    return (x[:, 0] < 0).float()

def right_of_diver(x: torch.Tensor) -> torch.Tensor:
    # Return 1 if player is right of diver (x-offset > 0), else 0
    return (x[:, 0] > 0).float()

def higher_than_diver(x: torch.Tensor) -> torch.Tensor:
    # x is a tensor of size [batch_size, 2]
    # dy is the vertical offset (player_y - diver_y)
    dy = x[:, 1]
    return (dy < 0).float()

def deeper_than_diver(x: torch.Tensor) -> torch.Tensor:
    # x is a tensor of size [batch_size, 2]
    # Check if vertical offset (y) > 0, which means player is below diver
    return (x[:, 1] > 0).float()

def close_by_enemy(x: torch.Tensor) -> torch.Tensor:
    # x: [batch_size, 2], where x[:, 0] is dx (player_x - enemy_x)
    #                        and x[:, 1] is dy (player_y - enemy_y)
    dx = x[:, 0]
    dy = x[:, 1]

    # Enemy must be in front of the player (to the right), within a reasonable firing range
    horizontal_ok = (dx >= -40) & (dx <= 4)
    # Enemy must be vertically aligned with the missile path
    vertical_ok = dy.abs() <= 8

    # Only when both conditions are satisfied can we shoot to kill
    return (horizontal_ok & vertical_ok).float()


def close_by_missile(x: torch.Tensor) -> torch.Tensor:
    # x is [batch_size, 2] representing [dx, dy] = player_pos - missile_pos
    dx = x[:, 0]
    dy = x[:, 1]

    # Missile is close if:
    # - vertically within 10 pixels
    # - horizontally within 40 pixels in front or behind
    vertical_close = dy.abs() <= 10
    horizontal_close = dx.abs() <= 40

    # The player can potentially dodge only if both conditions are met
    close = vertical_close & horizontal_close

    return close.float()