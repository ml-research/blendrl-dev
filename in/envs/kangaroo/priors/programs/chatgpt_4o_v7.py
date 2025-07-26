import torch

def left_of_ladder(x: torch.Tensor) -> torch.Tensor:
    # x is a tensor of size [batch_size, 2]
    dx = x[:, 0]  # horizontal offset: player_x - ladder_x
    dy = x[:, 1]  # vertical offset: player_y - ladder_y

    # Condition 1: player is left of the ladder
    left_condition = dx < 0

    # Condition 2: player is roughly aligned vertically with the ladder (on same platform)
    vertical_alignment = dy.abs() <= 8

    # Both conditions must be true
    return (left_condition & vertical_alignment).float()


def right_of_ladder(x: torch.Tensor) -> torch.Tensor:
    # x[:, 0] is horizontal offset (player_x - ladder_x)
    # x[:, 1] is vertical offset (player_y - ladder_y)

    x_offset = x[:, 0]
    y_offset = x[:, 1]

    # Player is right of the ladder if x_offset > a small threshold (e.g., > 4 pixels to the right)
    right_condition = x_offset > 4

    # Player is on the same platform level as ladder: small vertical offset
    vertical_align = torch.abs(y_offset) < 5

    return (right_condition & vertical_align).float()


def on_ladder(x: torch.Tensor) -> torch.Tensor:
    # x is a tensor of size [batch_size, 2], with each row [dx, dy]
    dx = x[:, 0]
    dy = x[:, 1]

    # Conditions:
    # Horizontally within ladder width (±4)
    # Vertically, ladder must be above and close enough: 0 <= dy <= 23
    on_ladder_mask = (dx.abs() <= 4) & (dy >= 0) & (dy <= 23)

    return on_ladder_mask.float()

def close_by_monkey(x: torch.Tensor) -> torch.Tensor:
    # x: [batch_size, 2] where each row is [dx, dy]
    dx = x[:, 0]
    dy = x[:, 1]

    # Vertical proximity (same platform level)
    close_y = dy.abs() <= 10

    # Horizontal punching range: monkey in front and within 12 pixels
    close_x = (dx >= 0) & (dx <= 12)

    # Both conditions must be true
    close = close_x & close_y

    return close.float()


def close_by_throwncoconut(x: torch.Tensor) -> torch.Tensor:
    # x: [batch_size, 2] where x[:, 0] is dx (horizontal), x[:, 1] is dy (vertical)
    dx = torch.abs(x[:, 0])
    dy = torch.abs(x[:, 1])

    # Define thresholds based on proximity where dodging is possible
    horizontal_threshold = 20.0  # pixels
    vertical_threshold = 10.0  # same platform height range

    close = (dx <= horizontal_threshold) & (dy <= vertical_threshold)
    return close.float()