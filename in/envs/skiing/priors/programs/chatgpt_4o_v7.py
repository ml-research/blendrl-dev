import torch


def left_of_flag(x: torch.Tensor) -> torch.Tensor:
    # x is a tensor of size [batch_size, 2], where each row is [dx, dy]
    # Extract horizontal offset (dx)
    dx = x[:, 0]

    # Define the condition: player is left of the flag (dx < 0)
    # Add a buffer zone: player must be within -15 units horizontally to reasonably reach the gate
    left_condition = (dx < 0) & (dx > -15)

    # Output probabilities: 1 if condition is met, 0 otherwise
    return left_condition.float()


def right_of_flag(x: torch.Tensor) -> torch.Tensor:
    # x is a tensor of size [batch_size, 2]
    # Horizontal offset (x[:, 0]) determines "right of the flag"
    # Threshold derived from player and flag dimensions
    threshold = 2.5  # (player width - flag width) / 2
    # Compute the condition: is the player right of the flag?
    result = (x[:, 0] > threshold).float()  # Convert boolean to float (0 or 1)
    return result


def next_flag(x: torch.Tensor) -> torch.Tensor:
    # Calculate Euclidean distance (sqrt(dx^2 + dy^2))
    distance = torch.sqrt(torch.sum(x ** 2, dim=1))  # Shape: [batch_size]

    # Horizontal alignment: |dx| <= 10
    horizontal_alignment = torch.abs(x[:, 0]) <= 10  # Shape: [batch_size]

    # Proximity threshold: distance <= 20
    proximity = distance <= 20  # Shape: [batch_size]

    # Combine conditions: both proximity and alignment must be true
    result = proximity & horizontal_alignment  # Logical AND; Shape: [batch_size]

    # Convert boolean tensor to float (0.0 or 1.0)
    return result.float()