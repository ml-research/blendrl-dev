import torch


def left_of_diver(x: torch.Tensor) -> torch.Tensor:
    """
    Determines if player is left of diver and can go right to get closer.

    Args:
        x: Tensor of size [batch_size, 2] containing offset vectors [x_offset, y_offset]
           where offset = player_position - diver_position

    Returns:
        Tensor of size [batch_size] with probabilities (0 to 1) indicating
        confidence that player is left of diver and can move right to get closer
    """
    # Extract x and y offsets
    x_offset = x[:, 0]  # player_x - diver_x
    y_offset = x[:, 1]  # player_y - diver_y

    # Player is left of diver when x_offset < 0
    # (because if player_x < diver_x, then player_x - diver_x < 0)
    left_of_diver_mask = x_offset < 0

    # For cases where player is left of diver, calculate probability
    # based on how reasonable it is to move right to collect the diver

    # Consider vertical distance - closer vertically means higher probability
    # Use absolute value since direction doesn't matter for distance
    vertical_distance = torch.abs(y_offset)

    # Consider horizontal distance - closer horizontally means higher probability
    # Use absolute value of x_offset for distance calculation
    horizontal_distance = torch.abs(x_offset)

    # Create a probability based on distances
    # Higher probability when both distances are smaller
    # Use exponential decay to handle distance penalty
    vertical_penalty = torch.exp(-vertical_distance / 20.0)  # Penalty decreases with vertical distance
    horizontal_bonus = torch.exp(-horizontal_distance / 30.0)  # Bonus for being closer horizontally

    # Combine factors for final probability
    probability = vertical_penalty * horizontal_bonus

    # Only assign non-zero probability where player is actually left of diver
    result = torch.where(left_of_diver_mask, probability, torch.zeros_like(probability))

    # Ensure probabilities are in [0, 1] range
    result = torch.clamp(result, 0.0, 1.0)

    return result


def right_of_diver(x: torch.Tensor) -> torch.Tensor:
    # x is a tensor of size [batch_size, 2]
    # x[:, 0] is the horizontal offset (player_x - diver_x)
    # x[:, 1] is the vertical offset (player_y - diver_y)

    # Player is right of diver if horizontal offset is positive
    # This means player_x > diver_x, so moving left reduces distance
    return (x[:, 0] > 0).float()


def higher_than_diver(x: torch.Tensor) -> torch.Tensor:
    """
    Determines if player is above a diver and can move down to get closer.

    Args:
        x: Tensor of size [batch_size, 2] where each row is [x_offset, y_offset]
           offset = [x_player - x_diver, y_player - y_diver]

    Returns:
        Tensor of size [batch_size] with probabilities (0 or 1) indicating
        whether player is above diver and should move down
    """
    # Extract x and y offsets
    x_offset = x[:, 0]  # x_player - x_diver
    y_offset = x[:, 1]  # y_player - y_diver

    # Player is above diver if y_offset < 0 (since origin is top-left)
    # This means y_player < y_diver, so player is higher up on screen
    above_condition = y_offset < 0

    # Additional conditions for meaningful "above" relationship:
    # 1. Not too far apart horizontally (within reasonable collection range)
    # 2. Sufficient vertical separation to warrant moving down

    # Horizontal proximity check - within about 2-3 player widths
    horizontal_proximity = torch.abs(x_offset) <= 48  # ~3 * player_width

    # Vertical separation check - meaningful gap but not too extreme
    vertical_separation = torch.abs(y_offset) >= 5  # At least some separation
    vertical_reasonable = torch.abs(y_offset) <= 100  # Not too far apart

    # Combine all conditions
    result = above_condition & horizontal_proximity & vertical_separation & vertical_reasonable

    # Convert boolean to float (0.0 or 1.0)
    return result.float()


def deeper_than_diver(x: torch.Tensor) -> torch.Tensor:
    """
    Determines if the player is below a diver and can go up to get closer.

    Args:
        x: Tensor of size [batch_size, 2] containing offset vectors [dx, dy]
           where dx = player_x - diver_x and dy = player_y - diver_y

    Returns:
        Tensor of size [batch_size] with probabilities (0 or 1) indicating
        whether the player is below the diver and can move up to get closer.
    """
    # Extract the vertical offset (dy = player_y - diver_y)
    dy = x[:, 1]

    # Extract the horizontal offset (dx = player_x - diver_x)
    dx = x[:, 0]

    # Player is below diver if dy > 0 (player has higher Y coordinate)
    # Moving up (decreasing Y) would reduce the distance to the diver
    is_below = dy > 0

    # Convert boolean to float (0.0 or 1.0)
    result = is_below.float()

    return result


def close_by_enemy(x: torch.Tensor) -> torch.Tensor:
    """
    Determines the probability that a player can shoot and kill an enemy (shark or submarine)
    based on the positional offset between player and enemy.

    Args:
        x: Tensor of shape [batch_size, 2] containing offset vectors [dx, dy]
           where dx = player_x - enemy_x and dy = player_y - enemy_y

    Returns:
        Tensor of shape [batch_size] with probabilities between 0 and 1
    """
    # Calculate Euclidean distance
    distance = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)

    # Define effective shooting range based on Seaquest mechanics
    # Close range: high probability (within ~30 pixels)
    # Medium range: decreasing probability (30-60 pixels)
    # Far range: very low probability (>60 pixels)

    close_range = 30.0
    medium_range = 60.0
    max_range = 80.0

    # Calculate base probability based on distance
    prob = torch.zeros_like(distance)

    # Very close enemies: high probability (0.9-1.0)
    close_mask = distance <= close_range
    prob[close_mask] = 1.0 - (distance[close_mask] / close_range) * 0.1

    # Medium distance enemies: decreasing probability (0.2-0.9)
    medium_mask = (distance > close_range) & (distance <= medium_range)
    medium_factor = (distance[medium_mask] - close_range) / (medium_range - close_range)
    prob[medium_mask] = 0.9 - medium_factor * 0.7

    # Far enemies: very low probability (0.0-0.2)
    far_mask = (distance > medium_range) & (distance <= max_range)
    far_factor = (distance[far_mask] - medium_range) / (max_range - medium_range)
    prob[far_mask] = 0.2 - far_factor * 0.2

    # Extremely far enemies: essentially 0 probability
    prob[distance > max_range] = 0.0

    # Apply directional bonus/penalty
    # Enemies more horizontally aligned are easier to hit
    abs_dx = torch.abs(x[:, 0])
    abs_dy = torch.abs(x[:, 1])

    # Bonus for horizontal alignment (when dy is small relative to dx)
    horizontal_bonus = torch.where(
        abs_dx > 0,
        torch.clamp(abs_dx / (abs_dx + abs_dy + 1e-8), 0.0, 1.0) * 0.1,
        torch.zeros_like(abs_dx)
    )

    # Apply bonus
    prob = torch.clamp(prob + horizontal_bonus, 0.0, 1.0)

    return prob


def close_by_missile(x: torch.Tensor) -> torch.Tensor:
    """
    Determines if player is close enough to a missile to dodge it.

    Args:
        x: Tensor of shape [batch_size, 2] containing offset vectors [dx, dy]
           where dx = player_x - missile_x, dy = player_y - missile_y

    Returns:
        Tensor of shape [batch_size] with probabilities (0-1) indicating
        whether the player is in a position to dodge the missile
    """
    # Calculate Euclidean distance from offset vectors
    distances = torch.norm(x, dim=1)

    # Define distance thresholds based on game mechanics
    # Player: 16x11, Missile: 6x4
    collision_radius = 12.0  # Approximate collision distance
    dodge_min_distance = 15.0  # Minimum distance needed to dodge
    dodge_max_distance = 45.0  # Maximum distance where dodging is relevant

    # Create probability function based on distance
    # - Too close (< dodge_min_distance): Very low probability of successful dodge
    # - Optimal range (dodge_min_distance to ~30): High probability
    # - Too far (> dodge_max_distance): Not relevant for dodging

    probabilities = torch.zeros_like(distances)

    # Very close - emergency dodge possible but difficult
    very_close_mask = distances < dodge_min_distance
    probabilities[very_close_mask] = 0.2

    # Optimal dodging range - player has time and space to maneuver
    optimal_mask = (distances >= dodge_min_distance) & (distances <= 30.0)
    probabilities[optimal_mask] = 1.0

    # Medium range - still relevant but less urgent
    medium_mask = (distances > 30.0) & (distances <= dodge_max_distance)
    # Linear decay from 1.0 to 0.1 over this range
    medium_distances = distances[medium_mask]
    probabilities[medium_mask] = 1.0 - 0.9 * ((medium_distances - 30.0) / 15.0)

    # Far away - not immediate threat
    far_mask = distances > dodge_max_distance
    probabilities[far_mask] = 0.0

    # Special case: if missile is directly on player (distance ~0)
    # This should be treated as collision imminent
    collision_mask = distances < collision_radius
    probabilities[collision_mask] = 0.05

    return probabilities