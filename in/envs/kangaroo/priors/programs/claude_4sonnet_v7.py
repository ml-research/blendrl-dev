import torch


def left_of_ladder(x: torch.Tensor) -> torch.Tensor:
    """
    Determines if player is left of ladder and can reach it by moving right.

    Args:
        x: Tensor of shape [batch_size, 2] containing offset vectors [dx, dy]
           where dx = player_x - ladder_x, dy = player_y - ladder_y

    Returns:
        Tensor of shape [batch_size] with probabilities (0 to 1)
    """
    # Extract x and y offsets
    dx = x[:, 0]  # horizontal offset (player_x - ladder_x)
    dy = x[:, 1]  # vertical offset (player_y - ladder_y)

    # Player is left of ladder if dx < 0 (player_x < ladder_x)
    is_left = dx < 0

    # Check if player is within reasonable horizontal distance to reach ladder
    # Maximum reasonable distance: about 1/4 of screen width (40 pixels)
    reasonable_distance = torch.abs(dx) <= 40

    # Check if player and ladder are vertically aligned
    # Allow some tolerance based on object heights
    # Player height: 24, Ladder height: 35
    # Allow vertical misalignment up to half the sum of heights
    vertical_tolerance = (24 + 35) // 4  # About 15 pixels tolerance
    vertically_aligned = torch.abs(dy) <= vertical_tolerance

    # Combine all conditions
    # Use soft transitions for smoother probabilities

    # Distance-based probability (closer = higher probability)
    distance_prob = torch.clamp(1.0 - torch.abs(dx) / 40.0, 0.0, 1.0)

    # Vertical alignment probability
    vertical_prob = torch.clamp(1.0 - torch.abs(dy) / vertical_tolerance, 0.0, 1.0)

    # Final probability: must be left AND reasonably close AND vertically aligned
    result = is_left.float() * distance_prob * vertical_prob

    return result


def right_of_ladder(x: torch.Tensor) -> torch.Tensor:
    """
    Determines if player is right of ladder and can go left to reach it.

    Args:
        x: Tensor of size [batch_size, 2] where each row is [x_offset, y_offset]
           x_offset = player_x - ladder_x
           y_offset = player_y - ladder_y

    Returns:
        Tensor of size [batch_size] with probabilities (0 to 1)
    """
    # Extract x and y offsets
    x_offset = x[:, 0]  # player_x - ladder_x
    y_offset = x[:, 1]  # player_y - ladder_y

    # Player dimensions: 8x24, Ladder dimensions: 8x35
    player_half_width = 4.0
    ladder_half_width = 4.0
    player_half_height = 12.0
    ladder_half_height = 17.5

    # Condition 1: Player must be to the right of the ladder
    # For the player to be "right of" the ladder, the right edge of the player
    # should be to the right of the left edge of the ladder
    # This means: player_x + player_half_width > ladder_x - ladder_half_width
    # Which translates to: x_offset > -(player_half_width + ladder_half_width)
    right_of_ladder_condition = x_offset > -(player_half_width + ladder_half_width)

    # Condition 2: Player shouldn't be too far right (within reasonable reach)
    # Maximum reasonable distance to walk left to reach ladder
    max_horizontal_distance = 40.0  # Reasonable walking distance in pixels
    not_too_far_right = x_offset < max_horizontal_distance

    # Condition 3: Vertical alignment - player should be roughly at same level
    # Allow for some vertical tolerance since platforms have height
    vertical_tolerance = 20.0  # Allow some vertical misalignment
    vertically_aligned = torch.abs(y_offset) <= vertical_tolerance

    # Combine all conditions
    # All conditions must be true for the player to be in position
    valid_position = right_of_ladder_condition & not_too_far_right & vertically_aligned

    # For a more nuanced approach, we can also create a soft probability
    # based on how well the conditions are met

    # Soft horizontal positioning (sigmoid-like function)
    horizontal_prob = torch.sigmoid(x_offset * 0.2) * torch.sigmoid((max_horizontal_distance - x_offset) * 0.1)

    # Soft vertical alignment (gaussian-like function)
    vertical_prob = torch.exp(-torch.abs(y_offset) / (vertical_tolerance / 2))

    # Combine probabilities
    soft_prob = horizontal_prob * vertical_prob

    # Use hard conditions for cases where we're clearly out of range
    # Otherwise use soft probability
    result = torch.where(valid_position, soft_prob, torch.zeros_like(soft_prob))

    return result


def on_ladder(x: torch.Tensor) -> torch.Tensor:
    """
    Determines if a player is on a ladder and able to climb up.

    Args:
        x: Tensor of size [batch_size, 2] containing offset vectors [dx, dy]
           where dx = player_x - ladder_x, dy = player_y - ladder_y

    Returns:
        Tensor of size [batch_size] with probabilities (0 to 1) indicating
        if the player is on the ladder and can climb up
    """
    # Extract horizontal and vertical offsets
    dx = x[:, 0]  # horizontal offset
    dy = x[:, 1]  # vertical offset

    # Player and ladder dimensions
    player_width = 8
    ladder_width = 8
    ladder_height = 35
    player_height = 24

    # Horizontal alignment check
    # Both player and ladder have width 8, so centers should be very close
    horizontal_tolerance = 4.0  # Half the width
    horizontal_aligned = torch.abs(dx) <= horizontal_tolerance

    # Vertical positioning check
    # Player should be at bottom portion of ladder to climb up
    # dy > 0 means player is below ladder center
    # We want player to be positioned where they can grab the ladder
    # This happens when player is slightly below ladder center but not too far

    # The player should be positioned such that they can reach the ladder
    # Considering ladder height is 35 and player height is 24
    # Player should be within reasonable range below ladder center
    min_vertical_offset = -8.0  # Player can be slightly above ladder center
    max_vertical_offset = 20.0  # Player shouldn't be too far below

    vertical_positioned = (dy >= min_vertical_offset) & (dy <= max_vertical_offset)

    # Combine conditions
    on_ladder_condition = horizontal_aligned & vertical_positioned

    # Convert boolean to float and add some smoothing for more realistic probabilities
    base_probability = on_ladder_condition.float()

    # Add distance-based smoothing for more nuanced probabilities
    # Closer to ideal position = higher probability
    horizontal_score = torch.clamp(1.0 - torch.abs(dx) / horizontal_tolerance, 0.0, 1.0)

    # Optimal vertical position is slightly below ladder center (dy around 5-10)
    optimal_dy = 8.0
    vertical_score = torch.clamp(1.0 - torch.abs(dy - optimal_dy) / 15.0, 0.0, 1.0)

    # Combine scores
    final_probability = base_probability * horizontal_score * vertical_score

    return final_probability


def close_by_monkey(x: torch.Tensor) -> torch.Tensor:
    """
    Determines if a player in Atari Kangaroo is close enough to a monkey to throw punches.

    Args:
        x: Tensor of size [batch_size, 2] containing offset vectors [dx, dy]
           where dx = player_x - monkey_x, dy = player_y - monkey_y

    Returns:
        Tensor of size [batch_size] with probabilities (0 or 1) indicating
        whether the player can punch the monkey
    """
    # Extract horizontal and vertical offsets
    dx = x[:, 0]  # player_x - monkey_x
    dy = x[:, 1]  # player_y - monkey_y

    # Define punching range thresholds
    # Horizontal range: considering player width (8) + monkey width (6) + punch reach
    horizontal_threshold = 14.0  # pixels

    # Vertical range: players should be at roughly the same height
    # Allow for some vertical tolerance due to platform thickness and object heights
    vertical_threshold = 10.0  # pixels

    # Calculate absolute distances
    horizontal_distance = torch.abs(dx)
    vertical_distance = torch.abs(dy)

    # Player can punch if both horizontal and vertical distances are within thresholds
    can_punch = (horizontal_distance <= horizontal_threshold) & (vertical_distance <= vertical_threshold)

    # Return as float tensor (0.0 or 1.0)
    return can_punch.float()


def close_by_throwncoconut(x: torch.Tensor) -> torch.Tensor:
    """
    Determines if a player in Kangaroo is close by a thrown coconut and able to dodge it.

    Args:
        x: Tensor of shape [batch_size, 2] containing offset vectors [dx, dy]
           where dx = player_x - coconut_x and dy = player_y - coconut_y

    Returns:
        Tensor of shape [batch_size] with probabilities (0 or 1) indicating
        whether the player can dodge the coconut
    """
    batch_size = x.shape[0]

    # Extract x and y offsets
    dx = x[:, 0]  # horizontal offset (player_x - coconut_x)
    dy = x[:, 1]  # vertical offset (player_y - coconut_y)

    # Calculate absolute distances
    abs_dx = torch.abs(dx)
    abs_dy = torch.abs(dy)

    # Define dodge parameters based on game mechanics
    # Player dimensions: 8x24, Coconut dimensions: 2x3

    # Horizontal dodge range: Player should be within reasonable movement distance
    # Consider player can move left/right, so allow for some horizontal separation
    max_horizontal_dodge = 32.0  # About 4 times player width
    min_horizontal_dodge = 2.0  # Minimum distance to avoid immediate collision

    # Vertical dodge range: Consider jumping capability and falling coconuts
    # Player height is 24, so vertical range should account for jump height
    max_vertical_dodge = 40.0  # Reasonable vertical range for dodging
    min_vertical_dodge = 1.0  # Minimum vertical separation

    # Conditions for successful dodging:
    # 1. Not too close horizontally (avoid immediate collision)
    # 2. Not too far horizontally (within movement range)
    # 3. Not too close vertically (avoid immediate collision)
    # 4. Not too far vertically (within jump/fall range)

    horizontal_dodge_ok = (abs_dx >= min_horizontal_dodge) & (abs_dx <= max_horizontal_dodge)
    vertical_dodge_ok = (abs_dy >= min_vertical_dodge) & (abs_dy <= max_vertical_dodge)

    # Additional consideration: if coconut is directly above/below player (small dx),
    # dodging is easier as player can move horizontally
    direct_vertical_threat = abs_dx <= 8.0  # Within player width

    # If coconut is directly above/below, allow slightly larger vertical range
    extended_vertical_ok = direct_vertical_threat & (abs_dy <= 50.0)

    # Combine conditions
    can_dodge = (horizontal_dodge_ok & vertical_dodge_ok) | extended_vertical_ok

    # Convert boolean to float (0.0 or 1.0)
    return can_dodge.float()