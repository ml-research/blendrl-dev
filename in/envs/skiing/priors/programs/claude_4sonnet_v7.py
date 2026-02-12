import torch


def left_of_flag(x: torch.Tensor) -> torch.Tensor:
    # x is a tensor of size [batch_size, 2]
    # x[:, 0] is horizontal offset (player_x - flag_x)
    # x[:, 1] is vertical offset (player_y - flag_y)

    # Player is left of flag if horizontal offset is negative
    left_condition = x[:, 0] < 0

    # For vertical alignment, consider reasonable gate passage range
    # Flags are 14 pixels tall, player is 18 pixels tall
    # Allow some tolerance for vertical misalignment
    vertical_tolerance = 20  # pixels
    vertical_alignment = torch.abs(x[:, 1]) <= vertical_tolerance

    # Combine conditions: player must be left AND vertically aligned
    result = (left_condition & vertical_alignment).float()

    # For cases where player is left but not perfectly aligned,
    # provide a softer probability based on vertical distance
    soft_vertical = torch.exp(-torch.abs(x[:, 1]) / 15.0)  # Exponential decay
    soft_result = (left_condition.float() * soft_vertical)

    # Use hard condition for well-aligned cases, soft for others
    final_result = torch.where(vertical_alignment, result, soft_result)

    return final_result


def right_of_flag(x: torch.Tensor) -> torch.Tensor:
    # x is [batch_size, 2] where x[:, 0] is horizontal offset (player_x - flag_x)
    # and x[:, 1] is vertical offset (player_y - flag_y)

    horizontal_offset = x[:, 0]  # positive means player is right of flag
    vertical_offset = x[:, 1]  # positive means player is below flag

    # Player is right of flag if horizontal_offset > 0
    right_condition = horizontal_offset > 0

    # Consider vertical alignment - player should be reasonably close vertically
    # to be in position to reach the gate. Using a reasonable threshold based on
    # typical gate spacing and player movement capabilities
    vertical_threshold = 30.0  # pixels - reasonable distance for gate approach
    vertical_alignment = torch.abs(vertical_offset) < vertical_threshold

    # Combine conditions: player must be right of flag AND vertically aligned
    can_reach_gate = right_condition & vertical_alignment

    # Convert boolean to float probabilities
    probabilities = can_reach_gate.float()

    # Optional: Use sigmoid for smoother transitions near boundaries
    # This provides more nuanced probabilities near the decision boundaries
    horizontal_score = torch.sigmoid(horizontal_offset / 10.0)  # scale for smooth transition
    vertical_score = torch.sigmoid((vertical_threshold - torch.abs(vertical_offset)) / 10.0)

    # Combine scores multiplicatively
    final_probability = horizontal_score * vertical_score

    return final_probability


def next_flag(x: torch.Tensor) -> torch.Tensor:
    # x is [batch_size, 2] where each row is [x_offset, y_offset]
    # x_offset = player_x - flag_x
    # y_offset = player_y - flag_y

    x_offset = x[:, 0]  # horizontal offset
    y_offset = x[:, 1]  # vertical offset

    # Vertical component: player should be at or slightly above the flag
    # Penalize heavily if player is too far below the flag
    vertical_score = torch.sigmoid(-y_offset / 20.0 + 2.0)

    # Horizontal component: player should be reasonably close horizontally
    # Use absolute distance with sigmoid for smooth falloff
    horizontal_score = torch.sigmoid(-torch.abs(x_offset) / 30.0 + 2.0)

    # Combine both components - both need to be satisfied
    probability = vertical_score * horizontal_score

    return probability

