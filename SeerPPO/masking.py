import torch


def create_mask(obs, size):
    has_boost = obs[:, 13] > 0.0
    on_ground = obs[:, 14]
    has_flip = obs[:, 15]

    in_air = torch.logical_not(on_ground)
    mask = torch.ones((size, 22), dtype=torch.bool, device=obs.device)

    # mask[:, 0:3] = 1.0  # Throttle, always possible
    # mask[:, 3:8] = 1.0  # Steer yaw, always possible
    # mask[:, 8:13] = 1.0  # pitch, not on ground but (flip resets, walldashes)
    # mask[:, 13:16] = 1.0  # roll, not on ground
    # mask[:, 16:18] = 1.0  # jump, has flip (turtle)
    # mask[:, 18:20] = 1.0  # boost, boost > 0
    # mask[:, 20:22] = 1.0  # Handbrake, at least one wheel ground (not doable)

    in_air = in_air.unsqueeze(1)
    mask[:, 8:16] = in_air  # pitch + roll

    has_flip = has_flip.unsqueeze(1)
    mask[:, 16:18] = has_flip  # has flip

    has_boost = has_boost.unsqueeze(1)
    mask[:, 18:20] = has_boost  # boost

    on_ground = on_ground.unsqueeze(1)
    mask[:, 20:22] = on_ground  # Handbrake

    return mask
