import math

import numpy as np
import torch
from torch import nn


class Scaler(nn.Module):

    def __init__(self):
        super().__init__()
        self.scaler = None

    def forward(self, x):
        with torch.no_grad():

            if x.is_cuda:
                device_x = "cuda"
            else:
                device_x = "cpu"

            self.scaler = self.scaler.to(device_x)

            x = x * self.scaler
        return x


class SeerScaler(Scaler):
    def __init__(self):
        super().__init__()

        player_scaler = [
            1.0 / 4096.0,
            1.0 / 5120.0,
            1.0 / 2048.0,
            1.0 / math.pi,
            1.0 / math.pi,
            1.0 / math.pi,
            1.0 / 2300.0,
            1.0 / 2300.0,
            1.0 / 2300.0,
            1.0 / 5.5,
            1.0 / 5.5,
            1.0 / 5.5,
            1.0 / 3.0,
            1.0 / 100.0,
            1.0,
            1.0,
        ]

        ball_scaler = [
            1.0 / 4096.0,
            1.0 / 5120.0,
            1.0 / 2048.0,
            1.0 / 6000.0,
            1.0 / 6000.0,
            1.0 / 6000.0,
            1.0 / 6.0,
            1.0 / 6.0,
            1.0 / 6.0,
        ]

        boost_timer_scaler = [
            1.0 / 4.0,
            1.0 / 4.0,
            1.0 / 4.0,
            1.0 / 10.0,
            1.0 / 10.0,
            1.0 / 4.0,
            1.0 / 4.0,
            1.0 / 4.0,
            1.0 / 4.0,
            1.0 / 4.0,
            1.0 / 4.0,
            1.0 / 4.0,
            1.0 / 4.0,
            1.0 / 4.0,
            1.0 / 4.0,
            1.0 / 10.0,
            1.0 / 4.0,
            1.0 / 4.0,
            1.0 / 10.0,
            1.0 / 4.0,
            1.0 / 4.0,
            1.0 / 4.0,
            1.0 / 4.0,
            1.0 / 4.0,
            1.0 / 4.0,
            1.0 / 4.0,
            1.0 / 4.0,
            1.0 / 4.0,
            1.0 / 4.0,
            1.0 / 10.0,
            1.0 / 10.0,
            1.0 / 4.0,
            1.0 / 4.0,
            1.0 / 4.0,
        ]

        pos_diff = [
            1.0 / (4096.0 * 2.0),
            1.0 / (5120.0 * 2.0),
            1.0 / 2048.0,
            1.0 / 13272.55,
        ]
        vel_diff_player = [
            1.0 / (2300.0 * 2.0),
            1.0 / (2300.0 * 2.0),
            1.0 / (2300.0 * 2.0),
            1.0 / 2300.0,
        ]

        vel_diff_ball = [
            1.0 / (2300.0 + 6000.0),
            1.0 / (2300.0 + 6000.0),
            1.0 / (2300.0 + 6000.0),
            1.0 / 6000.0,
        ]

        boost_active = [
            1.0 for _ in range(34)
        ]
        player_alive = [1.0]

        player_speed = [
            1.0 / 2300,
            1.0,
        ]

        ball_speed = [
            1.0 / 6000.0
        ]

        prev_action = [1.0 for _ in range(19)]

        scaler = np.concatenate(
            [player_scaler, player_scaler, boost_timer_scaler, ball_scaler,
             pos_diff,
             vel_diff_player,
             pos_diff,
             vel_diff_ball,
             pos_diff,
             vel_diff_ball,
             boost_active,
             player_alive, player_alive,
             player_speed,
             player_speed,
             ball_speed, prev_action], dtype=np.float32
        )

        self.scaler = torch.tensor(scaler, dtype=torch.float32, requires_grad=False)

        assert torch.all(self.scaler <= 1.0)


class BallScalerv2(Scaler):

    def __init__(self):
        super(BallScalerv2, self).__init__()

        self.scaler = torch.tensor([
            1.0 / 4096.0,  # Pos_x
            1.0 / 5120.0,  # Pos_y
            1.0 / 2048.0,  # Pos_z
            1.0 / 6000.0,  # vel_x
            1.0 / 6000.0,  # vel_y
            1.0 / 6000.0,  # vel_z
            1.0 / 6.0,  # ang_vel_x
            1.0 / 6.0,  # ang_vel_y
            1.0 / 6.0,  # ang_vel_z
            1.0 / 6000.0,  # vel_norm
        ], dtype=torch.float32, requires_grad=False)


class BoostpadsScalerv2(Scaler):

    def __init__(self):
        super(BoostpadsScalerv2, self).__init__()

        boost_active = [
            1.0 for _ in range(34)
        ]

        # timers
        self.scaler = torch.tensor([
                                       1.0 / 4.0,
                                       1.0 / 4.0,
                                       1.0 / 4.0,
                                       1.0 / 10.0,
                                       1.0 / 10.0,
                                       1.0 / 4.0,
                                       1.0 / 4.0,
                                       1.0 / 4.0,
                                       1.0 / 4.0,
                                       1.0 / 4.0,
                                       1.0 / 4.0,
                                       1.0 / 4.0,
                                       1.0 / 4.0,
                                       1.0 / 4.0,
                                       1.0 / 4.0,
                                       1.0 / 10.0,
                                       1.0 / 4.0,
                                       1.0 / 4.0,
                                       1.0 / 10.0,
                                       1.0 / 4.0,
                                       1.0 / 4.0,
                                       1.0 / 4.0,
                                       1.0 / 4.0,
                                       1.0 / 4.0,
                                       1.0 / 4.0,
                                       1.0 / 4.0,
                                       1.0 / 4.0,
                                       1.0 / 4.0,
                                       1.0 / 4.0,
                                       1.0 / 10.0,
                                       1.0 / 10.0,
                                       1.0 / 4.0,
                                       1.0 / 4.0,
                                       1.0 / 4.0,
                                   ] + boost_active, dtype=torch.float32, requires_grad=False)


class PlayerEncoder(Scaler):

    def __init__(self):
        super(PlayerEncoder, self).__init__()
        self.scaler = torch.tensor([
            1.0 / 4096.0,  # pos_x
            1.0 / 5120.0,  # pos_y
            1.0 / 2048.0,  # pos_z
            1.0 / math.pi,  # rot_x
            1.0 / math.pi,  # rot_y
            1.0 / math.pi,  # rot_z
            1.0 / 2300.0,  # vel_x
            1.0 / 2300.0,  # vel_y
            1.0 / 2300.0,  # vel_z
            1.0 / 5.5,  # ang_vel_x
            1.0 / 5.5,  # ang_vel_y
            1.0 / 5.5,  # ang_vel_z
            1.0 / 3.0,  # demo timer
            1.0 / 100.0,  # boost
            1.0,  # wheel contact
            1.0,  # has flip
            1.0,  # alive
            1.0 / 6000.0,  # vel_norm
            1.0,  # supersonic
            1.0 / (4096.0 * 2.0),  # distance_ball_x
            1.0 / (5120.0 * 2.0),  # distance_ball_y
            1.0 / 2048.0,  # distance_ball_z
            1.0 / 13272.55,  # distance_ball_norm

        ], dtype=torch.float32, requires_grad=False)
