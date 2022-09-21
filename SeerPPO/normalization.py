import math

import numpy as np
import torch
from torch import nn


class SeerScaler(nn.Module):
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

    def forward(self, x):
        with torch.no_grad():

            if x.is_cuda:
                device_x = "cuda"
            else:
                device_x = "cpu"

            self.scaler = self.scaler.to(device_x)

            x = x * self.scaler
        return x
