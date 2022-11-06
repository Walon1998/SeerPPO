import math

import numpy as np
import torch
from torch import nn

from SeerPPO.distribution import MultiCategoricalDistribution


class SeerScalerV1(nn.Module):
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


class SeerNetworkV1(nn.Module):

    def __init__(self):
        super(SeerNetworkV1, self).__init__()

        self.activation = nn.LeakyReLU()

        self.scaler = SeerScalerV1()
        self.mlp_encoder = nn.Sequential(
            nn.Linear(159, 256),
            self.activation,
        )
        self.LSTM = nn.LSTM(256, 512, 1, batch_first=True)
        self.value_network = nn.Sequential(
            nn.Linear(512, 256),
            self.activation,
            nn.Linear(256, 128),
            self.activation,
            nn.Linear(128, 1),
        )
        self.policy_network = nn.Sequential(
            nn.Linear(512, 256),
            self.activation,
            nn.Linear(256, 256),
            self.activation,
            nn.Linear(256, 128),
            self.activation,
            nn.Linear(128, 22),
        )

        self.distribution = MultiCategoricalDistribution([3, 5, 5, 3, 2, 2, 2])
        self.HUGE_NEG = None

    def forward(self, obs, lstm_states, episode_starts, deterministic):

        if self.HUGE_NEG is None:
            self.HUGE_NEG = torch.tensor(-1e8, dtype=torch.float32).to(obs.device)

        # Rollout
        x = self.scaler(obs)
        x = self.mlp_encoder(x)

        lstm_reset = (1.0 - episode_starts).view(1, -1, 1)

        lstm_states = (lstm_states[0] * lstm_reset, lstm_states[1] * lstm_reset)
        x, lstm_states = self.LSTM(x.unsqueeze(1), lstm_states)

        x = x.squeeze(dim=1)

        value = self.value_network(x)
        policy_logits = self.policy_network(x)
        mask = self.create_mask(obs, policy_logits.shape[0])
        policy_logits = torch.where(mask, policy_logits, self.HUGE_NEG)
        self.distribution.proba_distribution(policy_logits)
        self.distribution.apply_mask(mask)

        actions = self.distribution.get_actions(deterministic=deterministic)
        log_prob = self.distribution.log_prob(actions)
        return actions, value, log_prob, lstm_states

    def predict_value(self, obs, lstm_states, episode_starts):
        # Rollout
        x = self.scaler(obs)
        x = self.mlp_encoder(x)

        lstm_reset = (1.0 - episode_starts).view(1, -1, 1)

        lstm_states = (lstm_states[0] * lstm_reset, lstm_states[1] * lstm_reset)
        x, lstm_states = self.LSTM(x.unsqueeze(1), lstm_states)
        x = x.squeeze(dim=1)

        value = self.value_network(x)
        return value

    def predict_actions(self, obs, lstm_states, episode_starts, deterministic):
        if self.HUGE_NEG is None:
            self.HUGE_NEG = torch.tensor(-1e8, dtype=torch.float32).to(obs.device)

            # Rollout
        x = self.scaler(obs)
        x = self.mlp_encoder(x)

        lstm_reset = (1.0 - episode_starts).view(1, -1, 1)

        lstm_states = (lstm_states[0] * lstm_reset, lstm_states[1] * lstm_reset)
        x, lstm_states = self.LSTM(x.unsqueeze(1), lstm_states)

        x = x.squeeze(dim=1)

        policy_logits = self.policy_network(x)
        mask = self.create_mask(obs, policy_logits.shape[0])
        policy_logits = torch.where(mask, policy_logits, self.HUGE_NEG)
        self.distribution.proba_distribution(policy_logits)
        self.distribution.apply_mask(mask)

        actions = self.distribution.get_actions(deterministic=deterministic)
        return actions, lstm_states

    def evaluate_actions(self, obs, actions, lstm_states, episode_starts, mask):

        if self.HUGE_NEG is None:
            self.HUGE_NEG = torch.tensor(-1e8, dtype=torch.float32).to(obs.device)

        lstm_states = (lstm_states[0].swapaxes(0, 1), lstm_states[1].swapaxes(0, 1))

        x = self.scaler(obs)
        x = self.mlp_encoder(x)

        lstm_output = []

        for i in range(16):
            features_i = x[:, i, :].unsqueeze(dim=1)
            episode_start_i = episode_starts[:, i]
            lstm_reset = (1.0 - episode_start_i).view(1, -1, 1)

            hidden, lstm_states = self.LSTM(features_i, (
                lstm_reset * lstm_states[0],
                lstm_reset * lstm_states[1],
            ))
            lstm_output += [hidden]

        x = torch.flatten(torch.cat(lstm_output, dim=1), start_dim=0, end_dim=1)
        actions = torch.flatten(actions, start_dim=0, end_dim=1)

        value = self.value_network(x)
        policy_logits = self.policy_network(x)
        policy_logits = torch.where(mask, policy_logits, self.HUGE_NEG)
        self.distribution.proba_distribution(policy_logits)
        log_prob = self.distribution.log_prob(actions)

        entropy = self.distribution.entropy()

        return value, log_prob, entropy

    def create_mask(self, obs, size):

        has_boost = obs[..., 13] > 0.0
        on_ground = obs[..., 14]
        has_flip = obs[..., 15]

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
