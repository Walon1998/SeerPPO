import torch
from torch import nn

from SeerPPO.distribution import MultiCategoricalDistribution
from SeerPPO.normalization import SeerScaler


class SeerNetwork(nn.Module):

    def __init__(self):
        super(SeerNetwork, self).__init__()

        self.activation = nn.LeakyReLU()

        self.scaler = SeerScaler()
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

    def apply_mask(self, obs, x):
        if self.HUGE_NEG is None:
            self.HUGE_NEG = torch.tensor(-1e8, dtype=torch.float32).to(x.device)

        has_boost = obs[:, 13] > 0.0
        on_ground = obs[:, 14]
        has_flip = obs[:, 15]

        in_air = torch.logical_not(on_ground)
        mask = torch.ones_like(x, dtype=torch.bool)

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

        x = torch.where(mask, x, self.HUGE_NEG)

        return x, mask

    def forward(self, obs, lstm_states, episode_starts, deterministic):
        # Rollout
        x = self.scaler(obs)
        x = self.mlp_encoder(x)

        lstm_reset = (1.0 - episode_starts).view(1, -1, 1)

        lstm_states = (lstm_states[0] * lstm_reset, lstm_states[1] * lstm_reset)
        x, lstm_states = self.LSTM(x.unsqueeze(1), lstm_states)

        x = x.squeeze()

        value = self.value_network(x)
        policy_logits = self.policy_network(x)
        # if use_masking:
        #     policy_logits, mask = self.apply_mask(obs, policy_logits)
        self.distribution.proba_distribution(policy_logits)
        # if use_masking:
        #     self.distribution.apply_mask(mask)

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
        x = x.squeeze()

        value = self.value_network(x)
        return value

    def evaluate_actions(self, obs, actions, lstm_states, episode_starts):

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
        self.distribution.proba_distribution(policy_logits)
        log_prob = self.distribution.log_prob(actions)

        entropy = self.distribution.entropy()

        return value, log_prob, entropy


if __name__ == '__main__':
    model = SeerNetwork()
    print(model)
