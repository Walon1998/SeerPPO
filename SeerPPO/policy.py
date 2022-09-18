import torch
from torch import nn

from SeerPPO.distribution import MultiCategoricalDistribution
from SeerPPO.masking import create_mask
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
            nn.Linear(159, 256),
            self.activation,
            nn.Linear(256, 256),
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
        value = self.value_network(x)
        x = self.mlp_encoder(x)

        lstm_reset = (1.0 - episode_starts).view(1, -1, 1)

        lstm_states = (lstm_states[0] * lstm_reset, lstm_states[1] * lstm_reset)
        x, lstm_states = self.LSTM(x.unsqueeze(1), lstm_states)

        x = x.squeeze(dim=1)

        policy_logits = self.policy_network(x)
        mask = create_mask(obs, policy_logits.shape[0])
        policy_logits = torch.where(mask, policy_logits, self.HUGE_NEG)
        self.distribution.proba_distribution(policy_logits)
        self.distribution.apply_mask(mask)

        actions = self.distribution.get_actions(deterministic=deterministic)
        log_prob = self.distribution.log_prob(actions)
        return actions, value, log_prob, lstm_states

    def predict_value(self, obs):
        # Rollout
        x = self.scaler(obs)
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
        mask = create_mask(obs, policy_logits.shape[0])
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
        value = self.value_network(x)
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

        policy_logits = self.policy_network(x)
        policy_logits = torch.where(mask, policy_logits, self.HUGE_NEG)
        self.distribution.proba_distribution(policy_logits)
        log_prob = self.distribution.log_prob(actions)

        entropy = self.distribution.entropy()

        return value, log_prob, entropy


if __name__ == '__main__':
    model = SeerNetwork()
    print(model)
