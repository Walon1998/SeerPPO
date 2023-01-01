import torch
from torch import nn

from SeerPPO.distribution import MultiCategoricalDistribution


class SeerNetworkV2(nn.Module):
    def __init__(self):
        super(SeerNetworkV2, self).__init__()

        self.activation = nn.LeakyReLU(inplace=True)

        self.BALL_SIZE = 10
        self.PREV_ACTION_SIZE = 15
        self.BOOSTPADS_SIZE = 34
        self.PLAYER_SIZE = 23

        self.OBS_SIZE = 142

        self.ENCODER_INTERMEDIATE_SIZE = 256
        self.LSTM_INPUT_SIZE = 256

        self.LSTM_OUTPUT_SIZE = 256

        self.encoder = nn.Sequential(
            nn.Linear(self.OBS_SIZE, self.ENCODER_INTERMEDIATE_SIZE),
            self.activation,
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE, self.ENCODER_INTERMEDIATE_SIZE),
            self.activation,
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE, self.LSTM_INPUT_SIZE),
            self.activation,
        )

        self.LSTM = nn.LSTM(self.LSTM_INPUT_SIZE, self.LSTM_OUTPUT_SIZE, 1, batch_first=True)

        self.value_network = nn.Sequential(
            nn.Linear(self.LSTM_OUTPUT_SIZE, 256),
            self.activation,
            nn.Linear(256, 128),
            self.activation,
            nn.Linear(128, 1),
        )

        self.policy_network = nn.Sequential(
            nn.Linear(self.LSTM_OUTPUT_SIZE, 256),
            self.activation,
            nn.Linear(256, 128),
            self.activation,
            nn.Linear(128, 18),
        )

        self.distribution = MultiCategoricalDistribution([3, 3, 3, 3, 2, 2, 2])
        self.HUGE_NEG = None

    def encode(self, obs):

        assert obs.shape[-1] in [142]

        if obs.shape[-1] == 142:
            return self.encoder(obs)

    def forward(self, obs, lstm_states, episode_starts, deterministic):

        if self.HUGE_NEG is None:
            self.HUGE_NEG = torch.tensor(-1e8, dtype=torch.float32).to(obs.device)

        # Rollout
        pre_lstm = self.encode(obs)

        lstm_reset = (1.0 - episode_starts).view(1, -1, 1)

        lstm_states = (lstm_states[0] * lstm_reset, lstm_states[1] * lstm_reset)
        x, lstm_states = self.LSTM(pre_lstm.unsqueeze(1), lstm_states)

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
        pre_lstm = self.encode(obs)

        lstm_reset = (1.0 - episode_starts).view(1, -1, 1)

        lstm_states = (lstm_states[0] * lstm_reset, lstm_states[1] * lstm_reset)
        x, lstm_states = self.LSTM(pre_lstm.unsqueeze(1), lstm_states)
        x = x.squeeze(dim=1)

        value = self.value_network(x)
        return value

    def predict_actions(self, obs, lstm_states, episode_starts, deterministic):
        if self.HUGE_NEG is None:
            self.HUGE_NEG = torch.tensor(-1e8, dtype=torch.float32).to(obs.device)

            # Rollout
        pre_lstm = self.encode(obs)
        lstm_reset = (1.0 - episode_starts).view(1, -1, 1)

        lstm_states = (lstm_states[0] * lstm_reset, lstm_states[1] * lstm_reset)
        x, lstm_states = self.LSTM(pre_lstm.unsqueeze(1), lstm_states)

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

        x = self.encode(obs)

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

        before = self.BALL_SIZE + self.BOOSTPADS_SIZE + self.PREV_ACTION_SIZE

        has_boost = obs[..., before + 13] > 0.0
        on_ground = obs[..., before + 14]
        has_flip = obs[..., before + 15]

        in_air = torch.logical_not(on_ground)
        mask = torch.ones((size, 18), dtype=torch.bool, device=obs.device)

        # mask[:, 0:3] = 1.0  # Throttle, always possible
        # mask[:, 3:6] = 1.0  # Steer yaw, always possible
        # mask[:, 6:9] = 1.0  # pitch, not on ground but (flip resets, walldashes)
        # mask[:, 9:12] = 1.0  # roll, not on ground
        # mask[:, 12:14] = 1.0  # jump, has flip (turtle)
        # mask[:, 14:16] = 1.0  # boost, boost > 0
        # mask[:, 16:18] = 1.0  # Handbrake, at least one wheel ground (not doable)

        in_air = in_air.unsqueeze(1)
        mask[:, 6:12] = in_air  # pitch + roll

        has_flip = has_flip.unsqueeze(1)
        mask[:, 12:14] = has_flip  # has flip

        has_boost = has_boost.unsqueeze(1)
        mask[:, 14:16] = has_boost  # boost

        on_ground = on_ground.unsqueeze(1)
        mask[:, 16:18] = on_ground  # Handbrake

        return mask


if __name__ == '__main__':
    n = SeerNetworkV2()

    print(n)

    torch.save(n.state_dict(), "./0.pt")

    pytorch_total_params = sum(p.numel() for p in n.parameters())

    print(pytorch_total_params)
