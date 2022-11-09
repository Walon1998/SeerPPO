import torch
from torch import nn

from SeerPPO.distribution import MultiCategoricalDistribution


class SeerNetworkV2(nn.Module):
    def __init__(self):
        super(SeerNetworkV2, self).__init__()

        self.activation = nn.LeakyReLU()

        self.BALL_SIZE = 10
        self.PREV_ACTION_SIZE = 15
        self.BOOSTPADS_SIZE = 68
        self.PLAYER_SIZE = 23

        self.BALL_ENC_SIZE = 32
        self.PREV_ACTION_ENC_SIZE = 32
        self.PADS_ENC_SIZE = 64
        self.PLAYER_ENC_SIZE = 64
        self.OTHER_ENC_SIZE = 256

        self.ball_encoder = nn.Sequential(
            nn.Linear(self.BALL_SIZE, self.BALL_ENC_SIZE),
            self.activation,
        )

        self.previous_action_encoder = nn.Sequential(
            nn.Linear(self.PREV_ACTION_SIZE, self.PREV_ACTION_ENC_SIZE),
            self.activation,
        )

        self.boost_pads_encoder = nn.Sequential(
            nn.Linear(self.BOOSTPADS_SIZE, self.PADS_ENC_SIZE),
            self.activation,
        )

        self.player_encoder = nn.Sequential(
            nn.Linear(self.PLAYER_SIZE, self.PLAYER_ENC_SIZE),
            self.activation,
        )

        self.ones_encoder = nn.Sequential(
            nn.Linear(self.PLAYER_ENC_SIZE, self.OTHER_ENC_SIZE),
            self.activation,
            nn.Linear(self.OTHER_ENC_SIZE, self.OTHER_ENC_SIZE),
            self.activation,
        )

        self.twos_encoder = nn.Sequential(
            nn.Linear(self.PLAYER_ENC_SIZE * 3, self.OTHER_ENC_SIZE),
            self.activation,
            nn.Linear(self.OTHER_ENC_SIZE, self.OTHER_ENC_SIZE),
            self.activation,
        )

        self.threes_encoder = nn.Sequential(
            nn.Linear(self.PLAYER_ENC_SIZE * 5, self.OTHER_ENC_SIZE),
            self.activation,
            nn.Linear(self.OTHER_ENC_SIZE, self.OTHER_ENC_SIZE),
            self.activation,
        )

        self.COMBINED_OUT = self.BALL_ENC_SIZE + self.PREV_ACTION_ENC_SIZE + self.PADS_ENC_SIZE + self.PLAYER_ENC_SIZE + self.OTHER_ENC_SIZE

        self.pre_LSTM = nn.Sequential(
            nn.Linear(self.COMBINED_OUT, self.COMBINED_OUT),
            self.activation,
            nn.Linear(self.COMBINED_OUT, self.COMBINED_OUT),
            self.activation,
        )

        self.LSTM_OUTPUT_SIZE = 384
        self.LSTM = nn.LSTM(self.COMBINED_OUT, self.LSTM_OUTPUT_SIZE, 1, batch_first=True)

        self.value_network = nn.Sequential(
            nn.Linear(self.LSTM_OUTPUT_SIZE, 128),
            self.activation,
            nn.Linear(128, 128),
            self.activation,
            nn.Linear(128, 1),
        )

        self.policy_network = nn.Sequential(
            nn.Linear(self.LSTM_OUTPUT_SIZE, 128),
            self.activation,
            nn.Linear(128, 128),
            self.activation,
            nn.Linear(128, 18),
        )

        self.distribution = MultiCategoricalDistribution([3, 3, 3, 3, 2, 2, 2])
        self.HUGE_NEG = None

    def encode(self, obs):

        assert obs.shape[-1] in [139, 185, 231]

        index = 0
        ball_encoding = self.ball_encoder(obs[..., index:index + self.BALL_SIZE])
        index += self.BALL_SIZE
        prev_action_encoding = self.previous_action_encoder(obs[..., index:index + self.PREV_ACTION_SIZE])
        index += self.PREV_ACTION_SIZE
        boost_pads_encoding = self.boost_pads_encoder(obs[..., index: index + self.BOOSTPADS_SIZE])
        index += self.BOOSTPADS_SIZE

        players = []

        for i in range(index, obs.shape[-1], self.PLAYER_SIZE):
            players.append(self.player_encoder(obs[..., i:i + self.PLAYER_SIZE]))

        player_0_encoding = players[0]

        if len(players) == 2:
            other_encoding = self.ones_encoder(players[1])
        elif len(players) == 4:
            other_encoding = self.twos_encoder(torch.cat([players[1], players[2], players[3]], dim=-1))
        elif len(players) == 6:
            other_encoding = self.threes_encoder(torch.cat([players[1], players[2], players[3], players[4], players[5]], dim=-1))

        else:
            raise NotImplementedError

        res = torch.cat([ball_encoding, prev_action_encoding, boost_pads_encoding, player_0_encoding, other_encoding], dim=-1)

        res = self.pre_LSTM(res)

        return res

    def forward(self, obs, lstm_states, episode_starts, deterministic):

        if self.HUGE_NEG is None:
            self.HUGE_NEG = torch.tensor(-1e8, dtype=torch.float32).to(obs.device)

        # Rollout
        x = self.encode(obs)

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
        x = self.encode(obs)

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
        x = self.encode(obs)
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
