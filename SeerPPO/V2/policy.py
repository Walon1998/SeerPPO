import torch
from torch import nn

from SeerPPO.distribution import MultiCategoricalDistribution


class SeerNetworkV2(nn.Module):
    def __init__(self):
        super(SeerNetworkV2, self).__init__()

        self.activation = nn.LeakyReLU(inplace=True)

        self.OBS_SIZE = 142

        self.ENCODER_INTERMEDIATE_SIZE = 1024

        self.encoder = nn.Sequential(
            nn.Linear(self.OBS_SIZE, self.ENCODER_INTERMEDIATE_SIZE),
            self.activation,
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE, self.ENCODER_INTERMEDIATE_SIZE),
            self.activation,
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE, self.ENCODER_INTERMEDIATE_SIZE),
            self.activation,
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE, self.ENCODER_INTERMEDIATE_SIZE),
            self.activation,
        )

        self.value_network = nn.Sequential(
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE, self.ENCODER_INTERMEDIATE_SIZE // 2),
            self.activation,
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE // 2, self.ENCODER_INTERMEDIATE_SIZE // 2),
            self.activation,
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE // 2, self.ENCODER_INTERMEDIATE_SIZE // 2),
            self.activation,
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE // 2, 1),
        )

        self.policy_network = nn.Sequential(
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE, self.ENCODER_INTERMEDIATE_SIZE // 2),
            self.activation,
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE // 2, self.ENCODER_INTERMEDIATE_SIZE // 2),
            self.activation,
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE // 2, self.ENCODER_INTERMEDIATE_SIZE // 2),
            self.activation,
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE // 2, 18),
        )

        self.distribution = MultiCategoricalDistribution([3, 3, 3, 3, 2, 2, 2])
        self.HUGE_NEG = None

    def encode(self, obs):

        assert obs.shape[-1] in [142]

        if obs.shape[-1] == 142:
            return self.encoder(obs)

    def forward(self, obs, deterministic):

        if self.HUGE_NEG is None:
            self.HUGE_NEG = torch.tensor(-1e8, dtype=torch.float32).to(obs.device)

        # Rollout
        x = self.encode(obs)

        value = self.value_network(x)
        policy_logits = self.policy_network(x)
        mask = self.create_mask(obs)
        policy_logits = torch.where(mask, policy_logits, self.HUGE_NEG)
        self.distribution.proba_distribution(policy_logits)
        self.distribution.apply_mask(mask)

        actions = self.distribution.get_actions(deterministic=deterministic)
        log_prob = self.distribution.log_prob(actions)
        return actions, value, log_prob,

    def predict_value(self, obs):
        # Rollout
        x = self.encode(obs)
        value = self.value_network(x)
        return value

    def predict_actions(self, obs, deterministic):
        if self.HUGE_NEG is None:
            self.HUGE_NEG = torch.tensor(-1e8, dtype=torch.float32).to(obs.device)

            # Rollout
        x = self.encode(obs)

        policy_logits = self.policy_network(x)
        mask = self.create_mask(obs)
        policy_logits = torch.where(mask, policy_logits, self.HUGE_NEG)
        self.distribution.proba_distribution(policy_logits)
        self.distribution.apply_mask(mask)

        actions = self.distribution.get_actions(deterministic=deterministic)
        return actions

    def evaluate_actions(self, obs, actions, mask):

        if self.HUGE_NEG is None:
            self.HUGE_NEG = torch.tensor(-1e8, dtype=torch.float32).to(obs.device)

        x = self.encode(obs)

        value = self.value_network(x)
        policy_logits = self.policy_network(x)
        policy_logits = torch.where(mask, policy_logits, self.HUGE_NEG)
        self.distribution.proba_distribution(policy_logits)
        log_prob = self.distribution.log_prob(actions)

        entropy = self.distribution.entropy()

        return value, log_prob, entropy

    def create_mask(self, obs):

        has_boost = obs[:, 13] > 0.0
        on_ground = obs[:, 14]
        has_flip = obs[:, 15]

        in_air = torch.logical_not(on_ground)
        mask = torch.ones((obs.shape[0], 18), dtype=torch.bool, device=obs.device)

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
