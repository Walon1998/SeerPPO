from typing import List, Tuple

import torch
from torch.distributions import Categorical


class MaskableCategorical(Categorical):
    def __init__(self, logits, validate_args, ZERO):
        super().__init__(None, logits, validate_args)
        self.mask = None
        self.zero = ZERO

    # TODO: Masking
    # def entropy(self) -> torch.Tensor:
    #     assert self.mask is not None
    #     p_log_p = self.logits * self.probs
    #     p_log_p = torch.where(self.mask, p_log_p, self.zero)
    #     return -p_log_p.sum(-1)


class MultiCategoricalDistribution:
    """
    MultiCategorical distribution for multi discrete actions.

    :param action_dims: List of sizes of discrete action spaces
    """

    def __init__(self, action_dims: List[int]):
        super(MultiCategoricalDistribution, self).__init__()
        self.distribution = None
        self.action_dims = action_dims
        self.zero = None

    def proba_distribution(self, action_logits: torch.Tensor) -> "MultiCategoricalDistribution":
        if self.zero is None:
            self.zero = torch.zeros(1, device=action_logits.device)
        self.distribution = [MaskableCategorical(logits=split, validate_args=False, ZERO=self.zero) for split in torch.split(action_logits, tuple(self.action_dims), dim=1)]
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        # Extract each discrete action and compute log prob for their respective distributions
        return torch.stack(
            [dist.log_prob(action) for dist, action in zip(self.distribution, torch.unbind(actions, dim=1))], dim=1
        ).sum(dim=1)

    def entropy(self) -> torch.Tensor:
        return torch.stack([dist.entropy() for dist in self.distribution], dim=1).sum(dim=1)

    def sample(self) -> torch.Tensor:
        return torch.stack([dist.sample() for dist in self.distribution], dim=1)

    def mode(self) -> torch.Tensor:
        return torch.stack([torch.argmax(dist.probs, dim=1) for dist in self.distribution], dim=1)

    def actions_from_params(self, action_logits: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()

    def apply_mask(self, mask):
        counter = 0
        for split in torch.split(mask, tuple(self.action_dims), dim=1):
            self.distribution[counter].mask = split
            counter += 1
