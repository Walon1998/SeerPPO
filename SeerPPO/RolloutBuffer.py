from typing import NamedTuple, Tuple, Union

import numpy as np
import torch
from numba import int32, float32, jit
from numba.experimental import jitclass


class RolloutBufferSamples(NamedTuple):
    observations: Union[np.ndarray, torch.Tensor]
    actions: Union[np.ndarray, torch.Tensor]
    log_prob: Union[np.ndarray, torch.Tensor]
    advantages: Union[np.ndarray, torch.Tensor]
    returns: Union[np.ndarray, torch.Tensor]
    lstm_states: Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]
    episode_starts: Union[np.ndarray, torch.Tensor]
    reward_mean: float32
    ep_len_mean: float32
    r2: float32


spec = [
    ('buffer_size', int32),
    ('observation_space', int32),
    ('action_space', int32),
    ('n_envs', int32),
    ('lstm_unroll_length', int32),
    ('lstm_hidden_size', int32),
    ('gamma', float32),
    ('gae_lambda', float32),
    ('pos', int32),
    ('observations', float32[:, :, :]),
    ('actions', float32[:, :, :]),
    ('rewards', float32[:, :]),
    ('episode_starts', float32[:, :]),
    ('values', float32[:, :]),
    ('log_prob', float32[:, :]),
    ('cell_states', float32[:, :, :]),
    ('hidden_states', float32[:, :, :]),
    ('advantages', float32[:, :]),
    ('returns', float32[:, :]),

]


@jit(fastmath=True)
def r2_score(y_true, y_pred):
    y_true_mean = y_true.mean()
    e_i = y_true - y_pred
    SS_res = (e_i * e_i).sum()
    b_i = y_true - y_true_mean
    SS_tot = (b_i * b_i).sum()
    r_2 = 1.0 - (SS_res / SS_tot)
    return r_2


@jitclass(spec)
class RolloutBuffer:
    def __init__(self, buffer_size, observation_space, action_space, n_envs, lstm_hidden_size, lstm_unroll_length, gamma, gae_lambda):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_envs = n_envs
        self.lstm_unroll_length = lstm_unroll_length
        self.lstm_hidden_size = lstm_hidden_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.pos = 0

        assert self.buffer_size % self.lstm_unroll_length == 0

        self.observations = np.empty((self.buffer_size, self.n_envs, self.observation_space), dtype=np.float32)
        self.actions = np.empty((self.buffer_size, self.n_envs, self.action_space), dtype=np.float32)
        self.rewards = np.empty((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.empty((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.empty((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_prob = np.empty((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cell_states = np.empty((self.buffer_size, self.n_envs, self.lstm_hidden_size), dtype=np.float32)
        self.hidden_states = np.empty((self.buffer_size, self.n_envs, self.lstm_hidden_size), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def reset(self):
        self.pos = 0

    def add(self, obs, action, reward, episode_start, value, log_prob, cell_states, hidden_states):
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.episode_starts[self.pos] = episode_start
        self.values[self.pos] = value
        self.log_prob[self.pos] = log_prob
        self.cell_states[self.pos] = cell_states
        self.hidden_states[self.pos] = hidden_states

        self.pos += 1

    def is_full(self):
        return self.pos == self.buffer_size

    def compute_returns_and_advantage(self, last_values, dones):

        last_gae_lam = np.zeros(1)
        for step in range(self.buffer_size - 1, -1, -1):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def get_samples(self):

        stack_size = int(self.buffer_size / self.lstm_unroll_length)

        new_obs = self.observations.transpose((1, 0, 2)).copy().reshape(self.n_envs * stack_size, self.lstm_unroll_length, self.observation_space)
        new_action = self.actions.transpose((1, 0, 2)).copy().reshape(self.n_envs * stack_size, self.lstm_unroll_length, self.action_space)
        new_episode_starts = self.episode_starts.transpose((1, 0)).copy().reshape(self.n_envs * stack_size, self.lstm_unroll_length)
        new_log_prob = self.log_prob.transpose((1, 0)).copy().reshape(self.n_envs * stack_size, self.lstm_unroll_length)
        new_advantages = self.advantages.transpose((1, 0)).copy().reshape(self.n_envs * stack_size, self.lstm_unroll_length)
        new_returns = self.returns.transpose((1, 0)).copy().reshape(self.n_envs * stack_size, self.lstm_unroll_length)

        new_cell_states = self.cell_states.transpose((1, 0, 2)).copy().reshape(self.n_envs * stack_size, self.lstm_unroll_length, self.lstm_hidden_size)
        new_hidden_states = self.hidden_states.transpose((1, 0, 2)).copy().reshape(self.n_envs * stack_size, self.lstm_unroll_length, self.lstm_hidden_size)

        new_cell_states = np.expand_dims(new_cell_states[:, 0, :], axis=1)
        new_hidden_states = np.expand_dims(new_hidden_states[:, 0, :], axis=1)

        lstm_states = new_hidden_states, new_cell_states

        r2 = r2_score(self.values.ravel(), self.returns.ravel())
        episode_starts_sum = self.episode_starts.sum()
        reward_mean = self.rewards.sum() / episode_starts_sum
        ep_len_mean = (self.episode_starts.size / episode_starts_sum)

        return RolloutBufferSamples(
            observations=new_obs,
            actions=new_action,
            log_prob=new_log_prob,
            advantages=new_advantages,
            returns=new_returns,
            lstm_states=lstm_states,
            episode_starts=new_episode_starts,
            reward_mean=reward_mean,
            ep_len_mean=ep_len_mean,
            r2=r2,
        )
