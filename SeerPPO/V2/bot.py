import math
import os
import time

import numpy as np
from numba import jit
from numpy import ndarray
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
import torch
from sklearn.preprocessing import OneHotEncoder

from SeerPPO.V2 import SeerNetworkV2, SeerGameCondition, SeerObsV2, SeerActionV2
from rlbot.utils.structures.game_data_struct import GameTickPacket

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

import numpy as np

from rlgym_compat import GameState


class Agent:
    def __init__(self, filename):
        self.filename = filename
        print("{} Loading...".format(self.filename))

        torch.set_num_threads(1)
        self.policy = SeerNetworkV2()
        self.policy.load_state_dict(torch.load(self.filename, map_location=torch.device('cpu')))
        self.policy.eval()

        self.episode_starts = torch.zeros(1, dtype=torch.float32, requires_grad=False)
        self.lstm_states = None
        self.reset()

        print("Ready: {}".format(self.filename))

    def reset(self):
        self.lstm_states = (torch.zeros(1, 1, self.policy.LSTM.hidden_size, requires_grad=False), torch.zeros(1, 1, self.policy.LSTM.hidden_size, requires_grad=False))

    def act(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state)
            action, self.lstm_states = self.policy.predict_actions(state, self.lstm_states, self.episode_starts, True)
        return action.numpy()


class SeerV2Template(BaseAgent):
    def __init__(self, name, team, index, filename):
        super().__init__(name, team, index)

        self.condition = SeerGameCondition()
        self.obs_builder = SeerObsV2(1, self.condition)
        self.act_parser = SeerActionV2()
        self.agent = Agent(filename)
        self.tick_skip = 8

        self.game_state: GameState = None
        self.controls = None
        self.action = None
        self.update_action = True
        self.ticks = 0
        self.prev_time = 0
        print('RLGymExampleBot Ready - Index:', index)

    def initialize_agent(self):
        # Initialize the rlgym GameState object now that the game is active and the info is available
        self.game_state = GameState(self.get_field_info())
        self.ticks = self.tick_skip  # So we take an action the first tick
        self.prev_time = 0
        self.controls = SimpleControllerState()
        self.action = np.zeros(8)
        self.update_action = True
        self.agent.reset()

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        cur_time = packet.game_info.seconds_elapsed
        delta = cur_time - self.prev_time
        self.prev_time = cur_time

        ticks_elapsed = round(delta * 120)
        self.ticks += ticks_elapsed
        self.game_state.decode(packet, ticks_elapsed)

        if not packet.game_info.is_round_active:
            self.ticks = self.tick_skip  # So we take an action the first tick
            self.prev_time = 0
            self.action = np.zeros(8)
            self.update_action = True
            self.agent.reset()

        if self.update_action:
            self.update_action = False

            player = self.game_state.players[self.index]
            teammates = [p for p in self.game_state.players if p.team_num == self.team and p != player]
            opponents = [p for p in self.game_state.players if p.team_num != self.team]

            self.game_state.players = [player] + teammates + opponents

            self.condition.overtime = packet.game_info.is_overtime
            self.condition.timer = packet.game_info.game_time_remaining
            self.condition.score = packet.teams[0].score - packet.teams[1].score
            if self.team == 1:
                self.condition.score *= -1

            self.obs_builder.pre_step(self.game_state)
            obs = self.obs_builder.build_obs(player, self.game_state, self.action)
            self.action = self.act_parser.parse_actions(self.agent.act(obs.reshape(1, -1)), self.game_state)[0]  # Dim is (N, 8)

        if self.ticks >= self.tick_skip - 1:
            self.update_controls(self.action)

        if self.ticks >= self.tick_skip:
            self.ticks = 0
            self.update_action = True

        return self.controls

    def update_controls(self, action):
        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = 0 if action[5] > 0 else action[3]
        self.controls.roll = action[4]
        self.controls.jump = action[5] > 0
        self.controls.boost = action[6] > 0
        self.controls.handbrake = action[7] > 0
