import numpy as np
import torch
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat import GameTickPacket
from rlgym_compat import GameState


class SeerBot(BaseAgent):
    def __init__(self, name, team, index):
        super(SeerBot, self).__init__(name, team, index)

        self.obs_builder = None
        self.act_parser = None
        self.agent = None
        self.name = None

        self.tick_skip = 8
        self.game_state: GameState = None
        self.controls = None
        self.action = None
        self.update_action = True
        self.ticks = 0
        self.prev_time = 0
        self.compiled = False
        print('{name} Ready - Index:'.format(name=name), index)

    def initialize_agent(self):
        # Initialize the rlgym GameState object now that the game is active and the info is available
        self.game_state = GameState(self.get_field_info())
        self.ticks = self.tick_skip  # So we take an action the first tick
        self.prev_time = 0
        self.controls = SimpleControllerState()
        self.action = np.zeros(8)
        self.update_action = True

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        cur_time = packet.game_info.seconds_elapsed
        delta = cur_time - self.prev_time
        self.prev_time = cur_time

        ticks_elapsed = round(delta * 120)
        self.ticks += ticks_elapsed
        self.game_state.decode(packet, ticks_elapsed)

        if not packet.game_info.is_round_active and self.compiled:
            self.agent.reset_states()
            self.compiled = True
            return self.controls

        if self.update_action:
            self.update_action = False

            player = self.game_state.players[self.index]
            teammates = [p for p in self.game_state.players if p.team_num == self.team]
            opponents = [p for p in self.game_state.players if p.team_num != self.team]

            self.game_state.players = teammates + opponents

            self.obs_builder.pre_step(self.game_state)
            obs = self.obs_builder.build_obs(player, self.game_state, self.action)
            self.action = self.act_parser.parse_actions(self.agent.act(obs), self.game_state)[0]

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


class Agent:
    def __init__(self, filename, policy):
        self.filename = filename

        self.policy = policy
        self.policy.load_state_dict(torch.load(self.filename, map_location=torch.device('cpu')))
        self.policy.eval()
        torch.set_num_threads(1)
        self.lstm_states = None
        self.episode_starts = torch.zeros(1, dtype=torch.float32, requires_grad=False)
        self.reset_states()

    def reset_states(self):
        self.lstm_states = (torch.zeros(1, 1, self.policy.LSTM.hidden_size, requires_grad=False), torch.zeros(1, 1, self.policy.LSTM.hidden_size, requires_grad=False))

    def act(self, obs):
        with torch.no_grad():
            action, self.lstm_states = self.policy.predict_actions(torch.from_numpy(obs).unsqueeze(0), self.lstm_states, self.episode_starts, True)
            return action.numpy()
