import math
import os
import time

import numpy as np
from numba import jit
from numpy import ndarray
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
import torch
from sklearn.preprocessing import OneHotEncoder

from SeerPPO.V2 import SeerNetworkV2
from rlbot.utils.structures.game_data_struct import GameTickPacket


@jit(nopython=True, fastmath=True)
def invert_ball_dataV2(array):
    assert len(array) == 10
    ball_data = array * np.array([-1, -1, 1, -1, -1, 1, -1, -1, 1, 1], dtype=np.float32)
    return ball_data


def encode_ballV2(packet, inverted):
    array = np.array([
        packet.game_ball.physics.location.x * (1.0 / 4096.0),
        packet.game_ball.physics.location.y * (1.0 / 5120.0),
        packet.game_ball.physics.location.z * (1.0 / 2048.0),

        packet.game_ball.physics.velocity.x * (1.0 / 6000.0),
        packet.game_ball.physics.velocity.y * (1.0 / 6000.0),
        packet.game_ball.physics.velocity.z * (1.0 / 6000.0),

        packet.game_ball.physics.angular_velocity.x * (1.0 / 6.0),
        packet.game_ball.physics.angular_velocity.y * (1.0 / 6.0),
        packet.game_ball.physics.angular_velocity.z * (1.0 / 6.0),
        np.linalg.norm([packet.game_ball.physics.velocity.x, packet.game_ball.physics.velocity.y, packet.game_ball.physics.velocity.z]) * (1.0 / 6000.0)
    ], dtype=np.float32)

    if inverted:
        array = invert_ball_dataV2(array)

    return array


encV2 = OneHotEncoder(sparse=False, drop='if_binary',
                      categories=[np.array([0., 1., 2.]), np.array([0., 1., 2., ]), np.array([0., 1., 2., ]), np.array([0., 1., 2.]), np.array([0., 1.]), np.array([0., 1.]),
                                  np.array([0., 1.])])


def get_action_encodingV2(action):
    assert action.shape[1] == 7

    result = encV2.fit_transform(action)

    assert result.shape[1] == 15

    return result


@jit(nopython=True, fastmath=True)
def invert_boost_dataV2(array):
    assert array.shape[0] == 34
    return array[::-1]


def encode_boostV2(packet, inverted):
    boost = np.empty(34, dtype=np.float32)
    for i in range(34):
        boost[i] = packet.game_boosts[i].is_active

    if inverted:
        boost = invert_boost_dataV2(boost)

    return boost


@jit(nopython=True, fastmath=True)
def invert_yawV2(yaw):
    tol = 1e-4
    assert -math.pi - tol <= yaw <= math.pi + tol
    yaw += math.pi  # yaw in [- pi, pi]
    if yaw > math.pi:
        yaw -= 2 * math.pi
    assert -math.pi - tol <= yaw <= math.pi + tol
    return yaw


@jit(nopython=True, fastmath=True)
def invert_player_dataV2(player_data: np.ndarray) -> np.ndarray:
    assert len(player_data) == 22
    player_data = player_data * np.array([-1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1], dtype=np.float32)
    player_data[4] = invert_yawV2(player_data[4])
    return player_data


def encode_playerV2(player, ball, has_flip, inverted):
    vel_norm = np.linalg.norm([player.physics.velocity.x,
                               player.physics.velocity.y,
                               player.physics.velocity.z])

    ball_diff_x = ball.physics.location.x - player.physics.location.x
    ball_diff_y = ball.physics.location.y - player.physics.location.y
    ball_diff_z = ball.physics.location.z - player.physics.location.z
    ball_diff_norm = np.linalg.norm([ball_diff_x, ball_diff_y, ball_diff_z])

    array = np.array([
        player.physics.location.x * (1.0 / 4096.0),
        player.physics.location.y * (1.0 / 5120.0),
        player.physics.location.z * (1.0 / 2048.0),
        player.physics.rotation.pitch * (1.0 / math.pi),
        player.physics.rotation.yaw,
        player.physics.rotation.roll * (1.0 / math.pi),
        player.physics.velocity.x * (1.0 / 2300.0),
        player.physics.velocity.y * (1.0 / 2300.0),
        player.physics.velocity.z * (1.0 / 2300.0),
        player.physics.angular_velocity.x * (1.0 / 5.5),
        player.physics.angular_velocity.y * (1.0 / 5.5),
        player.physics.angular_velocity.z * (1.0 / 5.5),
        player.is_demolished,
        player.boost * (1 / 100.0),
        player.has_wheel_contact,
        has_flip,
        vel_norm * (1.0 / 6000.0),
        vel_norm > 2200,
        ball_diff_x * (1.0 / (4096.0 * 2.0)),
        ball_diff_y * (1.0 / (5120.0 * 2.0)),
        ball_diff_z * (1.0 / 2048.0),
        ball_diff_norm * (1.0 / 13272.55),
    ], dtype=np.float32)

    assert array.shape[0] == 22

    if inverted:
        array = invert_player_dataV2(array)

    array[4] *= (1.0 / math.pi)  # Yaw scale after invert

    return array


def encode_all_playersV2(player_index, packet: GameTickPacket, flips, inverted):
    player_encoding = encode_playerV2(packet.game_cars[player_index], packet.game_ball, flips[packet.game_cars[player_index].name], inverted)

    same_team = []
    opponent_team = []

    for p in range(packet.num_cars):

        p = packet.game_cars[p]

        if p.name == packet.game_cars[player_index].name:
            continue

        if p.team == packet.game_cars[player_index].team:
            same_team.append(p)
        else:
            opponent_team.append(p)

    same_team.sort(key=lambda x: x.name)
    opponent_team.sort(key=lambda x: x.name)

    encodings = [player_encoding]
    for p in same_team + opponent_team:
        encodings.append(encode_playerV2(p, packet.game_ball, flips[p.name], inverted))

    return encodings


def encode_gamestate(packet, inverted):
    dif = packet.teams[0].score - packet.teams[1].score
    timer = packet.game_info.game_time_remaining
    overtime = packet.game_info.is_overtime

    if inverted:
        dif *= -1

    if overtime:
        timer = 0

    return [dif / 10, timer / 300, overtime]


class SeerV2Template(BaseAgent):
    def __init__(self, name, team, index, filename):
        super().__init__(name, team, index)

        self.filename = filename
        self.game_over = False
        self.tick_skip = 8
        self.ticks = 8  # # So we take an action the first tick
        self.prev_time = 0

        torch.set_num_threads(1)

        if self.team == 1:
            self.inverted = True
        else:
            self.inverted = False

        print("{} Loading...".format(self.name))

        self.policy = SeerNetworkV2()
        self.policy.load_state_dict(torch.load(self.filename, map_location=torch.device('cpu')))
        self.policy.eval()

        self.controls = SimpleControllerState()
        self.latest_wheel_contact = {}

        self.prev_action = None
        self.lstm_states = None
        self.episode_starts = torch.zeros(1, dtype=torch.float32, requires_grad=False)
        self.reset_states()

        self.compiled = False

        print("Ready: {}".format(self.name))

    def initialize_agent(self):
        pass

    def reset_states(self):
        self.lstm_states = (torch.zeros(1, 1, self.policy.LSTM.hidden_size, requires_grad=False), torch.zeros(1, 1, self.policy.LSTM.hidden_size, requires_grad=False))
        self.prev_action = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def get_flips(self, packet: GameTickPacket):

        if len(self.latest_wheel_contact) == 0:
            for i in range(packet.num_cars):
                self.latest_wheel_contact[packet.game_cars[i].name] = 0.0

        has_flip = {}

        for i in range(packet.num_cars):

            p = packet.game_cars[i]

            if p.has_wheel_contact:
                self.latest_wheel_contact[p.name] = packet.game_info.seconds_elapsed

            flip_timeout = (packet.game_info.seconds_elapsed - self.latest_wheel_contact[p.name]) > 1.5 and p.jumped

            flip = not p.double_jumped and not flip_timeout

            has_flip[p.name] = flip

        return has_flip

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:

        cur_time = packet.game_info.seconds_elapsed
        delta = cur_time - self.prev_time
        self.prev_time = cur_time

        ticks_elapsed = delta * 120
        self.ticks += ticks_elapsed

        if not packet.game_info.is_round_active and self.compiled:
            self.controls.throttle = 0.0
            self.controls.steer = 0.0
            self.controls.yaw = 0.0
            self.controls.pitch = 0.0
            self.controls.roll = 0.0
            self.controls.boost = False
            self.controls.handbrake = False
            self.controls.jump = False
            self.reset_states()
            return self.controls

        elif self.ticks >= self.tick_skip:
            self.ticks = 0
            self.update_controls(packet)

        return self.controls

    def build_obs(self, packet):
        flips = self.get_flips(packet)

        game_state = encode_gamestate(packet, self.inverted)
        ball = encode_ballV2(packet, self.inverted)
        prev_action_encoding = get_action_encodingV2(self.prev_action.reshape(1, -1)).reshape(-1)
        pads_encoding = encode_boostV2(packet, self.inverted)

        player_encodings = encode_all_playersV2(self.index, packet, flips, self.inverted)

        obs = np.concatenate([*player_encodings, ball, prev_action_encoding, *pads_encoding, game_state]).reshape(1, -1)

        obs = torch.tensor(obs, dtype=torch.float32)
        self.compiled = True

        return obs

    def update_controls(self, packet: GameTickPacket):

        with torch.no_grad():
            obs = self.build_obs(packet)
            action, self.lstm_states = self.policy.predict_actions(obs, self.lstm_states, self.episode_starts, True)
            action = action.numpy()[0]
            self.prev_action = action

        self.update_controller_from_action(action)

    def update_controller_from_action(self, actions: ndarray):

        steer_yaw = actions[1] - 1.0

        self.controls.throttle = actions[0] - 1.0
        self.controls.steer = steer_yaw
        self.controls.pitch = actions[2] - 1.0
        self.controls.yaw = steer_yaw
        self.controls.roll = actions[3] - 1.0

        self.controls.jump = actions[4]
        self.controls.boost = actions[5]
        self.controls.handbrake = actions[6]
