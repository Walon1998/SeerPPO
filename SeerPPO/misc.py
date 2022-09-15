from numba import jit
import numpy as np
from typing import Any, Tuple


@jit(nopython=True, fastmath=True)
def get_encoded_action(action: np.ndarray) -> np.ndarray:
    # throttle, steer, pitch, yaw, roll, jump, boost, handbrake

    action_encoding = np.zeros(19, dtype=np.float32)

    acc = 0
    throttle_index = action[0] + 1 + acc
    acc += 3
    steer_yaw_index = (action[1] + 1) * 2 + acc
    acc += 5
    pitch_index = (action[2] + 1) * 2 + acc
    acc += 5
    roll_index = action[4] + 1 + acc

    action_encoding[int(throttle_index)] = 1.0
    action_encoding[int(steer_yaw_index)] = 1.0
    action_encoding[int(pitch_index)] = 1.0
    action_encoding[int(roll_index)] = 1.0

    action_encoding[18] = action[7]
    action_encoding[17] = action[6]
    action_encoding[16] = action[5]

    # action_without_yaw = action[[0, 1, 2, 4, 5, 6, 7]]  # remove yaw

    # encoder = OneHotEncoder(sparse=False, drop='if_binary',
    #                         categories=[np.array([-1., 0., 1.]), np.array([-1., -0.5, 0., 0.5, 1.]), np.array([-1., -0.5, 0., 0.5, 1.]), np.array([-1., 0., 1.]), np.array([0., 1.]),
    #                                     np.array([0., 1.]),
    #                                     np.array([0., 1.])])

    return action_encoding


@jit(nopython=True, fastmath=True)
def get_distance(array_0: np.ndarray, array_1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # assert array_0.shape[0] == 3
    # assert array_1.shape[0] == 3

    diff = array_0 - array_1

    norm = np.linalg.norm(diff)

    return diff, np.array(norm, dtype=np.float32).reshape(1)


@jit(nopython=True, fastmath=True)
def get_speed(array: np.ndarray):
    speed = np.linalg.norm(array)

    is_super_sonic = speed >= 2200.0

    return np.array(speed, dtype=np.float32).reshape(1), np.array(is_super_sonic, dtype=np.float32).reshape(1)


@jit(nopython=True, fastmath=True)
def impute_features(player_car_state: np.ndarray, opponent_car_data: np.ndarray, pads, ball_data: np.ndarray, prev_action_enc: np.ndarray):
    # assert x_train.shape[0] == input_features_replay

    player_0 = player_car_state
    player_1 = opponent_car_data
    boost_pads_timer = pads
    ball = ball_data

    # assert player_0.shape[0] == 16
    # assert player_1.shape[0] == 16
    # assert ball.shape[0] == 9

    player_0_pos = player_0[0:3]
    # player_0_rotation = player_0[:, 3:6]
    player_0_velocity = player_0[6:9]
    # player_0_ang_velocity = player_0[:, 9:12]
    player_0_demo_timer = player_0[12]

    player_1_pos = player_1[0:3]
    # player_1_rotation = player_1[:, 3:6]
    player_1_velocity = player_1[6:9]
    # player_1_ang_velocity = player_1[:, 9:12]
    player_1_demo_timer = player_1[12]

    ball_pos = ball[0:3]
    ball_velocity = ball[3:6]
    # ball_1_ang_velocity = ball[:, 6:9]

    is_boost_active = boost_pads_timer == 0.0
    player_0_is_alive = player_0_demo_timer == 0.0
    player_1_is_alive = player_1_demo_timer == 0.0

    player_0_is_alive = np.array(player_0_is_alive, dtype=np.float32).reshape(1)
    player_1_is_alive = np.array(player_1_is_alive, dtype=np.float32).reshape(1)

    player_0_speed, player_0_super_sonic = get_speed(player_0_velocity)
    player_1_speed, player_1_super_sonic = get_speed(player_1_velocity)
    ball_speed, _ = get_speed(ball_velocity)

    player_opponent_pos_diff, player_opponent_pos_norm = get_distance(player_0_pos, player_1_pos)
    player_opponent_vel_diff, player_opponent_vel_norm = get_distance(player_0_velocity, player_1_velocity)

    player_ball_pos_diff, player_ball_pos_norm = get_distance(player_0_pos, ball_pos)
    player_ball_vel_diff, player_ball_vel_norm = get_distance(player_0_velocity, ball_velocity)

    opponent_ball_pos_diff, opponent_ball_pos_norm = get_distance(player_1_pos, ball_pos)
    opponent_ball_vel_diff, opponent_ball_vel_norm = get_distance(player_1_velocity, ball_velocity)

    result = np.concatenate((
        player_car_state, opponent_car_data, pads, ball_data,
        player_opponent_pos_diff, player_opponent_pos_norm,
        player_opponent_vel_diff, player_opponent_vel_norm,
        player_ball_pos_diff, player_ball_pos_norm,
        player_ball_vel_diff, player_ball_vel_norm,
        opponent_ball_pos_diff, opponent_ball_pos_norm,
        opponent_ball_vel_diff, opponent_ball_vel_norm,
        is_boost_active,
        player_0_is_alive, player_1_is_alive,
        player_0_speed, player_0_super_sonic,
        player_1_speed, player_1_super_sonic,
        ball_speed, prev_action_enc)
    )

    return result
