# -*- coding: utf-8 -*-
# @Time    : 2025/8/7 10:00
# @Author  : Gemini
# @File    : multi_uav_env.py

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt

from .formation_layer import FormationLayer
from ..utils.visualization import animate_step


class MultiUAVEnv(gym.Env):
    metadata = {'render_modes': ['human', '3d_plot', 'live']}

    def __init__(self, env_config, formation_config, reward_config):
        super().__init__()
        self.env_config = env_config
        self.reward_config = reward_config

        self.num_uavs = env_config['num_uavs']
        self.num_obstacles = env_config['num_obstacles']
        self.env_size = env_config['env_size']
        self.dt = env_config['dt']
        self.max_steps = env_config['max_steps']
        self.goal_threshold = env_config['goal_threshold']
        self.collision_threshold = env_config['collision_threshold']
        self.narrow_channel_scenario = env_config['narrow_channel_scenario']

        self.observation_dim = 21
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_uavs, self.observation_dim),
                                            dtype=np.float32)
        self.action_dim = 4
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_uavs, self.action_dim), dtype=np.float32)
        self.max_velocity = 5.0
        self.max_yaw_rate = np.pi / 2

        self.uav_states, self.obstacle_positions, self.goal_position, self.current_step = None, None, None, 0
        self.formation_layer = FormationLayer(self.num_uavs, formation_config)
        self.trajectory, self.render_fig, self.render_ax = [], None, None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step, self.trajectory = 0, [[] for _ in range(self.num_uavs)]
        self.goal_position = self.np_random.uniform(
            low=[self.env_size[0] * 0.8, self.env_size[1] * 0.8, self.env_size[2] * 0.5],
            high=[self.env_size[0] * 0.9, self.env_size[1] * 0.9, self.env_size[2] * 0.8]
        )
        self._reset_obstacles()
        self._reset_uavs()
        obs, info = self._get_obs(), self._get_info()
        for i in range(self.num_uavs): self.trajectory[i].append(self.uav_states[i, :3])
        return obs, info

    def step(self, actions):
        self.current_step += 1
        scaled_actions = self._scale_actions(actions)
        self._update_uav_states(scaled_actions)
        for i in range(self.num_uavs): self.trajectory[i].append(self.uav_states[i, :3])

        collision_info = self._check_collisions()
        done, goal_reached = self._check_done(collision_info)
        reward = self._compute_reward(collision_info)

        obs, info = self._get_obs(), self._get_info()
        info['is_success'], info['collisions'] = goal_reached, collision_info
        terminated, truncated = done, self.current_step >= self.max_steps
        return obs, reward, terminated, truncated, info

    def render(self, mode='3d_plot', title="Final Trajectory", save_path=None):
        from ..utils.visualization import plot_trajectories_3d
        plot_trajectories_3d(self.trajectory, self.goal_position, self.obstacle_positions, self.env_size, title,
                             save_path)

    def render_live(self, frame_delay=0.01):
        if self.render_fig is None or not plt.fignum_exists(self.render_fig.number):
            plt.ion()
            self.render_fig = plt.figure(figsize=(10, 8))
            self.render_ax = self.render_fig.add_subplot(111, projection='3d')
        animate_step(self.render_fig, self.render_ax, self.trajectory, self.goal_position, self.obstacle_positions,
                     self.env_size, f"Live Simulation (Step: {self.current_step})")
        plt.pause(frame_delay)

    def close(self):
        if self.render_fig is not None:
            plt.ioff()
            plt.close(self.render_fig)
            self.render_fig, self.render_ax = None, None
        print("环境已关闭。")

    # --- [修正版] 奖励函数 V2.1: 增加了奖励裁剪 ---
    def _compute_reward(self, collision_info):
        R_CFG = self.reward_config
        rewards = np.zeros(self.num_uavs)

        leader_pos, leader_yaw = self.uav_states[0, :3], self.uav_states[0, 8]
        formation_goals = self.formation_layer.get_formation_goal_positions(leader_pos, leader_yaw)

        for i in range(self.num_uavs):
            uav_pos = self.uav_states[i, :3]

            target_pos = self.goal_position if i == 0 else formation_goals[i]
            d_ug = np.linalg.norm(uav_pos - target_pos)

            is_obstacle_near = False
            if self.num_obstacles > 0 and np.any(
                    np.linalg.norm(self.obstacle_positions - uav_pos, axis=1) < R_CFG['d0_obstacle_safe_dist']):
                is_obstacle_near = True

            # --- 1. R_ug (连续部分) ---
            factor = R_CFG['beta_obstacle_decay'] if is_obstacle_near else 1.0
            continuous_R_ug = R_CFG['C1_goal_reached'] * factor * R_CFG['chi_attraction_coeff'] * np.exp(-d_ug)

            # --- 2. R_uo (连续部分) ---
            continuous_R_uo = 0
            if self.num_obstacles > 0 and not collision_info['uav_obs'][i]:
                for d_uo in np.linalg.norm(self.obstacle_positions - uav_pos, axis=1):
                    if d_uo < R_CFG['d0_obstacle_safe_dist']:
                        term1 = (1 / d_uo - 1 / R_CFG['d0_obstacle_safe_dist']) ** 2
                        term2 = d_ug ** (1 / R_CFG['t_adapt_factor']) if R_CFG['t_adapt_factor'] > 0 else 1.0
                        continuous_R_uo += -R_CFG['eta_obstacle_repulsion'] * term1 * term2

            # --- 3. R_uu (连续部分) ---
            continuous_R_uu = 0
            if not collision_info['uav_uav'][i]:
                for j in range(self.num_uavs):
                    if i == j: continue
                    d_uu = np.linalg.norm(uav_pos - self.uav_states[j, :3])
                    if d_uu < R_CFG['d0_uav_safe_dist']:
                        continuous_R_uu += -R_CFG['phi_uav_repulsion'] * np.exp(R_CFG['d0_uav_safe_dist'] - d_uu)

            # --- 4. 组合最终奖励 (包含裁剪) ---

            # 先计算连续部分的加权和
            continuous_reward = R_CFG['w1_goal'] * continuous_R_ug + \
                                R_CFG['w2_obstacle'] * continuous_R_uo + \
                                R_CFG['w3_uav'] * continuous_R_uu

            # [核心修正] 对连续奖励进行裁剪，防止其值过大或过小，以稳定训练
            clipped_continuous_reward = np.clip(continuous_reward, -10.0, 10.0)

            # 然后根据事件决定最终奖励
            if collision_info['uav_obs'][i]:
                final_reward = R_CFG['C2_obstacle_collision']
            elif collision_info['uav_uav'][i]:
                final_reward = R_CFG['C3_uav_collision']
            # 注意：成功奖励现在由 check_done 和 trainer 循环处理，这里只处理连续部分
            # 为了避免与巨大的 C1 奖励混淆，我们将成功事件的奖励判断移出
            # 这里只判断是否接近目标
            elif d_ug < self.goal_threshold and i == 0:
                # 接近目标时，给予一个比裁剪值略大的奖励以示鼓励
                final_reward = 15.0
            else:
                final_reward = clipped_continuous_reward

            rewards[i] = final_reward

        # [补充] 如果领航机成功，给整个团队一个巨大的最终奖励
        if self._check_done(collision_info)[1]:  # _check_done 返回 (is_done, goal_reached)
            rewards += R_CFG['C1_goal_reached']

        return rewards

    def _reset_obstacles(self):
        self.obstacle_positions = np.zeros((self.num_obstacles, 3))
        if self.narrow_channel_scenario:
            num_per_wall = self.num_obstacles // 2
            x_coords = np.linspace(self.env_size[0] * 0.25, self.env_size[0] * 0.75, num_per_wall)
            z_coords = self.env_size[2] / 2
            channel_center_y, channel_width = self.env_size[1] / 2, 15.0
            self.obstacle_positions[:num_per_wall, :] = np.c_[
                x_coords, np.full(num_per_wall, channel_center_y - channel_width / 2), np.full(num_per_wall, z_coords)]
            self.obstacle_positions[num_per_wall:2 * num_per_wall, :] = np.c_[
                x_coords, np.full(num_per_wall, channel_center_y + channel_width / 2), np.full(num_per_wall, z_coords)]
        else:
            self.obstacle_positions = self.np_random.uniform(low=0, high=self.env_size, size=(self.num_obstacles, 3))

    def _reset_uavs(self):
        self.uav_states = np.zeros((self.num_uavs, 9))
        if self.env_config.get('fixed_start_formation', False):
            formation_center = np.array([self.env_size[0] * 0.1, self.env_size[1] / 2, self.env_size[2] / 2])
            initial_yaw = 0.0
            self.uav_states[:, :3] = self.formation_layer.get_formation_goal_positions(formation_center, initial_yaw)
            self.uav_states[:, 8] = initial_yaw
        else:
            leader_start_pos = self.np_random.uniform(
                low=[self.env_size[0] * 0.1, self.env_size[1] * 0.1, self.env_size[2] * 0.5],
                high=[self.env_size[0] * 0.2, self.env_size[1] * 0.9, self.env_size[2] * 0.6])
            leader_start_yaw = self.np_random.uniform(-np.pi, np.pi)
            self.uav_states[:, :3] = self.formation_layer.get_formation_goal_positions(leader_start_pos,
                                                                                       leader_start_yaw)
            self.uav_states[:, 8] = leader_start_yaw

    def _scale_actions(self, actions):
        scaled_actions = np.zeros_like(actions)
        scaled_actions[:, :3] = actions[:, :3] * self.max_velocity
        scaled_actions[:, 3] = actions[:, 3] * self.max_yaw_rate
        return scaled_actions

    def _update_uav_states(self, actions):
        self.uav_states[:, 3:6] = actions[:, :3]
        self.uav_states[:, :3] += self.uav_states[:, 3:6] * self.dt
        self.uav_states[:, 8] = (self.uav_states[:, 8] + actions[:, 3] * self.dt + np.pi) % (2 * np.pi) - np.pi
        self.uav_states[:, :3] = np.clip(self.uav_states[:, :3], 0, self.env_size)

    def _get_obs(self):
        all_obs = np.zeros((self.num_uavs, self.observation_dim))
        leader_pos, leader_yaw = self.uav_states[0, :3], self.uav_states[0, 8]
        formation_goals = self.formation_layer.get_formation_goal_positions(leader_pos, leader_yaw)
        for i in range(self.num_uavs):
            uav_pos, uav_vel, uav_rpy = self.uav_states[i, :3], self.uav_states[i, 3:6], self.uav_states[i, 6:9]
            formation_delta = formation_goals[i] - uav_pos
            goal_delta = self.goal_position - leader_pos if i == 0 else formation_goals[i] - uav_pos
            if self.num_obstacles > 0:
                dist_to_obs = np.linalg.norm(self.obstacle_positions - uav_pos, axis=1)
                closest_obs_delta = self.obstacle_positions[np.argmin(dist_to_obs)] - uav_pos
            else:
                closest_obs_delta = np.zeros(3)
            other_uav_indices = [j for j in range(self.num_uavs) if i != j]
            if other_uav_indices:
                dist_to_uavs = [np.linalg.norm(self.uav_states[j, :3] - uav_pos) for j in other_uav_indices]
                closest_uav_delta = self.uav_states[other_uav_indices[np.argmin(dist_to_uavs)], :3] - uav_pos
            else:
                closest_uav_delta = np.zeros(3)
            all_obs[i] = np.concatenate(
                [uav_pos, uav_vel, uav_rpy, formation_delta, goal_delta, closest_obs_delta, closest_uav_delta])
        return all_obs

    def _check_collisions(self):
        collisions = {'uav_obs': np.zeros(self.num_uavs, dtype=bool), 'uav_uav': np.zeros(self.num_uavs, dtype=bool)}
        if self.num_obstacles > 0:
            for i in range(self.num_uavs):
                if np.any(np.linalg.norm(self.obstacle_positions - self.uav_states[i, :3],
                                         axis=1) < self.collision_threshold):
                    collisions['uav_obs'][i] = True
        for i in range(self.num_uavs):
            for j in range(i + 1, self.num_uavs):
                if np.linalg.norm(self.uav_states[i, :3] - self.uav_states[j, :3]) < self.collision_threshold:
                    collisions['uav_uav'][i] = True;
                    collisions['uav_uav'][j] = True
        return collisions

    def _check_done(self, collision_info):
        is_done = np.any(collision_info['uav_obs']) or np.any(collision_info['uav_uav'])
        dist_to_goal = np.linalg.norm(self.uav_states[0, :3] - self.goal_position)
        goal_reached = dist_to_goal < self.goal_threshold
        if goal_reached: is_done = True
        return is_done, goal_reached

    def _get_info(self):
        return {"distance_to_goal": np.linalg.norm(self.uav_states[0, :3] - self.goal_position),
                "uav_states": self.uav_states.copy(), "goal_position": self.goal_position.copy(),
                "obstacle_positions": self.obstacle_positions.copy(), "trajectory": self.trajectory}