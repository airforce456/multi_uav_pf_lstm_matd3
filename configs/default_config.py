# -*- coding: utf-8 -*-
# @Time    : 2025/8/7 10:00
# @Author  : Gemini
# @File    : default_config.py

import torch
import numpy as np

# --- 全局与环境配置 ---
ENV_CONFIG = {
    'num_uavs': 3,
    'num_obstacles': 5,
    'env_size': np.array([100, 100, 50]),
    'dt': 0.1,
    'max_steps': 500,
    'goal_threshold': 2.0,
    'collision_threshold': 1.5,
    'fixed_start_formation': True,
    # [重要建议] 在初步训练时，请将此项设为 False 以简化任务
    'narrow_channel_scenario': False,
}

# --- 编队配置 ---
FORMATION_CONFIG = {
    'type': 'triangle',
    'distance': 5.0,
    'formation_switch_steps': None,
}

# --- 算法超参数 ---
ALG_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'gamma': 0.99,
    'tau': 0.005,
    'actor_lr': 0.001,
    'critic_lr': 0.001,
    'batch_size': 256,
    'buffer_size': int(1e6),
    'seq_len': 8,
    'use_per': False,
    'alpha': 0.6,
    'beta_start': 0.4,
    'beta_end': 1.0,
    'policy_noise': 0.2,
    'noise_clip': 0.5,
    'policy_freq': 2,
}

# --- [最终版] 奖励函数配置 (V2) ---
# 这个配置对应我们最终修正的 V4 版奖励函数逻辑
REWARD_CONFIG_V2 = {
    # --- 总体权重 ---
    'w1_goal': 0.5,     # R_ug 的权重
    'w2_obstacle': 0.3, # R_uo 的权重
    'w3_uav': 0.2,      # R_uu 的权重

    # --- 1. R_ug: 目标吸引奖励参数 ---
    'C1_goal_reached': 100.0,       # 到达目标的固定正奖励
    'rho_goal_radius': 15.0,        # 接近目标的半径阈值 (用于切换长/近程奖励)
    'chi_attraction_coeff': 0.8,    # (0,1) 近程指数吸引强度系数
    # 'beta_obstacle_decay' 在最终版奖励函数中不再需要，但保留以防万一
    'beta_obstacle_decay': 0.5,

    # --- 2. R_uo: 障碍物排斥奖励参数 ---
    'd0_obstacle_safe_dist': 6.0,   # 障碍物安全距离
    'eta_obstacle_repulsion': 0.8,  # (0,1) 障碍斥力系数
    # 't_adapt_factor' 在最终版奖励函数中不再需要，已解耦
    't_adapt_factor': 0.5,
    'C2_obstacle_collision': -100.0,# 碰撞障碍的固定负奖励

    # --- 3. R_uu: 无人机间斥力奖励参数 ---
    'd0_uav_safe_dist': 8.0,        # 无人机间安全距离
    'phi_uav_repulsion': 0.8,       # (0,1) 无人机间斥力系数
    'C3_uav_collision': -100.0,     # 碰撞UAV的固定负奖励
}

# --- 训练与评估配置 ---
TRAIN_CONFIG = {
    'max_episodes': 5000,
    'exploration_noise': 0.1,
    'start_timesteps': 25e3,
    'log_interval': 100,
    'save_interval': 100,
    'live_animation_while_training': False,
}

EVAL_CONFIG = {
    'eval_episodes': 10,
    'live_animation': True,
    'plot_final_trajectory': False,
    'animation_frame_delay': 0.05,
}