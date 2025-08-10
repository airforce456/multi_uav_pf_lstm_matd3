# -*- coding: utf-8 -*-
# @Time    : 2025/8/7 10:00
# @Author  : Gemini
# @File    : critic.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """
    [无 LSTM 版本] Critic 网络 (价值网络)，采用双 Critic 结构
    - 输入: 所有智能体的联合观测 + 所有智能体的联合动作
    - 输出: 两个 Q 值 (Q1, Q2)
    """

    def __init__(self, num_uavs, obs_dim, action_dim):
        """
        初始化 Critic 网络。
        Args:
            num_uavs (int): 无人机数量。
            obs_dim (int): 单个无人机的观测维度。
            action_dim (int): 单个无人机的动作维度。
        """
        super(Critic, self).__init__()

        # 中心化输入维度
        central_obs_dim = num_uavs * obs_dim
        central_action_dim = num_uavs * action_dim

        # --- Q1 网络 ---
        # 输入是拼接后的 (联合观测 + 联合动作)
        self.fc1_q1 = nn.Linear(central_obs_dim + central_action_dim, 512)
        self.fc2_q1 = nn.Linear(512, 256)
        self.fc3_q1 = nn.Linear(256, 1)

        # --- Q2 网络 ---
        self.fc1_q2 = nn.Linear(central_obs_dim + central_action_dim, 512)
        self.fc2_q2 = nn.Linear(512, 256)
        self.fc3_q2 = nn.Linear(256, 1)

    def forward(self, central_obs, central_action):
        """
        前向传播。
        Args:
            central_obs (torch.Tensor): 中心化的观测, 形状 (batch_size, num_uavs * obs_dim)。
            central_action (torch.Tensor): 中心化的动作, 形状 (batch_size, num_uavs * action_dim)。
        Returns:
            tuple: (Q1, Q2) 两个 Q 值。
        """
        # 将观测和动作拼接起来
        x = torch.cat([central_obs, central_action], dim=1)

        # --- Q1 计算 ---
        q1 = F.relu(self.fc1_q1(x))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)

        # --- Q2 计算 ---
        q2 = F.relu(self.fc1_q2(x))
        q2 = F.relu(self.fc2_q2(q2))
        q2 = self.fc3_q2(q2)

        return q1, q2

    def Q1(self, central_obs, central_action):
        """
        只计算 Q1 值，用于 Actor 的损失计算。
        """
        x = torch.cat([central_obs, central_action], dim=1)
        q1 = F.relu(self.fc1_q1(x))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)
        return q1