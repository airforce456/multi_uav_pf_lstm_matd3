# -*- coding: utf-8 -*-
# @Time    : 2025/8/7 10:00
# @Author  : Gemini
# @File    : actor.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    [无 LSTM 版本] Actor 网络 (策略网络)
    - 输入: 单个智能体的当前观测 (一个向量)
    - 输出: 单个智能体的确定性动作
    """

    def __init__(self, obs_dim, action_dim, max_action):
        """
        初始化 Actor 网络。
        Args:
            obs_dim (int): 观测空间的维度。
            action_dim (int): 动作空间的维度。
            max_action (float): 动作的最大值。
        """
        super(Actor, self).__init__()
        self.max_action = max_action

        # 使用简单的全连接层 (MLP)
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, obs):
        """
        前向传播。
        Args:
            obs (torch.Tensor): 当前的观测，形状为 (batch_size, obs_dim)。
        Returns:
            torch.Tensor: 动作值，形状为 (batch_size, action_dim)，值域在 [-max_action, max_action]。
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))

        # 输出层使用 tanh 激活函数，将输出范围缩放到 [-1, 1]
        # 然后乘以 max_action 得到最终的动作范围
        action = self.max_action * torch.tanh(self.fc3(x))

        return action