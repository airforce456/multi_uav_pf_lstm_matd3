# -*- coding: utf-8 -*-
# @Time    : 2025/8/7 10:00
# @Author  : Gemini
# @File    : lstm_encoder.py

import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    """
    LSTM 编码器，用于从观测序列中提取时序特征。
    """
    def __init__(self, input_dim, hidden_dim, num_layers=1, batch_first=True):
        """
        初始化 LSTM 编码器。
        Args:
            input_dim (int): 输入特征的维度。
            hidden_dim (int): LSTM 隐藏层的维度。  <--- [修正] 添加了此参数
            num_layers (int): LSTM 的层数。
            batch_first (bool): 输入和输出张量的第一个维度是否是 batch_size。
        """
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,  # <--- [修正] 使用了此参数
            num_layers=num_layers,
            batch_first=batch_first
        )

    def forward(self, x, hidden_state=None):
        """
        前向传播。
        Args:
            x (torch.Tensor): 输入的序列数据，形状为 (batch_size, seq_len, input_dim)。
            hidden_state (tuple, optional): LSTM 的初始隐藏状态 (h_0, c_0)。默认为 None。
        Returns:
            tuple:
                - torch.Tensor: LSTM 最后一层的输出特征序列，形状为 (batch_size, seq_len, hidden_dim)。
                - tuple: 最终的隐藏状态 (h_n, c_n)。
        """
        # LSTM 的输出包括:
        # output: (batch_size, seq_len, hidden_dim) - 每个时间步的输出
        # (h_n, c_n): (num_layers, batch_size, hidden_dim) - 最后一个时间步的隐藏状态和细胞状态
        output, (h_n, c_n) = self.lstm(x, hidden_state)
        return output, (h_n, c_n)