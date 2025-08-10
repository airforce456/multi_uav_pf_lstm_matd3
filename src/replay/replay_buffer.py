# 文件: src/replay/replay_buffer.py (无序列版本)

import numpy as np
import torch


class ReplayBuffer:
    """
    标准的、非序列的经验回放池。
    """

    def __init__(self, buffer_size, batch_size, num_uavs, obs_dim, action_dim, device):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0
        self.device = device

        self.obs = np.zeros((buffer_size, num_uavs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_uavs, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, num_uavs), dtype=np.float32)
        self.next_obs = np.zeros((buffer_size, num_uavs, obs_dim), dtype=np.float32)
        self.dones = np.zeros((buffer_size, num_uavs), dtype=np.float32)

    def add(self, obs, actions, rewards, next_obs, dones):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = dones

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self):
        if self.size < self.batch_size:
            # 返回 None 或空元组，让调用者知道数据不足
            return None, None, None, None, None

        ind = np.random.randint(0, self.size, size=self.batch_size)

        return (
            torch.FloatTensor(self.obs[ind]).to(self.device),
            torch.FloatTensor(self.actions[ind]).to(self.device),
            torch.FloatTensor(self.rewards[ind]).to(self.device),
            torch.FloatTensor(self.next_obs[ind]).to(self.device),
            torch.FloatTensor(self.dones[ind]).to(self.device)
        )

    def __len__(self):
        """
        [核心修正] 返回当前缓冲区中存储的经验数量。
        """
        return self.size