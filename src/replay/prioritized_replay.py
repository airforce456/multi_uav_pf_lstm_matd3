# -*- coding: utf-8 -*-
# @Time    : 2025/8/7 10:00
# @Author  : Gemini
# @File    : prioritized_replay.py

import numpy as np
import torch
import random


class SumTree:
    """
    SumTree 数据结构，用于高效地按优先级采样。
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0

    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx], data_idx

    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    """
    优先经验回放 (PER) 缓冲区。
    """

    def __init__(self, buffer_size, batch_size, seq_len, num_uavs, obs_dim, action_dim, device, alpha=0.6,
                 beta_start=0.4, beta_end=1.0, beta_decay_steps=100000):
        self.tree = SumTree(buffer_size)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_uavs = num_uavs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.epsilon = 1e-5
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_decay_steps = beta_decay_steps
        self.current_step = 0
        self.max_priority = 1.0

    def add(self, obs, actions, rewards, next_obs, dones):
        experience = (
            np.array(obs, dtype=np.float32), np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32), np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=bool)
        )
        self.tree.add(self.max_priority, experience)

    def sample(self):
        if len(self) < self.batch_size * self.seq_len:
            return None, None, None

        batch_indices = np.empty((self.batch_size,), dtype=np.int32)
        batch_memory = [[] for _ in range(self.batch_size)]
        is_weights = np.empty((self.batch_size, 1), dtype=np.float32)

        priority_segment = self.tree.total_priority / self.batch_size
        self.beta = min(self.beta_end, self.beta_start + (self.beta_end - self.beta_start) * (
                    self.current_step / self.beta_decay_steps))
        self.current_step += 1

        for i in range(self.batch_size):
            # [最终修正] 增加一个重试计数器和逃逸机制，彻底防止无限循环
            retry_count = 0
            while True:
                if retry_count < 10:  # 先在指定的小区间内尝试10次
                    a = priority_segment * i
                    b = priority_segment * (i + 1)
                    value = random.uniform(a, b)
                else:  # 如果10次都失败了，就从整个范围随机采样，确保能找到一个点
                    value = random.uniform(0, self.tree.total_priority)

                tree_idx, priority, _, data_idx = self.tree.get_leaf(value)

                if self._is_sequence_valid(data_idx):
                    break  # 找到有效序列，跳出 while 循环

                retry_count += 1

            sampling_prob = priority / self.tree.total_priority
            is_weights[i, 0] = np.power(sampling_prob * len(self), -self.beta)
            batch_indices[i] = tree_idx

            start_idx = (data_idx - self.seq_len + 1 + self.tree.capacity) % self.tree.capacity

            if start_idx + self.seq_len <= self.tree.capacity:
                seq_data = [self.tree.data[j] for j in range(start_idx, start_idx + self.seq_len)]
            else:
                part1_len = self.tree.capacity - start_idx
                part2_len = self.seq_len - part1_len
                part1 = [self.tree.data[j] for j in range(start_idx, self.tree.capacity)]
                part2 = [self.tree.data[j] for j in range(part2_len)]
                seq_data = part1 + part2

            batch_memory[i] = seq_data

        is_weights /= is_weights.max()

        obs, act, rew, next_obs, done = [], [], [], [], []
        for seq in batch_memory:
            o, a, r, no, d = zip(*seq)
            obs.append(np.stack(o))
            act.append(np.stack(a))
            rew.append(np.stack(r))
            next_obs.append(np.stack(no))
            done.append(np.stack(d))

        return self._to_torch(obs, act, rew, next_obs, done), batch_indices, torch.tensor(is_weights,
                                                                                          device=self.device)

    def _is_sequence_valid(self, end_data_idx):
        if self.tree.n_entries < self.seq_len:
            return False  # 缓冲区数据少于一个序列长度

        start_data_idx = (end_data_idx - self.seq_len + 1 + self.tree.capacity) % self.tree.capacity

        # 检查序列是否跨越了写入指针 (data_pointer)，这表示序列不连续
        if self.tree.n_entries == self.tree.capacity:  # 缓冲区已满
            if start_data_idx <= end_data_idx:  # 正常情况
                if start_data_idx < self.tree.data_pointer <= end_data_idx:
                    return False
            else:  # 环形情况
                if self.tree.data_pointer > start_data_idx or self.tree.data_pointer <= end_data_idx:
                    return False
        else:  # 缓冲区未满
            if start_data_idx > end_data_idx:
                return False

        return True

    def batch_update(self, tree_indices, abs_errors):
        abs_errors += self.epsilon
        clipped_errors = np.minimum(abs_errors, self.max_priority)
        priorities = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_indices, priorities):
            self.tree.update(ti, p)

    def _to_torch(self, obs, actions, rewards, next_obs, dones):
        return (
            torch.tensor(np.array(obs), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(actions), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(next_obs), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(dones), dtype=torch.bool, device=self.device)
        )

    def __len__(self):
        return self.tree.n_entries