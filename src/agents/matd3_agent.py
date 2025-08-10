# -*- coding: utf-8 -*-
# @Time    : 2025/8/7 10:00
# @Author  : Gemini
# @File    : pf_lstm_matd3_agent.py

import torch
import torch.nn.functional as F
import numpy as np
from collections import deque

from ..models.actor import Actor
from ..models.critic import Critic


class MATD3_Agent:  # [修改] 类名
    """
    [无 LSTM 版本] MATD3 算法的核心智能体。
    """

    def __init__(self, env_config, alg_config, reward_config):
        self.num_uavs = env_config['num_uavs']
        self.action_dim = 4
        self.obs_dim = 21
        self.max_action = 1.0

        self.device = alg_config['device']
        self.gamma = alg_config['gamma']
        self.tau = alg_config['tau']
        self.policy_noise = alg_config['policy_noise']
        self.noise_clip = alg_config['noise_clip']
        self.policy_freq = alg_config['policy_freq']

        # [修改] Actor 和 Critic 的初始化不再需要 seq_len
        self.actors = [Actor(self.obs_dim, self.action_dim, self.max_action).to(self.device) for _ in
                       range(self.num_uavs)]
        self.actor_targets = [Actor(self.obs_dim, self.action_dim, self.max_action).to(self.device) for _ in
                              range(self.num_uavs)]
        for i in range(self.num_uavs):
            self.actor_targets[i].load_state_dict(self.actors[i].state_dict())

        self.critic = Critic(self.num_uavs, self.obs_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.num_uavs, self.obs_dim, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizers = [torch.optim.Adam(self.actors[i].parameters(), lr=alg_config['actor_lr']) for i in
                                 range(self.num_uavs)]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=alg_config['critic_lr'])

        self.train_step_counter = 0

    def select_actions(self, obs, noise_std=0.1):
        """
        [修改] 不再需要处理序列
        """
        actions = []
        for i in range(self.num_uavs):
            # 直接将单个观测向量转换为 Tensor
            obs_tensor = torch.FloatTensor(obs[i]).to(self.device).unsqueeze(0)  # shape: (1, obs_dim)

            self.actors[i].eval()
            with torch.no_grad():
                action = self.actors[i](obs_tensor).cpu().data.numpy().flatten()
            self.actors[i].train()

            if noise_std > 0:
                noise = np.random.normal(0, self.max_action * noise_std, size=self.action_dim)
                action = (action + noise).clip(-self.max_action, self.max_action)

            actions.append(action)

        return np.array(actions)

    def reset_obs_seq_buffers(self):
        # [修改] 这个方法现在是空的，但为了兼容 trainer.py，暂时保留
        pass

    def update(self, replay_buffer, use_per=False):
        """
        [修改] 不再处理序列数据，直接使用 (o, a, r, o', d)
        """
        self.train_step_counter += 1

        # --- [修改] 采样的数据不再是序列 ---
        # 注意：ReplayBuffer 也需要修改，这里先假设它返回非序列数据
        if use_per:
            obs, actions, rewards, next_obs, dones, batch_indices, is_weights = replay_buffer.sample()
            if obs is None: return None, None
        else:
            obs, actions, rewards, next_obs, dones = replay_buffer.sample()
            if obs is None: return None, None
            is_weights = torch.ones((replay_buffer.batch_size, 1)).to(self.device)

        # --- 数据预处理 ---
        batch_size = obs.shape[0]
        central_obs = obs.view(batch_size, -1)
        central_next_obs = next_obs.view(batch_size, -1)
        central_actions = actions.view(batch_size, -1)

        rewards = rewards[:, 0].unsqueeze(1)  # 取团队奖励
        dones = dones[:, 0].unsqueeze(1)

        with torch.no_grad():
            next_actions = [self.actor_targets[i](next_obs[:, i, :]) for i in range(self.num_uavs)]
            central_next_actions = torch.cat(next_actions, dim=1)

            noise = (torch.randn_like(central_next_actions) * self.policy_noise).clamp(-self.noise_clip,
                                                                                       self.noise_clip)
            central_next_actions = (central_next_actions + noise).clamp(-self.max_action, self.max_action)

            target_q1, target_q2 = self.critic_target(central_next_obs, central_next_actions)
            target_q = torch.min(target_q1, target_q2)

            target_q = rewards + self.gamma * (1 - dones.float()) * target_q

        current_q1, current_q2 = self.critic(central_obs, central_actions)

        critic_loss_q1 = F.mse_loss(current_q1, target_q, reduction='none')
        critic_loss_q2 = F.mse_loss(current_q2, target_q, reduction='none')
        critic_loss = torch.mean(is_weights * (critic_loss_q1 + critic_loss_q2))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        actor_loss = None
        if self.train_step_counter % self.policy_freq == 0:
            actor_actions = [self.actors[i](obs[:, i, :]) for i in range(self.num_uavs)]
            central_actor_actions = torch.cat(actor_actions, dim=1)

            actor_loss_value = -self.critic.Q1(central_obs, central_actor_actions).mean()

            self.critic_optimizer.zero_grad()
            for i in range(self.num_uavs): self.actor_optimizers[i].zero_grad()
            actor_loss_value.backward()
            for i in range(self.num_uavs):
                torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
                self.actor_optimizers[i].step()

            self.soft_update()
            actor_loss = actor_loss_value.item()

        if use_per:
            with torch.no_grad():
                td_error = torch.max(torch.abs(current_q1 - target_q), torch.abs(current_q2 - target_q))
                replay_buffer.batch_update(batch_indices, td_error.squeeze(1).cpu().numpy())

        return critic_loss.item(), actor_loss

    def soft_update(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for i in range(self.num_uavs):
            for param, target_param in zip(self.actors[i].parameters(), self.actor_targets[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, directory, episode):
        for i in range(self.num_uavs):
            torch.save(self.actors[i].state_dict(), f'{directory}/actor_{i}_episode_{episode}.pth')
            torch.save(self.actor_targets[i].state_dict(), f'{directory}/actor_target_{i}_episode_{episode}.pth')

        torch.save(self.critic.state_dict(), f'{directory}/critic_episode_{episode}.pth')
        torch.save(self.critic_target.state_dict(), f'{directory}/critic_target_episode_{episode}.pth')

    def load(self, directory, episode):
        for i in range(self.num_uavs):
            self.actors[i].load_state_dict(
                torch.load(f'{directory}/actor_{i}_episode_{episode}.pth', map_location=self.device))
            self.actor_targets[i].load_state_dict(
                torch.load(f'{directory}/actor_target_{i}_episode_{episode}.pth', map_location=self.device))

        self.critic.load_state_dict(torch.load(f'{directory}/critic_episode_{episode}.pth', map_location=self.device))
        self.critic_target.load_state_dict(
            torch.load(f'{directory}/critic_target_episode_{episode}.pth', map_location=self.device))