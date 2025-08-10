# -*- coding: utf-8 -*-
# @Time    : 2025/8/7 10:00
# @Author  : Gemini
# @File    : trainer.py

import os
import sys
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.envs.multi_uav_env import MultiUAVEnv
# [核心修正 1/3] 从文件中导入新的、不含LSTM的智能体类名
from src.agents.matd3_agent import MATD3_Agent
from src.replay.replay_buffer import ReplayBuffer
from src.replay.prioritized_replay import PrioritizedReplayBuffer
from src.train.utils import set_seed
from configs.default_config import ENV_CONFIG, FORMATION_CONFIG, REWARD_CONFIG_V2, ALG_CONFIG, TRAIN_CONFIG


def main():
    print("==============================================")
    print("正在初始化训练环境、智能体和经验回放池...")
    set_seed(ALG_CONFIG['seed'])

    env = MultiUAVEnv(ENV_CONFIG, FORMATION_CONFIG, REWARD_CONFIG_V2)

    # [核心修正 2/3] 使用新的类名来实例化智能体
    agent = MATD3_Agent(ENV_CONFIG, ALG_CONFIG, REWARD_CONFIG_V2)

    if ALG_CONFIG['use_per']:
        print("使用优先经验回放 (PER)")
        # 注意：这里的ReplayBuffer也应该是无序列的版本
        replay_buffer = PrioritizedReplayBuffer(
            buffer_size=ALG_CONFIG['buffer_size'], batch_size=ALG_CONFIG['batch_size'],
            num_uavs=ENV_CONFIG['num_uavs'], obs_dim=env.observation_dim, action_dim=env.action_dim,
            device=ALG_CONFIG['device'], alpha=ALG_CONFIG['alpha']
        )
    else:
        print("使用标准经验回放")
        replay_buffer = ReplayBuffer(
            buffer_size=ALG_CONFIG['buffer_size'], batch_size=ALG_CONFIG['batch_size'],
            num_uavs=ENV_CONFIG['num_uavs'], obs_dim=env.observation_dim, action_dim=env.action_dim,
            device=ALG_CONFIG['device']
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/MATD3_{timestamp}"  # 更新日志文件夹名称
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 日志将保存在: {log_dir}")
    print("==============================================\n")

    total_timesteps = 0
    log_episode_rewards, log_critic_losses, log_actor_losses = [], [], []

    print("==============================================")
    print(f"开始训练，共 {TRAIN_CONFIG['max_episodes']} 个回合...")
    print(f"训练设备: {ALG_CONFIG['device']}")
    if TRAIN_CONFIG['live_animation_while_training']:
        print("\033[93m注意: 已开启训练时渲染，这会显著降低训练速度！\033[0m")
    print("==============================================")

    for episode in range(1, TRAIN_CONFIG['max_episodes'] + 1):
        print(f"\n--- Episode {episode}/{TRAIN_CONFIG['max_episodes']} ---")

        obs, _ = env.reset()
        agent.reset_obs_seq_buffers()  # 即使是空方法，也保持调用
        episode_reward, episode_steps = 0, 0

        while True:
            total_timesteps, episode_steps = total_timesteps + 1, episode_steps + 1

            if total_timesteps < TRAIN_CONFIG['start_timesteps']:
                actions = env.action_space.sample()
            else:
                actions = agent.select_actions(obs, noise_std=TRAIN_CONFIG['exploration_noise'])

            next_obs, rewards, terminated, truncated, info = env.step(actions)

            done_flag = terminated or truncated
            replay_buffer.add(obs, actions, np.full(ENV_CONFIG['num_uavs'], rewards.mean()), next_obs,
                              np.full(ENV_CONFIG['num_uavs'], done_flag))

            obs, episode_reward = next_obs, episode_reward + rewards.mean()

            if len(replay_buffer) > ALG_CONFIG['batch_size']:  # 无序列，不再需要乘以seq_len
                critic_loss, actor_loss = agent.update(replay_buffer, use_per=ALG_CONFIG['use_per'])
                if critic_loss is not None:
                    writer.add_scalar('Loss/Critic_Loss', critic_loss, total_timesteps);
                    log_critic_losses.append(critic_loss)
                if actor_loss is not None and actor_loss != 0:
                    writer.add_scalar('Loss/Actor_Loss', actor_loss, total_timesteps);
                    log_actor_losses.append(actor_loss)

            if done_flag: break

        log_episode_rewards.append(episode_reward)
        writer.add_scalar('Reward/Episode_Reward', episode_reward, episode)
        writer.add_scalar('Metrics/Episode_Steps', episode_steps, episode)
        writer.add_scalar('Metrics/Distance_to_Goal', info['distance_to_goal'], episode)
        writer.add_scalar('Metrics/Success', 1 if info.get('is_success', False) else 0, episode)

        print(f"回合结束 | 步数: {episode_steps} | 奖励: {episode_reward:.2f} | 成功: {info.get('is_success', False)}")

        if TRAIN_CONFIG['live_animation_while_training']: env.render_live()

        if episode % TRAIN_CONFIG['log_interval'] == 0 and episode > 0:
            avg_reward = np.mean(log_episode_rewards) if log_episode_rewards else 0
            avg_critic_loss = np.mean(log_critic_losses) if log_critic_losses else 0
            avg_actor_loss = np.mean(log_actor_losses) if log_actor_losses else 0
            print(
                f"\n\033[92m周期报告 (Episodes {episode - len(log_episode_rewards) + 1}-{episode}):\n  平均奖励: {avg_reward:.2f}\n  平均 Critic Loss: {avg_critic_loss:.6f}\n  平均 Actor Loss: {avg_actor_loss:.6f}\n========================================================\033[0m\n")
            log_episode_rewards.clear();
            log_critic_losses.clear();
            log_actor_losses.clear()

        if episode % TRAIN_CONFIG['save_interval'] == 0 and episode > 0:
            print(f"\n在回合 {episode} 保存模型...");
            agent.save(log_dir, episode);
            print("模型保存完毕。\n")

    writer.close();
    env.close()
    print("==============================================");
    print("训练完成！");
    print(f"最终模型和日志保存在: {log_dir}");
    print("==============================================")


if __name__ == '__main__':
    main()