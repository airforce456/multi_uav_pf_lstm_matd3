# -*- coding: utf-8 -*-
# @Time    : 2025/8/7 10:00
# @Author  : Gemini
# @File    : evaluate.py

import os
import sys
import torch
import numpy as np
import argparse
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.envs.multi_uav_env import MultiUAVEnv
from src.agents.matd3_agent import MATD3_Agent
from src.utils.metrics import calculate_metrics
# [核心修正] 导入 REWARD_CONFIG_V2 而不是旧的 REWARD_CONFIG
from configs.default_config import ENV_CONFIG, FORMATION_CONFIG, REWARD_CONFIG_V2, ALG_CONFIG, EVAL_CONFIG


def main(args):
    print("==============================================")
    print("开始评估已训练的模型...")

    if args.episode is None:
        files = [f for f in os.listdir(args.model_dir) if f.startswith('actor_0_episode_')]
        if not files: raise FileNotFoundError(f"在目录 {args.model_dir} 中找不到任何模型文件。")
        latest_episode = max([int(f.split('_')[-1].split('.')[0]) for f in files])
        print(f"自动选择最新模型，回合: {latest_episode}")
    else:
        latest_episode = args.episode

    # [核心修正] 使用 REWARD_CONFIG_V2 初始化环境
    env = MultiUAVEnv(ENV_CONFIG, FORMATION_CONFIG, REWARD_CONFIG_V2)
    # [核心修正] 保持一致性，也传入 REWARD_CONFIG_V2
    agent = MATD3_Agent(ENV_CONFIG, ALG_CONFIG, REWARD_CONFIG_V2)

    try:
        agent.load(args.model_dir, latest_episode)
        print(f"成功加载回合 {latest_episode} 的模型。")
    except FileNotFoundError:
        print(f"错误: 找不到回合 {latest_episode} 的模型文件。请检查路径和回合数。")
        return

    all_results = []
    for ep in range(EVAL_CONFIG['eval_episodes']):
        print(f"\n--- 评估回合 {ep + 1}/{EVAL_CONFIG['eval_episodes']} ---")
        obs, _ = env.reset()
        agent.reset_obs_seq_buffers()
        done = False

        while not done:
            if EVAL_CONFIG['live_animation']:
                env.render_live(frame_delay=EVAL_CONFIG['animation_frame_delay'])

            actions = agent.select_actions(obs, noise_std=0)
            obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

        print(
            f"回合结束 | 成功: {info.get('is_success', False)} | 碰撞: {np.any(info['collisions']['uav_obs']) or np.any(info['collisions']['uav_uav'])}")

        if EVAL_CONFIG['plot_final_trajectory']:
            env.render(title=f"Final Trajectory of Eval Episode {ep + 1}")

        all_results.append({
            'success': info.get('is_success', False),
            'collision': np.any(info['collisions']['uav_obs']) or np.any(info['collisions']['uav_uav']),
            'reward': 0,  # 评估时可以不严格计算奖励
            'distance_to_goal': info['distance_to_goal'],
        })

    env.close()

    metrics = calculate_metrics(all_results)
    print("\n================ 评估结果 ================")
    print(f"成功率: {metrics.get('success_rate', 0):.2%}")
    print(f"碰撞率: {metrics.get('collision_rate', 0):.2%}")
    print(f"平均最终目标距离: {metrics.get('average_final_distance', 0):.2f}")
    print("==============================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="评估 PF-LSTM-MATD3 模型")
    parser.add_argument("--model_dir", type=str, required=True, help="包含已保存模型的目录路径。")
    parser.add_argument("--episode", type=int, default=None, help="要加载的特定模型的回合数。")
    main(parser.parse_args())