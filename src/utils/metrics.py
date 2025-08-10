# -*- coding: utf-8 -*-
# @Time    : 2025/8/7 10:00
# @Author  : Gemini
# @File    : metrics.py

import numpy as np

def calculate_metrics(results):
    """
    根据评估结果列表计算各项性能指标。
    Args:
        results (list[dict]): 包含每次评估回合结果的字典列表。
                              每个字典应包含 'success', 'collision', 'reward', 'distance_to_goal' 等键。
    Returns:
        dict: 包含计算出的指标的字典，如成功率、碰撞率等。
    """
    num_episodes = len(results)
    if num_episodes == 0:
        return {
            'success_rate': 0,
            'collision_rate': 0,
            'average_reward': 0,
            'average_final_distance': 0
        }

    success_count = sum(1 for r in results if r['success'])
    collision_count = sum(1 for r in results if r['collision'])
    total_reward = sum(r['reward'] for r in results)
    total_final_distance = sum(r['distance_to_goal'] for r in results)

    metrics = {
        'success_rate': success_count / num_episodes,
        'collision_rate': collision_count / num_episodes,
        'average_reward': total_reward / num_episodes,
        'average_final_distance': total_final_distance / num_episodes
    }

    return metrics