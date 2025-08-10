# -*- coding: utf-8 -*-
# @Time    : 2025/8/7 10:00
# @Author  : Gemini
# @File    : formation_layer.py

import numpy as np

class FormationLayer:
    """
    编队层：负责计算和维护无人机编队的期望位置。
    """
    def __init__(self, num_uavs, formation_config):
        """
        初始化编队层。
        Args:
            num_uavs (int): 无人机数量。
            formation_config (dict): 编队配置参数。
        """
        self.num_uavs = num_uavs
        self.formation_type = formation_config['type']
        self.distance = formation_config['distance']
        self.formation_switch_steps = formation_config.get('formation_switch_steps', None)

        if self.num_uavs < 2:
            raise ValueError("编队至少需要2架无人机。")

    def _get_relative_positions(self, formation_type):
        """
        根据编队类型计算每个无人机相对于编队中心（领航机）的期望相对位置。
        领航机（UAV 0）的位置是 (0, 0, 0)。
        Args:
            formation_type (str): 'line'（一字型）或 'triangle'（三角型）。
        Returns:
            np.ndarray: 形状为 (num_uavs, 3) 的相对位置数组。
        """
        rel_pos = np.zeros((self.num_uavs, 3))
        if formation_type == 'line':
            # 一字型编队，沿 y 轴排列
            for i in range(self.num_uavs):
                rel_pos[i] = np.array([0, -i * self.distance, 0])
        elif formation_type == 'triangle':
            # 三角形编队 (假设 num_uavs >= 3)
            if self.num_uavs < 3:
                print("警告: 三角形编队至少需要3架无人机，将使用一字型编队替代。")
                return self._get_relative_positions('line')
            # 领航机在顶点
            rel_pos[0] = np.array([0, 0, 0])
            # 后续飞机在两侧
            rel_pos[1] = np.array([-self.distance * np.cos(np.pi/6), -self.distance * np.sin(np.pi/6), 0])
            rel_pos[2] = np.array([self.distance * np.cos(np.pi/6), -self.distance * np.sin(np.pi/6), 0])
            # 更多无人机可以继续排列
            for i in range(3, self.num_uavs):
                 rel_pos[i] = np.array([0, - (i//2) * self.distance, 0]) # 简单处理
        else:
            raise ValueError(f"未知的编队类型: {formation_type}")
        return rel_pos

    def update_formation(self, current_step):
        """
        根据当前步数决定是否切换编队形态。
        Args:
            current_step (int): 当前环境步数。
        """
        if self.formation_switch_steps and current_step == self.formation_switch_steps:
            print(f"在步骤 {current_step} 切换编队形态!")
            # 示例：从三角切换到一字型
            if self.formation_type == 'triangle':
                self.formation_type = 'line'
            else:
                self.formation_type = 'triangle'

    def get_formation_goal_positions(self, leader_pos, leader_yaw):
        """
        计算整个编队中所有无人机的绝对目标位置。
        Args:
            leader_pos (np.ndarray): 领航机 (UAV 0) 的当前位置。
            leader_yaw (float): 领航机的航向角（弧度）。
        Returns:
            np.ndarray: 形状为 (num_uavs, 3) 的期望绝对位置数组。
        """
        rel_pos = self._get_relative_positions(self.formation_type)

        # 创建绕 Z 轴的旋转矩阵
        cos_yaw = np.cos(leader_yaw)
        sin_yaw = np.sin(leader_yaw)
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw,  cos_yaw, 0],
            [0,        0,       1]
        ])

        # 将相对位置旋转并加上领航机位置，得到绝对目标位置
        rotated_rel_pos = rel_pos @ rotation_matrix.T
        goal_positions = leader_pos + rotated_rel_pos
        return goal_positions

if __name__ == '__main__':
    # --- 测试 FormationLayer ---
    from configs.default_config import FORMATION_CONFIG
    num_uavs = 3
    formation_layer = FormationLayer(num_uavs, FORMATION_CONFIG)

    leader_position = np.array([10, 20, 5])
    leader_orientation_yaw = np.pi / 4  # 45度

    # 获取初始编队目标
    formation_goals = formation_layer.get_formation_goal_positions(leader_position, leader_orientation_yaw)
    print(f"领航机位置: {leader_position}, 航向角: {np.rad2deg(leader_orientation_yaw)} 度")
    print(f"编队类型 '{formation_layer.formation_type}' 的目标位置:")
    print(formation_goals)

    # 模拟切换编队
    formation_layer.formation_type = 'triangle'
    formation_goals_tri = formation_layer.get_formation_goal_positions(leader_position, leader_orientation_yaw)
    print(f"\n切换到编队类型 '{formation_layer.formation_type}' 的目标位置:")
    print(formation_goals_tri)