# -*- coding: utf-8 -*-
# @Time    : 2025/8/7 10:00
# @Author  : Gemini
# @File    : visualization.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def animate_step(fig, ax, trajectories, goal, obstacles, env_size, title="UAV Live Animation"):
    """
    [新增] 绘制动画中的一帧。
    这个函数会清空当前的坐标轴并重绘所有元素。
    """
    ax.cla()  # 清空上一帧的画面

    # --- 绘制轨迹线 ---
    colors = plt.cm.jet(np.linspace(0, 1, len(trajectories)))
    for i, traj in enumerate(trajectories):
        if len(traj) > 1:
            traj_array = np.array(traj)
            ax.plot(traj_array[:, 0], traj_array[:, 1], traj_array[:, 2], '-', color=colors[i], alpha=0.5)

    # --- 绘制无人机当前位置 ---
    for i, traj in enumerate(trajectories):
        if traj:  # 如果轨迹不为空
            current_pos = traj[-1]
            ax.scatter(current_pos[0], current_pos[1], current_pos[2], color=colors[i], marker='o', s=80,
                       label=f'UAV {i}' if len(traj) == 1 else None, depthshade=True)

    # --- 绘制目标和障碍物 ---
    ax.scatter(goal[0], goal[1], goal[2], color='green', marker='*', s=200, label='Goal', depthshade=True)
    if obstacles is not None and len(obstacles) > 0:
        ax.scatter(obstacles[:, 0], obstacles[:, 1], obstacles[:, 2], color='red', marker='s', s=100, label='Obstacles',
                   depthshade=True)

    # --- 设置坐标轴和标题 ---
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_xlim(0, env_size[0])
    ax.set_ylim(0, env_size[1])
    ax.set_zlim(0, env_size[2])
    ax.set_title(title)

    # 只在第一帧时创建图例，避免重复
    if len(trajectories[0]) == 1:
        ax.legend()

    fig.canvas.draw()
    fig.canvas.flush_events()


def plot_trajectories_3d(trajectories, goal, obstacles, env_size, title="UAV Final Trajectory", save_path=None):
    """
    (保持不变) 使用 Matplotlib 绘制静态的、完整的 3D 轨迹图。
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.jet(np.linspace(0, 1, len(trajectories)))
    for i, traj in enumerate(trajectories):
        traj_array = np.array(traj)
        ax.plot(traj_array[:, 0], traj_array[:, 1], traj_array[:, 2], '-', color=colors[i], label=f'UAV {i}')
        ax.scatter(traj_array[0, 0], traj_array[0, 1], traj_array[0, 2], color=colors[i], marker='o', s=100,
                   label=f'UAV {i} Start')
        ax.scatter(traj_array[-1, 0], traj_array[-1, 1], traj_array[-1, 2], color=colors[i], marker='x', s=100,
                   label=f'UAV {i} End')

    ax.scatter(goal[0], goal[1], goal[2], color='green', marker='*', s=200, label='Goal')
    if obstacles is not None and len(obstacles) > 0:
        ax.scatter(obstacles[:, 0], obstacles[:, 1], obstacles[:, 2], color='red', marker='s', s=150, label='Obstacles')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_xlim(0, env_size[0])
    ax.set_ylim(0, env_size[1])
    ax.set_zlim(0, env_size[2])
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"轨迹图已保存到: {save_path}")

    plt.show()