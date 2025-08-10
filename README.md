# 多无人机协同轨迹规划 (PF-LSTM-MATD3)

本项目基于论文《Multi-UAV Adaptive Cooperative Formation Trajectory Planning Based on an Improved MATD3 Algorithm of Deep Reinforcement Learning》，使用 PyTorch 实现 PF-LSTM-MATD3 算法，用于多无人机在复杂环境下的协同编队与轨迹规划。

## 功能特性

- **核心算法**: MATD3 (Multi-Agent Twin Delayed Deep Deterministic Policy Gradient)
- **时序感知**: Actor 和 Critic 网络均集成 LSTM，处理观测序列。
- **稠密奖励**: 结合人工势场法（目标吸引、障碍物排斥、无人机间排斥）设计奖励函数。
- **高效学习**: 支持优先经验回放 (Prioritized Experience Replay, PER)。
- **分层控制**: 逻辑上分为编队层、轨迹规划层和动作执行层。
- **灵活编队**: 支持多种编队形态切换（如线性、三角形）。
- **可视化**: 提供 3D 轨迹可视化和 TensorBoard 训练过程监控。
- **模块化设计**: 代码结构清晰，易于理解和扩展。

## 环境配置 (Windows + PyCharm)

1.  **创建 Conda 环境 (推荐)**
    打开 Anaconda Prompt，执行以下命令：
    ```bash
    conda create -n uav_env python=3.9
    conda activate uav_env
    ```

2.  **安装依赖库**
    在激活的 Conda 环境中，进入项目根目录，然后运行：
    ```bash
    pip install -r requirements.txt
    ```

3.  **配置 PyCharm 解析器**
    - 打开 PyCharm > Settings > Project: multi_uav_pf_lstm_matd3 > Python Interpreter。
    - 点击 "Add Interpreter" > "Add Local Interpreter"。
    - 选择 "Conda Environment"，并从下拉菜单中选择你刚刚创建的 `uav_env` 环境。

## 运行项目

所有脚本都应在 PyCharm 的终端中，或配置 PyCharm 的 "Run/Debug Configurations" 来运行。请确保当前工作目录是项目根目录 (`multi_uav_pf_lstm_matd3`)。

### 1. 训练模型

执行 `trainer.py` 脚本开始训练。训练日志和模型文件将保存在 `logs/` 目录下。

```bash
# 在 PyCharm 终端中运行
python src/train/trainer.py

你可以在 configs/default_config.py 文件中修改超参数，如学习率、智能体数量、场景设置等。
训练过程中，可以启动 TensorBoard 查看学习曲线：
code
Bash
tensorboard --logdir=logs
然后在浏览器中打开 http://localhost:6006/。
2. 评估与可视化
训练完成后，使用 evaluate.py 脚本加载已保存的模型进行测试，并生成 3D 轨迹可视化结果。
code
Bash
# 在 PyCharm 终端中运行
# 需要将 'your_model_timestamp' 替换为实际训练生成的模型文件夹名称
python src/eval/evaluate.py --model_dir logs/your_model_timestamp
评估脚本会运行一个或多个测试回合，并在结束后显示 3D 轨迹图。
轨迹图等可视化结果也会保存在模型目录中。