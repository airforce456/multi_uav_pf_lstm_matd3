# -*- coding: utf-8 -*-
# @Time    : 2025/8/7 10:00
# @Author  : Gemini
# @File    : utils.py

import torch
import numpy as np
import random

def set_seed(seed):
    """
    设置随机种子以确保实验的可复现性。
    Args:
        seed (int): 随机种子。
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 确保 PyTorch 的卷积操作在每次运行时都是确定性的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False