"""
SPARK v2 - 奖励函数

Phase 1: R_outcome = +1 (correct) / -1 (incorrect)
Advantage: Dr. GRPO (Group-Relative, 无 std 归一化)

Dr. GRPO 论文 (Liu et al., 2025):
  标准 GRPO: A_i = (R_i - mean(R)) / std(R)  ← 有 difficulty bias
  Dr. GRPO:  A_i = R_i - mean(R)               ← 无 bias
"""

import math
from typing import List, Dict, Optional
from dataclasses import dataclass

from config import RewardConfig, KVIGConfig


# ============================================================
# Phase 1: Outcome Reward (+1 / -1)
# ============================================================

def compute_outcome_reward(is_correct: bool) -> float:
    return 1.0 if is_correct else -1.0


def compute_total_reward(
    trajectory: Dict,
    phase: int = 1,
    config: RewardConfig = None,
    **kwargs,
) -> float:
    """Phase 1: 只用 outcome reward"""
    if config is None:
        config = RewardConfig()
    is_correct = trajectory["is_correct"]
    if phase == 1:
        return compute_outcome_reward(is_correct)
    # Phase 2: 扩展（暂不实现）
    return compute_outcome_reward(is_correct)


# ============================================================
# Advantage 计算
# ============================================================

def compute_group_advantages(
    rewards: List[float],
    use_std_norm: bool = False,
    eps: float = 1e-8,
) -> List[float]:
    """
    Group-Relative Advantage (GRPO 核心)

    Dr. GRPO (默认, use_std_norm=False):
        A_i = R_i - mean(R)
        不除以 std → 消除 difficulty bias
        全对/全错 → mean=±1 → A_i=0 → 无梯度信号（正确行为）

    标准 GRPO (use_std_norm=True):
        A_i = (R_i - mean(R)) / (std(R) + eps)
        全对/全错 → std=0 → 返回 0
    """
    n = len(rewards)
    if n == 0:
        return []

    mean_r = sum(rewards) / n

    if not use_std_norm:
        # Dr. GRPO: 不除以 std
        return [r - mean_r for r in rewards]

    # 标准 GRPO: 除以 std
    var_r = sum((r - mean_r) ** 2 for r in rewards) / n
    std_r = var_r ** 0.5
    if std_r < eps:
        return [0.0 for _ in rewards]
    return [(r - mean_r) / (std_r + eps) for r in rewards]