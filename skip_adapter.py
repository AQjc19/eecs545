"""
SPARK v2 - Latent Bridge Adapter (升级版)

从旧版 Bottleneck Adapter 升级为全维度残差 Adapter:
  旧: h → W_up(d→d/4) → ReLU → W_down(d/4→d) → LayerNorm → z
  新: h → h + W_down(SiLU(W_up(RMSNorm(h))))

关键改进:
  1. 残差连接: Adapter 只学习差异修正 Δh，而非从零重建
  2. 全维度 d→d→d: 保留 100% 信息带宽 (vs 旧版 25%)
  3. RMSNorm 前置: 对齐 Qwen-2.5 的 Pre-Norm 架构
  4. SiLU 激活: 避免 ReLU 的死神经元问题
  5. W_down 零初始化: 第 0 步输出 = h_last（恒等映射冷启动）

参数量: ~25.7M (d²×2 + d×2)，占 7B 模型的 0.37%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipAdapter(nn.Module):
    """
    Latent Bridge Adapter: 将 Transformer 输出空间映射回输入嵌入空间。
    
    通过残差结构，Adapter 在训练初期等价于恒等映射（h_last → h_last），
    随训练逐渐学会最优的空间翻译。

    Args:
        hidden_size: 模型隐层维度 d (e.g., 3584 for Qwen-2.5-7B, 2048 for 1.5B)
    """

    def __init__(self, hidden_size: int, **kwargs):
        # **kwargs 吸收旧版传入的 bottleneck_ratio 等参数，保持接口兼容
        super().__init__()
        self.hidden_size = hidden_size

        # RMSNorm 前置 (对齐 Qwen Pre-Norm 惯例)
        self.norm = nn.RMSNorm(hidden_size)

        # 全维度投影 d → d → d
        self.w_up = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_down = nn.Linear(hidden_size, hidden_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        """
        初始化策略:
          W_up: Kaiming 正态初始化 (适配 SiLU 激活)
          W_down: 全零初始化 → 第 0 步 Δh = 0, output = h_last
          RMSNorm: 默认 γ=1
        """
        nn.init.kaiming_normal_(self.w_up.weight, nonlinearity='linear')
        nn.init.zeros_(self.w_down.weight)

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        """
        前向传播: e_next = h_last + W_down(SiLU(W_up(RMSNorm(h_last))))

        Args:
            h_last: Transformer 最后一层 hidden state
                    shape: (..., hidden_size)

        Returns:
            e_next: 映射后的伪嵌入向量，直接作为下一步 inputs_embeds
                    shape: 与输入相同
        """
        normed = self.norm(h_last)
        delta = self.w_down(F.silu(self.w_up(normed)))
        return h_last + delta

    def get_param_count(self) -> dict:
        """返回参数统计"""
        up_params = self.w_up.weight.numel()
        down_params = self.w_down.weight.numel()
        norm_params = sum(p.numel() for p in self.norm.parameters())
        total = up_params + down_params + norm_params
        return {
            "w_up": up_params,
            "w_down": down_params,
            "norm": norm_params,
            "total": total,
            "total_M": total / 1e6,
        }

    def check_output_norm_ratio(self, h_last: torch.Tensor) -> float:
        """检查输出范数比: ||e_next|| / ||h_last||"""
        with torch.no_grad():
            e_next = self.forward(h_last)
            in_norm = h_last.norm(dim=-1).mean()
            out_norm = e_next.norm(dim=-1).mean()
            return (out_norm / (in_norm + 1e-8)).item()