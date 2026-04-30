#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

# 假设你的 config.py 中将 KVIGConfig 更名为了 MCIGConfig，若未更名可兼容处理
from config import KVIGConfig as MCIGConfig, ModelConfig

@dataclass
class MCIGState:
    """
    用于在线自回归生成的增量状态保存 (Incremental State for Online RL)
    """
    h_norm_prev: Optional[torch.Tensor] = None       # 归一化后的前一步隐状态 (\hat{h}_{t-1})
    log_norm_prev: Optional[float] = None            # 前一步隐状态范数的对数 (\log ||h_{t-1}||_2)
    delta_h_norm_prev: Optional[torch.Tensor] = None # 前一步的切向量 (\Delta \hat{h}_{t-1})
    probs_prev: Optional[torch.Tensor] = None        # 前一步的概率分布 (P_{t-1})
    mcig_env_prev: float = 0.0                       # 前一步的动量包络值
    step: int = 0                                    # 当前生成步数

    def is_first_step(self) -> bool:
        return self.step == 0

class MCIGComputer:
    """
    流形因果动力学计算器 (Dense-MCIG Engine)
    实现了真正的 OR 逻辑 (Max-Pooling) 与正交的三维特征融合。
    """
    def __init__(self, mcig_config: MCIGConfig, model_config: ModelConfig):
        self.eps = getattr(mcig_config, "eps", 1e-8)
        self.hidden_size = model_config.hidden_size
        self.decay = 0.8  # 包络衰减系数 γ (与离线实验对齐)
        self.alpha_energy = 2.0  # 能量对齐系数

    @torch.no_grad()
    def compute_step(self, h_t: torch.Tensor, logits_t: torch.Tensor, state: MCIGState, **kwargs) -> Tuple[float, MCIGState]:
        """
        [Phase 2 在线生成专用] 
        增量计算当前 Token 的 Dense-MCIG 分数，用于 Token-level 的奖励调制。
        """
        h_t = h_t.detach().float().squeeze()
        logits_t = logits_t.detach().float().squeeze()
        
        # 1. 基础状态计算
        probs_t = F.softmax(logits_t, dim=-1).clamp(min=1e-10)
        norm_t = h_t.norm(p=2).item()
        log_norm_t = np.log(norm_t + self.eps)
        h_norm_t = F.normalize(h_t, p=2, dim=-1)

        # ====================================================
        # ★ 核心修复：将 state 中的 CPU 张量动态对齐到当前的 GPU
        # ====================================================
        device = h_t.device
        if state.h_norm_prev is not None:
            state.h_norm_prev = state.h_norm_prev.to(device)
        if state.delta_h_norm_prev is not None:
            state.delta_h_norm_prev = state.delta_h_norm_prev.to(device)
        if state.probs_prev is not None:
            state.probs_prev = state.probs_prev.to(device)

        # 处理起始步：没有前置状态时
        if state.is_first_step() or state.h_norm_prev is None or state.probs_prev is None:
            mcig_raw = 0.0
            new_delta_h_norm = None
        else:
            # ── ① C_t: 测地线曲率 (方向变化) ──
            delta_h_norm_t = h_norm_t - state.h_norm_prev
            if state.delta_h_norm_prev is None:
                # 生成第一个 Response Token 时，退化为绝对角度差
                cos_sim = F.cosine_similarity(h_norm_t.unsqueeze(0), state.h_norm_prev.unsqueeze(0), eps=self.eps).item()
                C_t = max(0.0, 1.0 - cos_sim)
            else:
                # 正常的切向量角度差
                cos_sim = F.cosine_similarity(delta_h_norm_t.unsqueeze(0), state.delta_h_norm_prev.unsqueeze(0), eps=self.eps).item()
                C_t = max(0.0, 1.0 - cos_sim)
            new_delta_h_norm = delta_h_norm_t.cpu()

            # ── ② J_t: 对称信息激波 (JSD) ──
            M = 0.5 * (probs_t + state.probs_prev)
            kl_curr = (probs_t * (probs_t.log() - M.log())).sum().item()
            kl_prev = (state.probs_prev * (state.probs_prev.log() - M.log())).sum().item()
            J_t = max(0.0, 0.5 * kl_curr + 0.5 * kl_prev)

            # ── ③ E_t: 绝对对数能量激波 (载荷突变) ──
            E_t = self.alpha_energy * abs(log_norm_t - state.log_norm_prev)

            # ── ④ Max-Pooling OR 逻辑融合 ──
            mcig_raw = max(C_t, J_t, E_t)

        # 动量包络平滑
        mcig_env = max(mcig_raw, state.mcig_env_prev * self.decay)
        
        # 更新状态
        new_state = MCIGState(
            h_norm_prev=h_norm_t.cpu(),
            log_norm_prev=log_norm_t,
            delta_h_norm_prev=new_delta_h_norm,
            probs_prev=probs_t.cpu(),
            mcig_env_prev=mcig_env,
            step=state.step + 1
        )
        
        return mcig_env, new_state

    @torch.no_grad()
    def compute_trajectory_from_model(self, model, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, prompt_length: int = 0, **kwargs) -> Dict:
        """
        [Phase 1.5 离线提取专用]
        批量计算单条完整轨迹的 Dense-MCIG。严格对齐 `compare_ada.py` 中的数学公式。
        """
        device = next(model.parameters()).device
        if input_ids.dim() == 1: input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1: attention_mask = attention_mask.unsqueeze(0)
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        resp_len = input_ids.shape[1] - prompt_length
        
        if resp_len <= 1:
            empty = [0.0] * max(0, resp_len)
            return {"mcig_values": empty, "mean_mcig": 0.0, "std_mcig": 0.0}

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
        last_hidden = outputs.hidden_states[-1][0].float()
        logits = outputs.logits[0].float()
        
        h_resp, logits_resp = last_hidden[prompt_length:], logits[prompt_length:]
        h_prev_first, logits_prev_first = last_hidden[prompt_length - 1], logits[prompt_length - 1]
        
        h_all = torch.cat([h_prev_first.unsqueeze(0), h_resp], dim=0)
        all_logits = torch.cat([logits_prev_first.unsqueeze(0), logits_resp], dim=0)

        # 计算概率分布
        probs = F.softmax(all_logits, dim=-1).clamp(min=1e-10)
        
        # 初始化基础张量
        C_vec = np.zeros(resp_len, dtype=np.float32)
        J_vec = np.zeros(resp_len, dtype=np.float32)
        E_vec = np.zeros(resp_len, dtype=np.float32)
        
        if resp_len >= 2:
            # 1. 计算 C (Curvature)
            h_all_norm = F.normalize(h_all, p=2, dim=-1)
            delta_h_norm = h_all_norm[1:] - h_all_norm[:-1]  
            cos_t0 = F.cosine_similarity(h_all_norm[1].unsqueeze(0), h_all_norm[0].unsqueeze(0), eps=self.eps).item()
            C_vec[0] = max(0.0, 1.0 - cos_t0)
            C_vec[1:] = (1.0 - F.cosine_similarity(delta_h_norm[1:], delta_h_norm[:-1], dim=-1, eps=self.eps)).cpu().numpy()
            C_vec = np.maximum(0.0, C_vec) # 兜底消除负数
            
            # 2. 计算 J (JSD Causal Shock)
            P_prev = probs[:-1]
            P_curr = probs[1:]
            M = 0.5 * (P_prev + P_curr)
            kl_prev_M = torch.sum(P_prev * (torch.log(P_prev) - torch.log(M)), dim=-1)
            kl_curr_M = torch.sum(P_curr * (torch.log(P_curr) - torch.log(M)), dim=-1)
            J_vec = (0.5 * (kl_prev_M + kl_curr_M)).cpu().numpy()
            J_vec = np.maximum(0.0, J_vec)
            
            # 3. 计算 E (Absolute Log Energy Spike)
            h_norms = h_all.norm(dim=-1).cpu().numpy()
            E_vec = np.abs(np.log(h_norms[1:] + self.eps) - np.log(h_norms[:-1] + self.eps)) * self.alpha_energy

        # 4. Max-Pooling (真正的 OR 逻辑融合)
        sig_mcig = np.maximum(C_vec, np.maximum(J_vec, E_vec))

        # 动量包络蔓延
        mcig_env = np.zeros_like(sig_mcig)
        if len(sig_mcig) > 0:
            mcig_env[0] = sig_mcig[0]
            for t in range(1, len(sig_mcig)):
                mcig_env[t] = max(sig_mcig[t], mcig_env[t-1] * self.decay)

        del outputs, logits, last_hidden, h_resp, logits_resp, h_all, all_logits, probs
        if resp_len >= 2: 
            del h_all_norm, delta_h_norm, P_prev, P_curr, M
        torch.cuda.empty_cache()

        mcig_env = np.nan_to_num(mcig_env, nan=0.0, posinf=0.0, neginf=0.0)
        mcig_values = mcig_env.tolist()
        
        return {
            "mcig_values": mcig_values,
            "mean_mcig": float(np.mean(mcig_values)) if mcig_values else 0.0,
            "std_mcig": float(np.std(mcig_values)) if mcig_values else 0.0,
            # 兼容需要原命名的数据管线 (避免改动过多外部依赖)
            "kvig_values": mcig_values, 
        }

def compute_mcig_batch(mcig_computer, model, batch_input_ids, batch_attention_masks, prompt_lengths, **kwargs):
    return [mcig_computer.compute_trajectory_from_model(model, input_ids=ids, attention_mask=mask, prompt_length=pl) 
            for ids, mask, pl in zip(batch_input_ids, batch_attention_masks, prompt_lengths)]