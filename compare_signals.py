#!/usr/bin/env python3
"""
SPARK v2 - 多信号对比与流形动力学验证 (Multi-Signal Comparison & Manifold Dynamics Validation)

包含指标:
  1. Omega_Dyn           (流形动力学积分器)
  2. IV_KL               (一阶信息速度/KL散度)
  3. Old_MCIG            (带有范数和隐层余弦的旧版本)
  4. cossim              (隐状态余弦距离)
  5. policy_entropy      (预测概率香农熵)
  6. causal_entropy_drop (因果熵降, max(0, H_{t-1} - H_t))
  7. attn_entropy        (最后一层注意力聚焦度, 1 - 归一化熵)
"""

import os
import sys
import argparse
import logging
import json
import glob
import time
import gc
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("manifold_compare")

# ============================================================
# 第一部分: 数据加载 (维持不变)
# ============================================================
def load_raw_chunks(raw_chunks_dir: str, max_trajectories: int = -1) -> Tuple[List[Dict], List[Dict]]:
    chunk_files = sorted(glob.glob(os.path.join(raw_chunks_dir, "chunk_*.pt")))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {raw_chunks_dir}")
    
    all_trajectories = []
    all_kvig_results = []
    
    for cf in chunk_files:
        try:
            data = torch.load(cf, map_location="cpu", weights_only=False)
            all_trajectories.extend(data["trajectories"])
            all_kvig_results.extend(data["kvig_results"])
        except Exception:
            continue
            
    if max_trajectories > 0 and len(all_trajectories) > max_trajectories:
        np.random.seed(42)
        indices = np.random.choice(len(all_trajectories), max_trajectories, replace=False).tolist()
        all_trajectories = [all_trajectories[i] for i in indices]
        all_kvig_results = [all_kvig_results[i] for i in indices]
    
    logger.info(f"Loaded {len(all_trajectories)} trajectories.")
    return all_trajectories, all_kvig_results


# ============================================================
# 第二部分: 流形动力学信号计算器 (包含完整 7 大指标)
# ============================================================
class ManifoldSignalComputer:
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def compute_all_signals(
        self,
        full_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_length: int,
    ) -> Dict[str, np.ndarray]:
        
        input_ids = full_ids.unsqueeze(0).to(self.device)
        attn_mask = attention_mask.unsqueeze(0).to(self.device)
        
        resp_len = full_ids.shape[0] - prompt_length
        if resp_len <= 2:
            return defaultdict(lambda: np.zeros(max(resp_len, 1)))

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
        )
        
        logits = outputs.logits[0]  
        last_hidden = outputs.hidden_states[-1][0]  
        last_attn = outputs.attentions[-1][0] 

        # ---------------------------------------------------------
        # 基于原版第一版切片的对齐逻辑 (严格提取 N+2 窗口)
        # ---------------------------------------------------------
        slice_start = max(0, prompt_length - 2)
        z = logits[slice_start:]  # Shape: (N+2, V)
        
        P = F.softmax(z, dim=-1)
        log_P = F.log_softmax(z, dim=-1)
        
        # 1. 核心动力学积分器: Omega_Dyn
        delta_z = z[1:] - z[:-1]  # Shape: (N+1, V)
        cos_sim = F.cosine_similarity(delta_z[1:], delta_z[:-1], dim=-1) # (N,)
        K_t = 1.0 - cos_sim
        
        v_t = torch.sum(P[1:] * (log_P[1:] - log_P[:-1]), dim=-1) # (N+1,)
        A_t = F.relu(v_t[1:] - v_t[:-1]) # (N,)
        C_t, _ = torch.max(P[2:], dim=-1) # (N,)
        I_t = K_t * A_t * C_t # (N,)
        
        Omega_t = np.zeros(resp_len, dtype=np.float32)
        I_np = I_t.cpu().numpy()
        C_np = C_t.cpu().numpy()
        
        omega_prev = 0.0
        for t in range(resp_len):
            omega_curr = I_np[t] + C_np[t] * omega_prev
            Omega_t[t] = omega_curr
            omega_prev = omega_curr

        # 2. IV_KL
        IV_KL = v_t[1:].cpu().numpy()
        
        # 3. Old_MCIG
        h_all = last_hidden[slice_start+1:] # Shape: (N+1, d)
        h_norms = h_all.float().norm(dim=-1).cpu().numpy()
        energy_diff = np.abs(np.log(h_norms[1:] + 1e-8) - np.log(h_norms[:-1] + 1e-8)) * 2.0
        
        causal_drop_val = F.relu(-torch.sum(P[1:-1] * log_P[1:-1], dim=-1) - 
                             -torch.sum(P[2:] * log_P[2:], dim=-1)).cpu().numpy()
        
        h_normed = F.normalize(h_all, p=2, dim=-1)
        h_delta_normed = h_normed[1:] - h_normed[:-1]
        h_K_t = 1.0 - F.cosine_similarity(h_delta_normed[1:], h_delta_normed[:-1], dim=-1)
        if h_K_t.shape[0] < resp_len:
            pad_len = resp_len - h_K_t.shape[0]
            h_K_t = F.pad(h_K_t, (pad_len, 0), value=0.0)
            
        old_MCIG = np.maximum(h_K_t.cpu().numpy(), energy_diff) * causal_drop_val * h_norms[1:]

        # ---------------------------------------------------------
        # 新增基线计算 (严格基于第一版的张量框架)
        # ---------------------------------------------------------
        h_slice = last_hidden[slice_start:] # (N+2, d)
        entropies = -torch.sum(P * log_P, dim=-1) # (N+2,)

        # 4. cossim (当前隐藏状态与上一步隐藏状态的余弦距离)
        cossim_arr = 1.0 - F.cosine_similarity(h_slice[2:], h_slice[1:-1], dim=-1)

        # 5. policy_entropy (模型预测当前 token 时的后验分布熵)
        policy_entropy_arr = entropies[1:-1]

        # 6. causal_entropy_drop (预测 token t 前后的熵差)
        causal_entropy_drop_arr = F.relu(entropies[1:-1] - entropies[2:])

        # 7. attn_entropy (基于查询对齐的聚焦度)
        attn_entropies_arr = np.zeros(resp_len, dtype=np.float32)
        for i in range(resp_len):
            p = prompt_length - 1 + i
            if p < last_attn.shape[-1]:
                attn = last_attn[:, p, :p+1].float().clamp(min=1e-10)
                H = -torch.sum(attn * torch.log(attn), dim=-1).mean().item()
                max_H = np.log(p + 1) if p > 0 else 1.0
                attn_entropies_arr[i] = max(0.0, min(1.0, 1.0 - (H / max_H)))

        # 内存释放
        del outputs, logits, last_hidden, last_attn, z, P, log_P, delta_z
        torch.cuda.empty_cache()

        return {
            "Omega_Dyn": np.nan_to_num(Omega_t),
            "IV_KL": np.nan_to_num(IV_KL),
            "Old_MCIG": np.nan_to_num(old_MCIG),
            "cossim": np.nan_to_num(cossim_arr.cpu().numpy()),
            "policy_entropy": np.nan_to_num(policy_entropy_arr.cpu().numpy()),
            "causal_entropy_drop": np.nan_to_num(causal_entropy_drop_arr.cpu().numpy()),
            "attn_entropy": np.nan_to_num(attn_entropies_arr)
        }


# ============================================================
# 第三部分: 零人工标注评测器 (Zero-Annotation Validation)
# ============================================================
class AutomatedManifoldDiagnostics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def validate_signals(self, signals_dict: Dict[str, List[np.ndarray]], trajectories: List[Dict]):
        results = {}
        for sig_name, sig_list in signals_dict.items():
            
            task_a_y_true, task_a_y_scores = [], []
            task_b_preservations = []
            task_c_percentiles = []
            
            for traj, sig in zip(trajectories, sig_list):
                if len(sig) < 5: continue
                resp_ids = traj["full_ids"][traj["prompt_length"]:]
                tokens = self.tokenizer.convert_ids_to_tokens(resp_ids.tolist())
                if len(tokens) != len(sig): continue
                
                # ------ Task A: 双域分离度测试 ------
                for t, tok in enumerate(tokens):
                    clean_tok = tok.replace('Ġ', '').replace('Ċ', '').strip()
                    if not clean_tok: continue
                    
                    if re.search(r'[\d=+\-*/\\]', clean_tok):
                        task_a_y_true.append(1)
                        task_a_y_scores.append(-sig[t] if sig_name == "policy_entropy" else sig[t])
                    elif re.match(r'^[a-zA-Z]+$', clean_tok) and len(clean_tok) > 1:
                        task_a_y_true.append(0)
                        task_a_y_scores.append(-sig[t] if sig_name == "policy_entropy" else sig[t])
                        
                # ------ Task B: 多 Token 实体保全测试 ------
                entity_start = -1
                for t, tok in enumerate(tokens):
                    is_digit_part = any(c.isdigit() for c in tok)
                    if is_digit_part:
                        if entity_start == -1: entity_start = t
                    else:
                        if entity_start != -1 and t - entity_start >= 2:
                            start_energy = sig[entity_start]
                            if start_energy > 1e-4: 
                                min_follow_energy = np.min(sig[entity_start+1:t])
                                preservation_ratio = min_follow_energy / start_energy
                                task_b_preservations.append(preservation_ratio)
                        entity_start = -1

                # ------ Task C: 全局极值锁定测试 ------
                boxed_idx = -1
                for t, tok in enumerate(tokens):
                    if 'boxed' in tok:
                        boxed_idx = t
                        break
                
                if boxed_idx != -1 and boxed_idx < len(sig):
                    end_idx = min(len(sig), boxed_idx + 4)
                    boxed_max = np.max(sig[boxed_idx:end_idx])
                    
                    if sig_name == "policy_entropy":
                        percentile = scipy_percentileofscore(-sig, -boxed_max)
                    else:
                        percentile = scipy_percentileofscore(sig, boxed_max)
                    task_c_percentiles.append(percentile)
            
            auc = roc_auc_score(task_a_y_true, task_a_y_scores) if task_a_y_true else 0.5
            preservation_success_rate = np.mean([1 if r > 0.5 else 0 for r in task_b_preservations]) if task_b_preservations else 0.0
            top5_hit_rate = np.mean([1 if p >= 95.0 else 0 for p in task_c_percentiles]) if task_c_percentiles else 0.0

            results[sig_name] = {
                "AUC": auc,
                "Entity_Preservation": preservation_success_rate,
                "Boxed_Top5_Hit": top5_hit_rate
            }
            
        return results

def scipy_percentileofscore(a, score):
    n = len(a)
    if n == 0: return 0.0
    return (np.count_nonzero(a < score) + 0.5 * np.count_nonzero(a == score)) / n * 100


# ============================================================
# 第四部分: 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--raw_chunks_dir", type=str, required=True)
    parser.add_argument("--max_trajectories", type=int, default=100)
    args = parser.parse_args()
    
    trajectories, _ = load_raw_chunks(args.raw_chunks_dir, max_trajectories=args.max_trajectories)
    
    sys.path.insert(0, os.path.dirname(os.path.dirname(args.checkpoint)))
    from config import get_config
    from model_utils import setup_model_for_phase1
    
    config = get_config()
    model, tokenizer, _, _ = setup_model_for_phase1(config)
    
    computer = ManifoldSignalComputer(model, tokenizer)
    
    all_signals = defaultdict(list)
    logger.info("Computing Manifold Dynamics and Baseline Signals...")
    for i, traj in enumerate(trajectories):
        sigs = computer.compute_all_signals(
            traj["full_ids"], traj["full_attention_mask"], traj["prompt_length"]
        )
        for k, v in sigs.items():
            all_signals[k].append(v)
            
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
            
    logger.info("\n" + "="*85)
    logger.info("Running Zero-Annotation Mathematical Verification")
    logger.info("="*85)
    
    diagnostics = AutomatedManifoldDiagnostics(tokenizer)
    final_results = diagnostics.validate_signals(all_signals, trajectories)
    
    print("\n" + "="*90)
    print(f"{'Signal Metric':<20} | {'Domain AUC (Task A) ↑':<22} | {'Entity Preserve (B) ↑':<22} | {'Boxed Top5% (C) ↑':<18}")
    print("-" * 90)
    
    sorted_sigs = sorted(final_results.items(), key=lambda x: -x[1]['AUC'])
    for sig_name, res in sorted_sigs:
        print(f"{sig_name:<20} | {res['AUC']:<22.4f} | {res['Entity_Preservation']:<22.1%} | {res['Boxed_Top5_Hit']:<18.1%}")
    print("="*90)
    
if __name__ == "__main__":
    main()