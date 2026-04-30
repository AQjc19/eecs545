"""
SPARK v2 - Phase 1.5 SFT 数据构造
反事实双重门控框架 (Counterfactual Dual-Gating Framework)

两阶段流程:
  Stage 1: 启发式双指标门控 (MCIG proposer + Attention Entropy veto)
  Stage 2: 反事实 Teacher-Forced 验证 (Causal Log-Likelihood Drop)

用法:
    python build_sft_data.py \
        --checkpoint ./checkpoints/phase1/best \
        --signal_data ./signal_comparison/raw_signal_values.pt \
        --calibration_data ./calibration/calibration_raw_data.pt \
        --output_dir ./phase15_sft_data \
        --epsilon 0.1 \
        --max_skip_ratio 0.15 \
        --min_segment_len 3

    # 快速测试 (跳过反事实验证)
    python build_sft_data.py \
        --checkpoint ./checkpoints/phase1/best \
        --calibration_data ./calibration/calibration_raw_data.pt \
        --output_dir ./phase15_sft_data_test \
        --skip_counterfactual \
        --max_trajectories 100
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn.functional as F
import numpy as np
import json
import gc
import glob
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("build_sft")


# ============================================================
# MCIG 计算 (如果没有预计算值)
# ============================================================

@torch.no_grad()
def compute_mcig_for_trajectory(
    model, full_ids, attention_mask, prompt_length, alpha_energy=2.0,
):
    """
    计算单条轨迹的 Dense-MCIG 信号。
    Score_t = max(C_t, J_t, E_t)
    """
    device = next(model.parameters()).device
    input_ids = full_ids.unsqueeze(0).to(device) if full_ids.dim() == 1 else full_ids.to(device)
    attn_mask = attention_mask.unsqueeze(0).to(device) if attention_mask.dim() == 1 else attention_mask.to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        output_hidden_states=True,
        use_cache=False,
    )

    logits = outputs.logits[0]                    # (seq_len, V)
    last_hidden = outputs.hidden_states[-1][0]    # (seq_len, d)

    resp_hidden = last_hidden[prompt_length:]     # (T, d)
    resp_logits = logits[prompt_length - 1:-1]    # (T, V) - 预测 response token 的 logits
    T = resp_hidden.shape[0]

    if T <= 2:
        return np.zeros(max(T, 0), dtype=np.float32)

    # ── C_t: 测地线曲率 ──
    h_norm = F.normalize(resp_hidden.float(), p=2, dim=-1)
    delta_h = h_norm[1:] - h_norm[:-1]                    # (T-1, d) 切向量
    delta_h_norm = F.normalize(delta_h, p=2, dim=-1)
    cos_delta = torch.sum(delta_h_norm[1:] * delta_h_norm[:-1], dim=-1)  # (T-3,)
    C = (1.0 - cos_delta).cpu().numpy()
    C_pad = np.concatenate([[np.median(C)], [np.median(C)], C])  # pad 前两位

    # ── J_t: JSD ──
    probs = F.softmax(resp_logits.float(), dim=-1)  # (T, V)
    eps = 1e-10
    probs = probs.clamp(min=eps)
    P_curr = probs[1:]   # (T-1, V)
    P_prev = probs[:-1]  # (T-1, V)
    M = 0.5 * (P_curr + P_prev)
    kl_pm = (P_curr * (P_curr / M).log()).sum(dim=-1)
    kl_qm = (P_prev * (P_prev / M).log()).sum(dim=-1)
    jsd = (0.5 * kl_pm + 0.5 * kl_qm).cpu().numpy()
    J_pad = np.concatenate([[np.median(jsd)], jsd])

    # ── E_t: 对数能量激波 ──
    norms = resp_hidden.float().norm(dim=-1)  # (T,)
    log_norms = torch.log(norms.clamp(min=eps))
    log_diff = torch.abs(log_norms[1:] - log_norms[:-1])  # (T-1,)
    E = (alpha_energy * log_diff).cpu().numpy()
    E_pad = np.concatenate([[np.median(E)], E])

    # ── Max-pooling ──
    min_len = min(len(C_pad), len(J_pad), len(E_pad), T)
    score = np.maximum(np.maximum(C_pad[:min_len], J_pad[:min_len]), E_pad[:min_len])

    # Clean NaN
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    del outputs
    return score.astype(np.float32)


# ============================================================
# Stage 1: 启发式双指标门控
# ============================================================

def find_candidate_segments(
    mcig_values: np.ndarray,
    #attn_entropy_values: Optional[np.ndarray],
    min_segment_len: int = 3,
    max_segment_len: int = 30,
    max_skip_ratio: float = 0.15,
    mcig_threshold_factor: float = 0.5,  # μ - factor * σ
    attn_veto_percentile: float = 70.0,
) -> List[Tuple[int, int]]:
    """
    启发式双指标门控，找出候选 skip 段。

    Args:
        mcig_values: (T,) 每个 token 的 MCIG 值
        attn_entropy_values: (T,) 每个 token 的 attention entropy（可选）
        min_segment_len: 最短候选段长度
        max_segment_len: 最长候选段长度
        max_skip_ratio: 每条轨迹最大 skip 比例
        mcig_threshold_factor: MCIG 阈值 = μ - factor * σ
        attn_veto_percentile: attention entropy 保护百分位

    Returns:
        segments: [(start, end), ...] 每个段的起止索引（inclusive）
    """
    T = len(mcig_values)
    if T < min_segment_len + 4:  # 太短的轨迹不处理
        return []

    # ── Step 1: MCIG 候选 ──
    mu = np.mean(mcig_values)
    sigma = np.std(mcig_values)
    threshold = mu - mcig_threshold_factor * sigma
    recall_mask = mcig_values < threshold  # True = 候选冗余


    """
    # ── Step 2: Attention Entropy 拦截 ──
    if attn_entropy_values is not None and len(attn_entropy_values) == T:
        veto_threshold = np.percentile(attn_entropy_values, attn_veto_percentile)
        veto_mask = attn_entropy_values > veto_threshold  # True = 被保护
        recall_mask = recall_mask & (~veto_mask)
    """
    # ── Step 3: 提取连续段 ──
    segments = []
    start = None
    for t in range(T):
        if recall_mask[t]:
            if start is None:
                start = t
        else:
            if start is not None:
                end = t - 1
                seg_len = end - start + 1
                if min_segment_len <= seg_len <= max_segment_len:
                    segments.append((start, end))
                start = None
    # 处理尾部
    if start is not None:
        end = T - 1
        seg_len = end - start + 1
        if min_segment_len <= seg_len <= max_segment_len:
            segments.append((start, end))

    # ── Step 4: 控制总 skip 比例 ──
    total_skip = sum(e - s + 1 for s, e in segments)
    max_skip_tokens = int(T * max_skip_ratio)

    if total_skip > max_skip_tokens:
        # 按 MCIG 均值排序，优先保留最冗余（均值最低）的段
        seg_scores = []
        for s, e in segments:
            seg_mean = np.mean(mcig_values[s:e+1])
            seg_scores.append((seg_mean, s, e))
        seg_scores.sort()  # 升序，最冗余在前

        selected = []
        cumulative = 0
        for score, s, e in seg_scores:
            seg_len = e - s + 1
            if cumulative + seg_len <= max_skip_tokens:
                selected.append((s, e))
                cumulative += seg_len
        segments = sorted(selected)

    # ── Step 5: 安全过滤 ──
    # 不在最后 10 个 token 中 skip（保护答案区域）
    segments = [(s, e) for s, e in segments if e < T - 10]

    return segments


# ============================================================
# Stage 2: 反事实 Teacher-Forced 验证
# ============================================================

@torch.no_grad()
def counterfactual_verify_segment(
    model,
    full_ids: torch.Tensor,       
    attention_mask: torch.Tensor,
    segment_start: int,           
    segment_end: int,             
    prompt_length: int,
    epsilon: float = 0.1,
    n_check_tokens: int = 20,     # ★ 改为只检查最后 20 个 token
) -> Tuple[bool, float]:
    
    device = next(model.parameters()).device
    seq_len = full_ids.shape[0]

    abs_start = prompt_length + segment_start
    abs_end = prompt_length + segment_end 

    # ★ 核心破局点：避开语法断层，直接检查整条轨迹最末尾的答案区域
    check_start = max(abs_end + 1, seq_len - n_check_tokens)
    check_end = seq_len

    if check_end <= check_start:
        return False, -999.0

    # ── 1. 完整序列的 log-likelihood ──
    full_input = full_ids.unsqueeze(0).to(device)
    full_mask = attention_mask.unsqueeze(0).to(device)

    full_out = model(input_ids=full_input, attention_mask=full_mask, use_cache=False)
    full_logits = full_out.logits[0]

    full_ll = 0.0
    n_tokens = 0
    for t in range(check_start, check_end):
        target = full_ids[t].item()
        log_prob = F.log_softmax(full_logits[t - 1].float(), dim=-1)
        full_ll += log_prob[target].item()
        n_tokens += 1
    full_ll_avg = full_ll / max(n_tokens, 1)

    # ── 2. 反事实序列（注意力擦除法）的 log-likelihood ──
    # 不改变绝对位置（保护 RoPE），只让未来的 Token "看不见"废话段
    cf_mask_input = full_mask.clone()
    cf_mask_input[0, abs_start:abs_end + 1] = 0

    cf_out = model(input_ids=full_input, attention_mask=cf_mask_input, use_cache=False)
    cf_logits = cf_out.logits[0]

    cf_ll = 0.0
    for t in range(check_start, check_end):
        target = full_ids[t].item()
        log_prob = F.log_softmax(cf_logits[t - 1].float(), dim=-1)
        cf_ll += log_prob[target].item()

    cf_ll_avg = cf_ll / max(n_tokens, 1)

    # ── 3. 判定 ──
    delta_ll = cf_ll_avg - full_ll_avg
    passed = delta_ll >= -epsilon

    del full_out, cf_out
    return passed, float(delta_ll)


# ============================================================
# SFT 数据组装
# ============================================================

def assemble_sft_trajectory(
    full_ids: torch.Tensor,
    prompt_length: int,
    verified_segments: List[Tuple[int, int]],  # response 内相对位置
    skip_token_id: int,
) -> torch.Tensor:
    """
    将验证通过的候选段替换为 <SKIP>，组装 SFT 训练序列。

    对每个验证通过的段 [start, end]：
    - 位置 start 替换为 <SKIP> token
    - 位置 start+1 到 end 的 token 被删除

    Returns:
        sft_ids: 组装后的 token ids (1D tensor)
    """
    if len(verified_segments) == 0:
        return full_ids.clone()

    # 按位置排序
    segments = sorted(verified_segments)

    # 构建新序列
    parts = []

    # Prompt 部分完整保留
    parts.append(full_ids[:prompt_length])

    # Response 部分逐段处理
    prev_end = 0  # response 内的位置
    for seg_start, seg_end in segments:
        if seg_start > prev_end:
            abs_start = prompt_length + prev_end
            abs_end = prompt_length + seg_start
            parts.append(full_ids[abs_start:abs_end])

        # ★ 动态压缩逻辑：根据废话长度插入 <SKIP>，上限为 K_max=8
        # 既实现了压缩，又给 Skip Adapter 留下了充足的推理步数
        seg_len = seg_end - seg_start + 1
        n_skips = max(1, min(8, seg_len // 4))  # 假设压缩比为 4:1
        
        skip_tensor = torch.full((n_skips,), skip_token_id, dtype=full_ids.dtype)
        parts.append(skip_tensor)

        prev_end = seg_end + 1

    # 最后一段之后的 token
    resp_len = full_ids.shape[0] - prompt_length
    if prev_end < resp_len:
        abs_start = prompt_length + prev_end
        parts.append(full_ids[abs_start:])

    return torch.cat(parts)


# ============================================================
# 数据加载
# ============================================================

def load_trajectories(
    calibration_data_path: Optional[str] = None,
    raw_chunks_dir: Optional[str] = None,
    signal_data_path: Optional[str] = None,
    max_trajectories: Optional[int] = None,
) -> Tuple[List[Dict], Optional[Dict]]:
    """加载轨迹数据和预计算信号"""

    trajectories = []

    if calibration_data_path and os.path.exists(calibration_data_path):
        logger.info(f"Loading from {calibration_data_path}")
        data = torch.load(calibration_data_path, map_location="cpu", weights_only=False)
        trajectories = data
    elif raw_chunks_dir and os.path.exists(raw_chunks_dir):
        logger.info(f"Loading from chunks in {raw_chunks_dir}")
        chunk_files = sorted(glob.glob(os.path.join(raw_chunks_dir, "chunk_*.pt")))
        for cf in chunk_files:
            try:
                chunk = torch.load(cf, map_location="cpu", weights_only=False)
                for traj, kvig_res in zip(chunk["trajectories"], chunk["kvig_results"]):
                    merged = {**traj}
                    merged["kvig_values"] = kvig_res.get("kvig_values", [])
                    trajectories.append(merged)
            except Exception as e:
                logger.warning(f"Failed to load {cf}: {e}")

    # 只保留正确轨迹
    correct = [t for t in trajectories if t.get("is_correct", False)]
    logger.info(f"Total trajectories: {len(trajectories)}, correct: {len(correct)}")

    if max_trajectories and len(correct) > max_trajectories:
        import random
        random.seed(42)
        correct = random.sample(correct, max_trajectories)

    # 加载预计算信号
    signal_data = None
    if signal_data_path and os.path.exists(signal_data_path):
        logger.info(f"Loading pre-computed signals from {signal_data_path}")
        signal_data = torch.load(signal_data_path, map_location="cpu", weights_only=False)

    return correct, signal_data


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 1.5 SFT Data Construction")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--calibration_data", type=str, default=None)
    parser.add_argument("--raw_chunks_dir", type=str, default=None)
    parser.add_argument("--signal_data", type=str, default=None,
                        help="Pre-computed signals (raw_signal_values.pt)")
    parser.add_argument("--output_dir", type=str, default="./phase15_sft_data")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Counterfactual log-likelihood tolerance")
    parser.add_argument("--max_skip_ratio", type=float, default=0.15)
    parser.add_argument("--min_segment_len", type=int, default=3)
    parser.add_argument("--max_segment_len", type=int, default=30)
    parser.add_argument("--mcig_threshold_factor", type=float, default=0.5,
                        help="MCIG threshold = mu - factor * sigma")
    parser.add_argument("--n_check_tokens", type=int, default=30,
                        help="Tokens to check after deleted segment")
    parser.add_argument("--skip_counterfactual", action="store_true",
                        help="Skip counterfactual verification (for testing)")
    parser.add_argument("--max_trajectories", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    logger.info("=" * 70)
    logger.info("SPARK v2 - Phase 1.5 SFT Data Construction")
    logger.info("  Counterfactual Dual-Gating Framework")
    logger.info("=" * 70)

    # ── 1. 加载模型 ──
    logger.info("Step 1: Loading model...")
    from config import get_config
    config = get_config()
    from model_utils import load_checkpoint, setup_model_for_phase1

    if os.path.exists(os.path.join(args.checkpoint, "config.json")):
        model, tokenizer, skip_token_id, skip_adapter, _ = load_checkpoint(
            args.checkpoint, config
        )
    else:
        model, tokenizer, skip_token_id, skip_adapter = setup_model_for_phase1(config)
    model.eval()

    # 确保 skip_token_id 存在
    if skip_token_id is None:
        skip_token_id = len(tokenizer)  # 假设尚未添加
        logger.info(f"  Using skip_token_id = {skip_token_id}")
    logger.info(f"  Model loaded, skip_token_id = {skip_token_id}")

    # ── 2. 加载数据 ──
    logger.info("Step 2: Loading correct trajectories...")
    trajectories, signal_data = load_trajectories(
        calibration_data_path=args.calibration_data,
        raw_chunks_dir=args.raw_chunks_dir,
        signal_data_path=args.signal_data,
        max_trajectories=args.max_trajectories,
    )
    logger.info(f"  Loaded {len(trajectories)} correct trajectories")

    # ── 3. Stage 1: 启发式双指标门控 ──
    logger.info("Step 3: Stage 1 - Heuristic Dual-Gating...")

    all_candidates = []   # (traj_idx, segments)
    stage1_stats = {"total_segments": 0, "total_tokens_proposed": 0}

    for i, traj in enumerate(trajectories):
        full_ids = traj["full_ids"]
        prompt_length = traj["prompt_length"]
        resp_len = full_ids.shape[0] - prompt_length

        # 获取 MCIG 值
        mcig_vals = None
        if "mcig_values" in traj:
            mcig_vals = np.array(traj["mcig_values"], dtype=np.float32)
        elif signal_data and "mcig" in signal_data.get("signals", {}):
            # 从预计算信号中查找（需要索引对齐）
            idx = signal_data.get("trajectory_indices", {}).get(i, i)
            if idx < len(signal_data["signals"]["mcig"]):
                mcig_vals = signal_data["signals"]["mcig"][idx]

        # 如果没有预计算 MCIG，在线计算
        if mcig_vals is None or len(mcig_vals) == 0:
            attention_mask = traj.get("full_attention_mask", torch.ones_like(full_ids))
            mcig_vals = compute_mcig_for_trajectory(
                model, full_ids, attention_mask, prompt_length
            )
            torch.cuda.empty_cache()

        # 获取 attention entropy（如果有的话）
        """
        attn_entropy = None
        if "attn_entropy_values" in traj:
            attn_entropy = np.array(traj["attn_entropy_values"], dtype=np.float32)
        elif signal_data and "attn_entropy" in signal_data.get("signals", {}):
            idx = signal_data.get("trajectory_indices", {}).get(i, i)
            if idx < len(signal_data["signals"]["attn_entropy"]):
                attn_entropy = signal_data["signals"]["attn_entropy"][idx]
        """

        # 确保长度一致
        min_len = min(len(mcig_vals), resp_len)
        mcig_vals = mcig_vals[:min_len]

        # 找候选段
        segments = find_candidate_segments(
            mcig_values=mcig_vals,
            #attn_entropy_values=attn_entropy[:min_len] if attn_entropy is not None else None,
            min_segment_len=args.min_segment_len,
            max_segment_len=args.max_segment_len,
            max_skip_ratio=args.max_skip_ratio,
            mcig_threshold_factor=args.mcig_threshold_factor,
        )

        if segments:
            all_candidates.append((i, segments))
            stage1_stats["total_segments"] += len(segments)
            stage1_stats["total_tokens_proposed"] += sum(e - s + 1 for s, e in segments)

        if (i + 1) % 200 == 0:
            logger.info(f"  Processed {i + 1}/{len(trajectories)} trajectories")

    logger.info(f"  Stage 1 complete:")
    logger.info(f"    Trajectories with candidates: {len(all_candidates)}/{len(trajectories)}")
    logger.info(f"    Total candidate segments: {stage1_stats['total_segments']}")
    logger.info(f"    Total tokens proposed for skip: {stage1_stats['total_tokens_proposed']}")

    # ── 4. Stage 2: 反事实验证 ──
    if args.skip_counterfactual:
        logger.info("Step 4: SKIPPING counterfactual verification (--skip_counterfactual)")
        verified_candidates = all_candidates
    else:
        logger.info("Step 4: Stage 2 - Counterfactual Teacher-Forced Verification...")
        logger.info(f"  Epsilon = {args.epsilon}")
        logger.info(f"  Check tokens after segment = {args.n_check_tokens}")

        verified_candidates = []
        stats = {"total_checked": 0, "passed": 0, "rejected": 0}

        for idx, (traj_idx, segments) in enumerate(all_candidates):
            traj = trajectories[traj_idx]
            full_ids = traj["full_ids"]
            attention_mask = traj.get("full_attention_mask", torch.ones_like(full_ids))
            prompt_length = traj["prompt_length"]

            verified_segs = []
            for seg_start, seg_end in segments:
                passed, delta_ll = counterfactual_verify_segment(
                    model=model,
                    full_ids=full_ids,
                    attention_mask=attention_mask,
                    segment_start=seg_start,
                    segment_end=seg_end,
                    prompt_length=prompt_length,
                    epsilon=args.epsilon,
                    n_check_tokens=args.n_check_tokens,
                )

                stats["total_checked"] += 1
                if passed:
                    stats["passed"] += 1
                    verified_segs.append((seg_start, seg_end))
                else:
                    stats["rejected"] += 1

            if verified_segs:
                verified_candidates.append((traj_idx, verified_segs))

            if (idx + 1) % 100 == 0:
                rate = stats["passed"] / max(stats["total_checked"], 1)
                logger.info(f"  Verified {idx + 1}/{len(all_candidates)} trajectories, "
                            f"pass rate: {rate:.1%}")
                torch.cuda.empty_cache()

        logger.info(f"  Stage 2 complete:")
        logger.info(f"    Segments checked: {stats['total_checked']}")
        logger.info(f"    Passed: {stats['passed']} ({stats['passed']/max(stats['total_checked'],1):.1%})")
        logger.info(f"    Rejected: {stats['rejected']}")

    # ── 5. 组装 SFT 数据 ──
    logger.info("Step 5: Assembling SFT training data...")

    sft_data = []
    assembly_stats = {"total_skips": 0, "total_skip_tokens": 0, "total_output_tokens": 0}

    for traj_idx, verified_segs in verified_candidates:
        traj = trajectories[traj_idx]
        full_ids = traj["full_ids"]
        prompt_length = traj["prompt_length"]

        sft_ids = assemble_sft_trajectory(
            full_ids=full_ids,
            prompt_length=prompt_length,
            verified_segments=verified_segs,
            skip_token_id=skip_token_id,
        )

        n_skips = len(verified_segs)
        n_skip_tokens = sum(e - s + 1 for s, e in verified_segs)
        orig_len = full_ids.shape[0]
        sft_len = sft_ids.shape[0]

        sft_entry = {
            "sft_ids": sft_ids,
            "prompt_length": prompt_length,
            "original_length": orig_len,
            "sft_length": sft_len,
            "num_skips": n_skips,
            "skip_segments": verified_segs,
            "skip_ratio": n_skip_tokens / max(orig_len - prompt_length, 1),
            "ground_truth": traj.get("ground_truth", traj.get("answer", "")),
        }
        sft_data.append(sft_entry)

        assembly_stats["total_skips"] += n_skips
        assembly_stats["total_skip_tokens"] += n_skip_tokens
        assembly_stats["total_output_tokens"] += sft_len

    logger.info(f"  SFT data assembled: {len(sft_data)} trajectories")
    logger.info(f"    Total <SKIP> insertions: {assembly_stats['total_skips']}")
    logger.info(f"    Total tokens replaced: {assembly_stats['total_skip_tokens']}")
    avg_skip_ratio = assembly_stats["total_skip_tokens"] / max(assembly_stats["total_output_tokens"], 1)
    logger.info(f"    Average skip ratio: {avg_skip_ratio:.1%}")

    # ── 6. 保存 ──
    logger.info("Step 6: Saving...")

    # 保存 SFT 数据
    sft_path = os.path.join(args.output_dir, "sft_training_data.pt")
    torch.save(sft_data, sft_path)
    logger.info(f"  SFT data saved to {sft_path}")

    # 保存统计
    stats_summary = {
        "num_sft_trajectories": len(sft_data),
        "num_source_trajectories": len(trajectories),
        "total_skip_insertions": assembly_stats["total_skips"],
        "total_tokens_replaced": assembly_stats["total_skip_tokens"],
        "avg_skip_ratio": avg_skip_ratio,
        "stage1_segments": stage1_stats["total_segments"],
        "config": {
            "epsilon": args.epsilon,
            "max_skip_ratio": args.max_skip_ratio,
            "min_segment_len": args.min_segment_len,
            "max_segment_len": args.max_segment_len,
            "mcig_threshold_factor": args.mcig_threshold_factor,
            "skip_counterfactual": args.skip_counterfactual,
        }
    }
    stats_path = os.path.join(args.output_dir, "sft_data_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats_summary, f, indent=2, ensure_ascii=False)
    logger.info(f"  Stats saved to {stats_path}")

    # ── 7. 质量报告 ──
    logger.info("\n" + "=" * 70)
    logger.info("SFT DATA QUALITY REPORT")
    logger.info("=" * 70)

    if sft_data:
        skip_ratios = [d["skip_ratio"] for d in sft_data]
        skip_counts = [d["num_skips"] for d in sft_data]
        logger.info(f"  Trajectories:        {len(sft_data)}")
        logger.info(f"  Skip ratio (mean):   {np.mean(skip_ratios):.1%}")
        logger.info(f"  Skip ratio (median): {np.median(skip_ratios):.1%}")
        logger.info(f"  Skip ratio (min):    {np.min(skip_ratios):.1%}")
        logger.info(f"  Skip ratio (max):    {np.max(skip_ratios):.1%}")
        logger.info(f"  Skips/trajectory:    {np.mean(skip_counts):.1f} (mean)")
        logger.info(f"  Zero-skip trajs:     {sum(1 for r in skip_ratios if r == 0)}")

    # 目标检查
    target_min = 2000
    if len(sft_data) < target_min:
        logger.warning(f"  ⚠️ Only {len(sft_data)} trajectories, target is {target_min}")
        logger.warning(f"  Try: lower --mcig_threshold_factor or increase source data")
    else:
        logger.info(f"  ✅ {len(sft_data)} trajectories meets target of {target_min}")

    logger.info("=" * 70)
    logger.info("Next step: Phase 1.5 SFT training (train_phase15.py)")
    logger.info(f"  python train_phase15.py --sft_data {sft_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
