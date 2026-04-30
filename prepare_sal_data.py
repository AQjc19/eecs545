"""
SAL Test — Step 1: Data Preparation

Generates correct GSM8K trajectories from Phase 1 checkpoint,
then creates K=1,2,3,4 latent versions with anchor targets for SAL training.

Usage:
    python prepare_sal_data.py \
        --checkpoint ./checkpoints/phase1/best \
        --output_dir ./sal_test_data \
        --num_trajectories 500
"""

import os, json, random, argparse, logging, re
from typing import List, Dict, Tuple

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import get_config, DataConfig
from data_utils import (GSM8KEvalDataset, MathProblemDataset, build_prompt,
                        extract_model_answer, check_answer)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_sal")


# ============================================================
# Critical Token Detection
# ============================================================

MATH_SYMBOLS = set("+-*/×÷=^{}()[]\\$<>|")
PROTECTED_WORDS = {"boxed", "frac", "sqrt", "cdot", "times", "div", "pm",
                   "step", "answer", "therefore", "hence", "thus", "total"}

def is_critical_token(token_str: str) -> bool:
    """A token is critical if it contains digits, math symbols, or protected words."""
    s = token_str.strip()
    if not s:
        return False
    if any(c.isdigit() for c in s):
        return True
    if any(c in MATH_SYMBOLS for c in s):
        return True
    s_lower = s.lower().strip()
    if any(w in s_lower for w in PROTECTED_WORDS):
        return True
    return False


def build_critical_mask(token_ids: List[int], tokenizer) -> List[bool]:
    """Return per-token critical mask."""
    mask = []
    for tid in token_ids:
        text = tokenizer.decode([tid], skip_special_tokens=False)
        mask.append(is_critical_token(text))
    # Always protect first and last 3 tokens
    for i in range(min(3, len(mask))):
        mask[i] = True
    for i in range(max(0, len(mask) - 3), len(mask)):
        mask[i] = True
    return mask


# ============================================================
# K-Version Construction
# ============================================================

def create_k_version(
    response_ids: List[int],
    critical_mask: List[bool],
    K: int,
    think_token_id: int,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Replace non-critical segments with <THINK> tokens, max K consecutive.

    Returns:
        new_ids: token IDs with <THINK> insertions
        anchor_targets: original token ID at <THINK> positions, -100 elsewhere
        think_mask: 1 at <THINK> positions, 0 elsewhere
    """
    new_ids = []
    anchor_targets = []
    think_mask = []

    i = 0
    n = len(response_ids)

    while i < n:
        if critical_mask[i]:
            new_ids.append(response_ids[i])
            anchor_targets.append(-100)
            think_mask.append(0)
            i += 1
        else:
            # Find extent of non-critical segment
            seg_start = i
            while i < n and not critical_mask[i]:
                i += 1
            seg = response_ids[seg_start:i]
            seg_len = len(seg)

            # Replace in groups: K <THINK> + 1 explicit per group
            pos = 0
            while pos < seg_len:
                remaining = seg_len - pos
                if remaining == 1:
                    # Single token: keep explicit
                    new_ids.append(seg[pos])
                    anchor_targets.append(-100)
                    think_mask.append(0)
                    pos += 1
                else:
                    # Group: replace up to K with <THINK>, keep 1 explicit
                    think_count = min(K, remaining - 1)
                    for j in range(think_count):
                        new_ids.append(think_token_id)
                        anchor_targets.append(seg[pos + j])
                        think_mask.append(1)
                    # Keep the token after the think group as explicit
                    explicit_idx = pos + think_count
                    new_ids.append(seg[explicit_idx])
                    anchor_targets.append(-100)
                    think_mask.append(0)
                    pos = explicit_idx + 1

    return new_ids, anchor_targets, think_mask


# ============================================================
# Trajectory Generation
# ============================================================
# ============================================================
# Trajectory Generation (OPTIMIZED FOR A100)
# ============================================================

@torch.inference_mode()
def generate_correct_trajectories(
    model, tokenizer, problems, data_config,
    num_target=500, max_attempts_per_problem=4,
    max_new_tokens=420, temperature=0.7, batch_size=64, # ADDED batch_size
):
    """Generate correct GSM8K trajectories using batched generation to saturate GPU."""
    device = next(model.parameters()).device
    model.eval()
    correct_trajs = []
    
    # Pre-build all prompt strings
    all_prompts = [build_prompt(prob["question"], tokenizer, data_config) for prob in problems]
    
    # We will loop over problems in batches. We do multiple attempts per batch if needed.
    # To maximize efficiency, we process the dataset in chunks.
    
    problem_idx = 0
    total_problems = len(problems)
    
    tokenizer.padding_side = 'left' # Crucial for batched generation with decoder-only models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logger.info(f"Starting batched generation with batch_size={batch_size}...")

    while len(correct_trajs) < num_target and problem_idx < total_problems:
        # 1. Grab a batch of problems
        end_idx = min(problem_idx + batch_size, total_problems)
        batch_probs = problems[problem_idx:end_idx]
        batch_prompts = all_prompts[problem_idx:end_idx]
        problem_idx = end_idx
        
        # We need to track which problems in this batch have succeeded
        succeeded_in_batch = [False] * len(batch_probs)
        
        # 2. Attempt generation up to max_attempts_per_problem
        for attempt in range(max_attempts_per_problem):
            # Find which indices still need attempting
            active_indices = [i for i, success in enumerate(succeeded_in_batch) if not success]
            if not active_indices or len(correct_trajs) >= num_target:
                break
                
            active_prompts = [batch_prompts[i] for i in active_indices]
            
            # Tokenize the active prompts with left padding
            inputs = tokenizer(
                active_prompts, 
                return_tensors="pt", 
                padding=True, 
                add_special_tokens=False
            ).to(device)
            
            # Generate!
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # 3. Process the batch results
            for local_i, active_idx in enumerate(active_indices):
                if len(correct_trajs) >= num_target:
                    break
                    
                prob = batch_probs[active_idx]
                prompt_text = batch_prompts[active_idx]
                full_id_seq = outputs[local_i]
                
                # We must carefully slice out the padding and the prompt to get just the response
                # inputs.attention_mask tells us how long the prompt actually was (excluding padding)
                prompt_len = inputs.attention_mask[local_i].sum().item()
                
                # The total sequence includes left padding, the prompt, and the response.
                # To get the response, we start slicing from the end of the prompt.
                # Since padding is on the left, the prompt ends at inputs.input_ids.shape[1]
                prompt_end_idx = inputs.input_ids.shape[1]
                resp_ids = full_id_seq[prompt_end_idx:]
                
                # Remove eos/pad tokens from the end of resp_ids if they exist
                valid_resp_len = len(resp_ids)
                for j, tid in enumerate(resp_ids):
                    if tid == tokenizer.eos_token_id or tid == tokenizer.pad_token_id:
                        valid_resp_len = j
                        break
                resp_ids = resp_ids[:valid_resp_len]

                resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True)
                pred = extract_model_answer(resp_text)
                
                if check_answer(pred, prob["answer"]):
                    succeeded_in_batch[active_idx] = True
                    
                    # Re-tokenize prompt without padding to save clean prompt_ids
                    clean_prompt_ids = tokenizer(
                        prompt_text, return_tensors="pt", add_special_tokens=False
                    )["input_ids"].squeeze(0).tolist()
                    
                    correct_trajs.append({
                        "question": prob["question"],
                        "answer": prob["answer"],
                        "prompt_text": prompt_text,
                        "prompt_ids": clean_prompt_ids,
                        "response_ids": resp_ids.cpu().tolist(),
                        "response_text": resp_text,
                    })
                    
                    if len(correct_trajs) % 50 == 0:
                        logger.info(f"  Collected {len(correct_trajs)}/{num_target} "
                                    f"correct trajectories...")

    logger.info(f"  Total correct trajectories: {len(correct_trajs)}")
    return correct_trajs

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./sal_test_data")
    parser.add_argument("--num_trajectories", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0")
    config = get_config()
    data_config = DataConfig()

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    think_token_id = tokenizer.convert_tokens_to_ids("<SKIP>")
    logger.info(f"  <THINK> token ID (using <SKIP>): {think_token_id}")

    # Load problems
    eval_ds = GSM8KEvalDataset(config.data, tokenizer)
    problems = eval_ds.problems if hasattr(eval_ds, "problems") else eval_ds
    logger.info(f"  Loaded {len(problems)} GSM8K problems")

    # Generate correct trajectories
    logger.info("Generating correct trajectories...")
    trajs = generate_correct_trajectories(
        model, tokenizer, problems, data_config,
        num_target=args.num_trajectories)

    # Create K-versions
    all_data = {1: [], 2: [], 3: [], 4: []}
    stats = {K: {"total": 0, "think_count": 0, "explicit_count": 0} for K in [1, 2, 3, 4]}

    for traj in trajs:
        resp_ids = traj["response_ids"]
        critical_mask = build_critical_mask(resp_ids, tokenizer)

        for K in [1, 2, 3, 4]:
            new_ids, anchors, t_mask = create_k_version(
                resp_ids, critical_mask, K, think_token_id)

            sample = {
                "prompt_ids": traj["prompt_ids"],
                "response_ids": new_ids,
                "anchor_targets": anchors,
                "think_mask": t_mask,
                "original_response_ids": resp_ids,
                "K": K,
                "question": traj["question"],
                "answer": traj["answer"],
            }
            all_data[K].append(sample)

            n_think = sum(t_mask)
            n_explicit = len(new_ids) - n_think
            stats[K]["total"] += 1
            stats[K]["think_count"] += n_think
            stats[K]["explicit_count"] += n_explicit

    # Report stats
    logger.info("=" * 60)
    logger.info("Data Statistics:")
    for K in [1, 2, 3, 4]:
        n = stats[K]["total"]
        avg_think = stats[K]["think_count"] / max(n, 1)
        avg_expl = stats[K]["explicit_count"] / max(n, 1)
        avg_len = avg_think + avg_expl
        ratio = avg_think / max(avg_len, 1)
        logger.info(f"  K={K}: {n} samples, avg_len={avg_len:.0f}, "
                    f"avg_think={avg_think:.1f}, avg_explicit={avg_expl:.1f}, "
                    f"think_ratio={ratio:.1%}")
    logger.info("=" * 60)

    # Save
    save_path = os.path.join(args.output_dir, "sal_data.pt")
    torch.save(all_data, save_path)
    logger.info(f"Saved to {save_path}")

    # Also save raw trajectories for eval
    raw_path = os.path.join(args.output_dir, "raw_trajectories.pt")
    torch.save(trajs, raw_path)
    logger.info(f"Saved raw trajectories to {raw_path}")


if __name__ == "__main__":
    main()
