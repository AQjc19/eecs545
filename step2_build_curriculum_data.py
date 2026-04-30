"""
Step 2: Build Coconut-style curriculum training data.

Takes correct trajectories and creates 5 stages:
  Stage 0: Full explicit CoT
  Stage 1: Last 1 reasoning step → latent
  Stage 2: Last 2 reasoning steps → latent
  Stage 3: Last 3 reasoning steps → latent
  Stage 4: Last 4 reasoning steps → latent

Each sample stores:
  - input_ids: full tokenized sequence
  - latent_mask: bool, True = this position is latent
  - original_ids: the token id that *would* have been at each latent position
  - prompt_length: length of the question prompt
"""

import os, json, re, argparse
import torch
from transformers import AutoTokenizer


def split_reasoning_steps(response: str):
    """Split response into reasoning steps by newlines and sentence boundaries."""
    # Split by double newline first, then single newline, then ". " for long lines
    raw_parts = re.split(r'\n+', response.strip())
    steps = []
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        # If a part is very long (>100 chars), split by ". "
        if len(part) > 100:
            sents = re.split(r'(?<=\.)\s+', part)
            steps.extend([s.strip() for s in sents if s.strip()])
        else:
            steps.append(part)
    # Merge very short steps (<10 chars) with previous
    merged = []
    for s in steps:
        if merged and len(s) < 10:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return merged if merged else [response.strip()]


def build_stage_data(traj, tokenizer, stage: int, max_seq_len: int = 1024):
    """
    Build a single training sample for a given stage.

    Stage K: the last K reasoning steps become latent.
    Stage 0: everything explicit.
    """
    prompt_text = traj["prompt"]
    response_text = traj["response"]

    # Tokenize prompt
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_ids)

    # Split response into steps
    steps = split_reasoning_steps(response_text)
    n_steps = len(steps)

    if stage > n_steps:
        stage = n_steps  # Cap at available steps

    # Determine which steps are latent
    # Steps [0, ..., n_steps-stage-1] are explicit
    # Steps [n_steps-stage, ..., n_steps-1] are latent
    latent_start_step = n_steps - stage

    # Rebuild the response token by token, tracking which belong to latent steps
    # We tokenize each step separately to get precise boundaries
    step_token_ranges = []  # (start_in_response, end_in_response, is_latent)
    all_response_ids = []
    for i, step_text in enumerate(steps):
        # Add newline between steps (except first)
        if i > 0:
            nl_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]
            all_response_ids.extend(nl_ids)

        step_ids = tokenizer(step_text, add_special_tokens=False)["input_ids"]
        start = len(all_response_ids)
        all_response_ids.extend(step_ids)
        end = len(all_response_ids)
        is_latent = (i >= latent_start_step) and (stage > 0)
        step_token_ranges.append((start, end, is_latent))

    # Add EOS
    eos_id = tokenizer.eos_token_id
    all_response_ids.append(eos_id)

    # Build full sequence
    full_ids = prompt_ids + all_response_ids

    # Truncate if needed
    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]

    # Build latent mask (True = latent, False = explicit)
    latent_mask = [False] * len(full_ids)
    for start, end, is_latent in step_token_ranges:
        if is_latent:
            for j in range(start, min(end, len(full_ids) - prompt_len)):
                abs_pos = prompt_len + j
                if abs_pos < len(full_ids):
                    latent_mask[abs_pos] = True

    # original_ids = full_ids (same sequence, latent positions have original tokens)
    original_ids = list(full_ids)

    # Compute latent stats
    n_latent = sum(latent_mask)
    n_total = len(full_ids) - prompt_len

    return {
        "input_ids": full_ids,
        "latent_mask": latent_mask,
        "original_ids": original_ids,
        "prompt_length": prompt_len,
        "stage": stage,
        "n_steps": n_steps,
        "n_latent": n_latent,
        "n_response": n_total,
        "latent_ratio": n_latent / max(n_total, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=str, default="./anchor_data/trajectories.jsonl")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", type=str, default="./anchor_data/curriculum_data.pt")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--num_stages", type=int, default=5)  # Stage 0-4
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading trajectories from {args.trajectories}")
    trajs = []
    with open(args.trajectories) as f:
        for line in f:
            trajs.append(json.loads(line))
    print(f"  Loaded {len(trajs)} trajectories")

    all_samples = []
    stage_counts = {s: 0 for s in range(args.num_stages)}
    stage_latent_ratios = {s: [] for s in range(args.num_stages)}

    for traj in trajs:
        for stage in range(args.num_stages):
            sample = build_stage_data(traj, tokenizer, stage, args.max_seq_len)
            if sample["n_response"] < 10:  # Skip too-short responses
                continue
            all_samples.append(sample)
            stage_counts[stage] += 1
            stage_latent_ratios[stage].append(sample["latent_ratio"])

    print(f"\nBuilt {len(all_samples)} samples across {args.num_stages} stages:")
    for s in range(args.num_stages):
        if stage_latent_ratios[s]:
            avg_ratio = sum(stage_latent_ratios[s]) / len(stage_latent_ratios[s])
            print(f"  Stage {s}: {stage_counts[s]} samples, avg latent ratio: {avg_ratio:.1%}")
        else:
            print(f"  Stage {s}: {stage_counts[s]} samples")

    # Save
    torch.save(all_samples, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()