"""
SAL Test — Step 3: Evaluation (A100 80GB Optimized)

Key optimization: Batched forced-K latent generation.
All B problems processed simultaneously at each generation step.
Think steps batched across all B problems: adapter(h_batch) → Transformer(z_batch, kv_batch).

B=32 for 1.5B on A100 80GB → 200 problems in ~6 batches per K value.

Usage:
    python eval_sal_test.py \
        --checkpoint ./sal_test_models/group_b_beta1.0/best \
        --data_dir ./sal_test_data \
        --output_dir ./sal_test_results/group_b_beta1.0 \
        --k_values 0,1,2,3,4 --eval_batch_size 32
"""

import os, sys, json, argparse, logging, time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import get_config, DataConfig
from data_utils import (GSM8KEvalDataset, build_prompt, extract_model_answer, check_answer)


# Inline adapter (matches training script — no external dependency)
class LatentBridgeAdapter(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = torch.nn.RMSNorm(d_model, elementwise_affine=True)
        self.up = torch.nn.Linear(d_model, d_model, bias=False)
        self.down = torch.nn.Linear(d_model, d_model, bias=False)
    def forward(self, h):
        return h + self.down(F.silu(self.up(self.norm(h))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("eval_sal")


# ============================================================
# Batched Forced-K Generation
# ============================================================

@torch.inference_mode()
def batched_forced_k_generate(
    model, adapter, tokenizer,
    prompts: List[str], K: int, think_token_id: int,
    max_new_tokens: int = 420,
    max_consec_think: int = 32,
) -> List[Tuple[str, Dict]]:
    """
    Generate for B problems simultaneously with forced K latent steps
    after each explicit token. When K=0, standard batched greedy.

    Returns list of (text, diagnostics) tuples.
    """
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id or 0
    eos_id = tokenizer.eos_token_id
    B = len(prompts)

    # Tokenize and left-pad
    encs = [tokenizer(p, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
            for p in prompts]
    max_pl = max(len(e) for e in encs)

    input_ids = torch.full((B, max_pl), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((B, max_pl), dtype=torch.long, device=device)
    for i, e in enumerate(encs):
        input_ids[i, max_pl-len(e):] = e.to(device)
        attn[i, max_pl-len(e):] = 1

    # Initial forward - all B prompts at once
    out = model(input_ids=input_ids, attention_mask=attn,
                use_cache=True, output_hidden_states=True)
    kv_cache = out.past_key_values
    h_last = out.hidden_states[-1][:, -1, :]  # (B, d)

    is_eos = torch.zeros(B, dtype=torch.bool, device=device)
    generated = [[] for _ in range(B)]
    think_counts = [0] * B
    explicit_counts = [0] * B

    # For KV consistency measurement
    kv_sims_per_sample = [[] for _ in range(B)]

    for gen_step in range(max_new_tokens):
        if is_eos.all():
            break

        # Generate explicit token (greedy for all B)
        logits = model.lm_head(h_last)  # (B, vocab)
        next_tokens = logits.argmax(dim=-1)  # (B,)

        # Check EOS
        hit_eos = (next_tokens == eos_id) | is_eos
        for b in range(B):
            if not is_eos[b] and not hit_eos[b]:
                generated[b].append(next_tokens[b].item())
                explicit_counts[b] += 1
        is_eos = is_eos | hit_eos
        if is_eos.all():
            break

        # Feed explicit token - batched
        emb = model.get_input_embeddings()(next_tokens.unsqueeze(1))  # (B, 1, d)
        new_col = (~is_eos).long().unsqueeze(1)
        attn = torch.cat([attn, new_col], dim=1)
        out = model(inputs_embeds=emb, attention_mask=attn,
                    past_key_values=kv_cache, use_cache=True,
                    output_hidden_states=True)
        kv_cache = out.past_key_values
        h_last = out.hidden_states[-1][:, -1, :]  # (B, d)

        # Record explicit KV key for consistency comparison
        if K > 0:
            explicit_key = kv_cache[-1][0][:, :, -1, :].clone()  # (B, n_heads, head_dim)

        # Force K latent steps - ALL B problems simultaneously
        for k_step in range(K):
            z = adapter(h_last)  # (B, d) — batched adapter!
            new_col = (~is_eos).long().unsqueeze(1)
            attn = torch.cat([attn, new_col], dim=1)
            out = model(inputs_embeds=z.unsqueeze(1), attention_mask=attn,
                        past_key_values=kv_cache, use_cache=True,
                        output_hidden_states=True)
            kv_cache = out.past_key_values
            h_last = out.hidden_states[-1][:, -1, :]

            for b in range(B):
                if not is_eos[b]:
                    think_counts[b] += 1

            # KV consistency: latent key vs explicit key
            if k_step == 0:  # Only first think step for efficiency
                latent_key = kv_cache[-1][0][:, :, -1, :]
                for b in range(B):
                    if not is_eos[b]:
                        sim = F.cosine_similarity(
                            latent_key[b].flatten().unsqueeze(0).float(),
                            explicit_key[b].flatten().unsqueeze(0).float()
                        ).item()
                        kv_sims_per_sample[b].append(sim)

    results = []
    for b in range(B):
        text = tokenizer.decode(generated[b], skip_special_tokens=True)
        avg_kv = np.mean(kv_sims_per_sample[b]) if kv_sims_per_sample[b] else -1.0
        results.append((text, {
            "think_steps": think_counts[b],
            "explicit_steps": explicit_counts[b],
            "kv_consistency": avg_kv,
        }))

    return results


# ============================================================
# Batched Exit Loss + Anchor Accuracy Measurement
# ============================================================

@torch.inference_mode()
def batched_measure_exit_anchor(
    model, adapter, tokenizer,
    trajectories: List[Dict], K: int, think_token_id: int,
    batch_size: int = 16,
) -> Tuple[List[float], List[float]]:
    """
    Measure exit loss and anchor accuracy on a batch of trajectories.
    For each trajectory: insert K latent steps before every explicit token,
    measure CE loss at exit points and LM Head accuracy at latent points.
    """
    device = next(model.parameters()).device
    all_exit_losses = []
    all_anchor_accs = []

    for bi in range(0, len(trajectories), batch_size):
        batch_trajs = trajectories[bi:bi+batch_size]
        B = len(batch_trajs)

        # Get all response ids
        resp_ids_list = []
        for t in batch_trajs:
            r = t.get("original_response_ids", t.get("response_ids", []))
            if isinstance(r, torch.Tensor):
                r = r.tolist()
            resp_ids_list.append(r)

        # Process each trajectory individually (varying lengths make batching complex)
        for ti, traj in enumerate(batch_trajs):
            prompt_ids = torch.tensor(traj["prompt_ids"], dtype=torch.long, device=device)
            resp = resp_ids_list[ti]
            if len(resp) == 0:
                continue

            # Forward prompt
            out = model(input_ids=prompt_ids.unsqueeze(0), use_cache=True, output_hidden_states=True)
            kv = out.past_key_values
            h = out.hidden_states[-1][:, -1, :]  # (1, d)

            exit_l = []
            anchor_ok, anchor_tot = 0, 0

            for t_idx, tok_id in enumerate(resp):
                # K latent steps before this explicit token
                for k in range(K):
                    z = adapter(h)
                    out = model(inputs_embeds=z.unsqueeze(1),  # (1, 1, d)
                                past_key_values=kv, use_cache=True,
                                output_hidden_states=True)
                    kv = out.past_key_values
                    h = out.hidden_states[-1][:, -1, :]

                    # Anchor accuracy
                    pred = model.lm_head(h.unsqueeze(0)).argmax(dim=-1).item()
                    if pred == tok_id:
                        anchor_ok += 1
                    anchor_tot += 1

                # Exit: predict this explicit token
                if K > 0:
                    el = model.lm_head(h.unsqueeze(0))
                    target = torch.tensor([tok_id], device=device)
                    exit_l.append(F.cross_entropy(el.view(1, -1), target).item())

                # Feed actual token
                emb = model.get_input_embeddings()(torch.tensor([[tok_id]], device=device))
                out = model(inputs_embeds=emb, past_key_values=kv,
                            use_cache=True, output_hidden_states=True)
                kv = out.past_key_values
                h = out.hidden_states[-1][:, -1, :]

                # Early stop for long sequences
                if t_idx >= 150:
                    break

            all_exit_losses.append(np.mean(exit_l) if exit_l else 0.0)
            all_anchor_accs.append(anchor_ok / max(anchor_tot, 1))

        torch.cuda.empty_cache()

    return all_exit_losses, all_anchor_accs


# ============================================================
# Main
# ============================================================

def evaluate(args):
    device = torch.device("cuda:0")
    config = get_config(); data_config = DataConfig()
    k_values = [int(k) for k in args.k_values.split(",")]

    logger.info(f"Loading model from {args.checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    think_id = tokenizer.convert_tokens_to_ids("<SKIP>")

    adapter = LatentBridgeAdapter(model.config.hidden_size)
    # Try new name first, fall back to old name
    for aname in ["adapter.pt", "skip_adapter.pt"]:
        ap = os.path.join(args.checkpoint, aname)
        if os.path.exists(ap):
            adapter.load_state_dict(torch.load(ap, map_location="cpu", weights_only=False))
            logger.info(f"  Loaded adapter from {ap}")
            break
    adapter = adapter.to(device=device, dtype=model.dtype); adapter.eval(); model.eval()

    ed = GSM8KEvalDataset(config.data, tokenizer)
    problems = (ed.problems if hasattr(ed, "problems") else ed)[:args.max_eval]
    logger.info(f"  {len(problems)} problems, K values: {k_values}")

    raw_trajs = None
    rtp = os.path.join(args.data_dir, "raw_trajectories.pt")
    if os.path.exists(rtp):
        raw_trajs = torch.load(rtp, weights_only=False)[:args.n_exit_trajs]
        logger.info(f"  {len(raw_trajs)} trajectories for exit loss measurement")

    results = {}
    eb = args.eval_batch_size

    for K in k_values:
        t0 = time.time()
        logger.info(f"\n{'='*60}\nEvaluating K={K}\n{'='*60}")

        # Metric 1: Accuracy@K (batched)
        correct, total = 0, len(problems)
        all_think, all_explicit, all_kvsim = 0, 0, []

        for bi in range(0, total, eb):
            batch_p = problems[bi:bi+eb]
            prompts = [build_prompt(p["question"], tokenizer, data_config) for p in batch_p]

            batch_results = batched_forced_k_generate(
                model, adapter, tokenizer, prompts,
                K=K, think_token_id=think_id, max_new_tokens=420)

            for j, (text, diag) in enumerate(batch_results):
                pred = extract_model_answer(text)
                if check_answer(pred, batch_p[j]["answer"]):
                    correct += 1
                all_think += diag["think_steps"]
                all_explicit += diag["explicit_steps"]
                if diag["kv_consistency"] >= 0:
                    all_kvsim.append(diag["kv_consistency"])

            if (bi + eb) % 100 < eb:
                logger.info(f"  [{min(bi+eb, total)}/{total}] acc={correct/(bi+len(batch_p)):.4f}")

            torch.cuda.empty_cache()

        accuracy = correct / max(total, 1)
        avg_kv = np.mean(all_kvsim) if all_kvsim else -1.0

        # Metrics 3&5: Exit loss + Anchor accuracy
        avg_exit, avg_anchor = -1.0, -1.0
        if raw_trajs and K > 0:
            logger.info(f"  Measuring exit loss & anchor accuracy on {len(raw_trajs)} trajectories...")
            exit_losses, anchor_accs = batched_measure_exit_anchor(
                model, adapter, tokenizer, raw_trajs,
                K=K, think_token_id=think_id, batch_size=8)
            avg_exit = np.mean(exit_losses) if exit_losses else -1.0
            avg_anchor = np.mean(anchor_accs) if anchor_accs else -1.0

        elapsed = time.time() - t0
        logger.info(f"  K={K} done in {elapsed:.0f}s: Acc={accuracy:.4f} "
                     f"Exit={avg_exit:.4f} Anchor={avg_anchor:.4f} KV={avg_kv:.4f}")

        results[K] = {
            "accuracy": accuracy, "correct": correct, "total": total,
            "avg_think": all_think / max(total, 1),
            "avg_explicit": all_explicit / max(total, 1),
            "kv_consistency": avg_kv,
            "exit_loss": avg_exit,
            "anchor_accuracy": avg_anchor,
            "time_sec": elapsed,
        }

    # Compute aLRH
    base_acc = results.get(0, {}).get("accuracy", 0.69)
    aLRH = 0
    for K in sorted(k_values):
        if results[K]["accuracy"] >= base_acc - 0.05:
            aLRH = K
    results["aLRH"] = aLRH
    results["baseline_accuracy"] = base_acc

    # Summary
    logger.info(f"\n{'='*80}\n{'K':>3} | {'Acc':>7} | {'ExitL':>7} | {'AnchAcc':>7} | {'KV-Sim':>7} | {'Think':>6} | {'Expl':>6}")
    logger.info("-" * 70)
    for K in sorted(k_values):
        r = results[K]
        logger.info(f"{K:>3} | {r['accuracy']:>7.4f} | {r['exit_loss']:>7.4f} | "
                     f"{r['anchor_accuracy']:>7.4f} | {r['kv_consistency']:>7.4f} | "
                     f"{r['avg_think']:>6.0f} | {r['avg_explicit']:>6.0f}")
    logger.info(f"\naLRH = {aLRH}  (baseline = {base_acc:.4f}, threshold = {base_acc-0.05:.4f})")

    if aLRH >= 3:
        logger.info("✅ SAL VERIFIED — proceed to full SPARK pipeline")
    elif aLRH >= 2:
        logger.info("⚠️  SAL PARTIAL — may work with K_max=2 limit")
    else:
        logger.info("❌ SAL INSUFFICIENT — revise approach")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved to {args.output_dir}/eval_results.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./sal_test_data")
    parser.add_argument("--output_dir", type=str, default="./sal_test_results")
    parser.add_argument("--k_values", type=str, default="0,1,2,3,4")
    parser.add_argument("--max_eval", type=int, default=200)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--n_exit_trajs", type=int, default=80)
    args = parser.parse_args()
    evaluate(args)

if __name__ == "__main__":
    main()