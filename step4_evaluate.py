"""
Step 4 (BATCHED): Evaluate with K latent steps, fully batched.

Key design:
  - K=0: model.generate() with batch_size=32 (maximum throughput)
  - K>0: batched autoregressive loop with per-sample state machine
         tracking newline counts, latent phase, and EOS per sample
  - Processes eval_batch=16 samples simultaneously for K>0
  - A100 80GB: 1.5B model + batch=32 → ~25GB peak, well within budget
"""

import os, json, re, argparse, logging, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("anchor_eval")


class LatentBridge(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.proj.weight)
        self.register_buffer("embed_norm", torch.tensor(2.0))
    def forward(self, h):
        z = self.proj(self.norm(h))
        return z * (self.embed_norm / z.norm(dim=-1, keepdim=True).clamp(min=1e-8))


def build_prompt(q, tok):
    return tok.apply_chat_template(
        [{"role": "system", "content": "Solve the math problem step by step. Put the final answer in \\boxed{}."},
         {"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True)

def extract_answer(text):
    m = re.findall(r'\\boxed\{([^}]*)\}', text)
    if m: return m[-1].strip().replace(",", "")
    m = re.findall(r'####\s*([\-\d,.]+)', text)
    if m: return m[-1].strip().replace(",", "")
    nums = re.findall(r'[\-\d,.]+', text)
    return nums[-1].replace(",", "") if nums else ""

def normalize_answer(a):
    a = a.strip().replace(",", "").replace("$", "").replace("%", "")
    try: return str(int(float(a)))
    except: return a

def check_answer(p, g): return normalize_answer(p) == normalize_answer(g)

def extract_gt(sol):
    m = re.findall(r'####\s*([\-\d,.]+)', sol)
    return m[-1].strip().replace(",", "") if m else ""


# ============================================================
# K=0: Pure batched generate (max throughput)
# ============================================================

@torch.inference_mode()
def eval_k0_batched(model, tokenizer, problems, batch_size=32, max_new=400):
    """Standard batched generation, no latent. Uses model.generate()."""
    correct = 0
    total = len(problems)

    for i in range(0, total, batch_size):
        batch = problems[i:i+batch_size]
        prompts = [build_prompt(p["question"], tokenizer) for p in batch]
        gts = [extract_gt(p["answer"]) for p in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=1024).to("cuda")

        outputs = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tokenizer.pad_token_id)

        for j in range(len(batch)):
            pl = inputs["input_ids"][j].ne(tokenizer.pad_token_id).sum().item()
            resp = tokenizer.decode(outputs[j][pl:], skip_special_tokens=True)
            if check_answer(extract_answer(resp), gts[j]):
                correct += 1

        if (i + batch_size) % 64 == 0 or i + batch_size >= total:
            logger.info(f"    K=0: {min(i+batch_size, total)}/{total}, "
                         f"acc={correct/min(i+batch_size, total):.1%}")

    return {"accuracy": correct/total, "correct": correct, "total": total,
            "avg_exit_loglik": 0.0, "avg_anchor_max_prob": 0.0}


# ============================================================
# K>0: Batched autoregressive with per-sample state machine
# ============================================================

@torch.inference_mode()
def eval_kN_batched(model, bridge, tokenizer, problems, K,
                    batch_size=16, max_new=400, insert_after=2):
    """
    Batched eval with K latent steps.

    State machine per sample:
      EXPLICIT_PRE  → counting newlines, generating explicit tokens
      LATENT        → doing K latent steps via bridge
      EXPLICIT_POST → normal generation after latent
      DONE          → hit EOS
    """
    EXPLICIT_PRE, LATENT, EXPLICIT_POST, DONE = 0, 1, 2, 3

    correct = 0
    total = len(problems)
    all_exit_ll = []
    all_anchor_probs = []

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or 0
    nl_ids = tokenizer.encode("\n", add_special_tokens=False)
    nl_id = nl_ids[0] if nl_ids else -1

    for batch_start in range(0, total, batch_size):
        batch = problems[batch_start:batch_start+batch_size]
        B = len(batch)
        gts = [extract_gt(p["answer"]) for p in batch]
        prompts = [build_prompt(p["question"], tokenizer) for p in batch]

        # Tokenize with left-padding for batched generation
        tokenizer.padding_side = "left"
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=1024).to("cuda")
        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]
        prompt_lens = attn_mask.sum(dim=1).tolist()

        # Initial forward (batched)
        out = model(input_ids=input_ids, attention_mask=attn_mask,
                    use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        h_last = out.hidden_states[-1][:, -1, :]  # (B, d)
        logits = out.logits[:, -1, :]  # (B, V)
        del out

        # Per-sample state
        states = [EXPLICIT_PRE] * B
        nl_counts = [0] * B
        latent_steps_done = [0] * B
        generated = [[] for _ in range(B)]
        sample_exit_ll = [None] * B
        sample_anchor_probs = [[] for _ in range(B)]
        cur_attn = attn_mask

        for step in range(max_new + K * 2):  # Extra room for latent steps
            if all(s == DONE for s in states):
                break

            # Determine actions per sample based on state
            next_tokens = torch.full((B,), pad_id, dtype=torch.long, device="cuda")
            use_bridge_mask = torch.zeros(B, dtype=torch.bool, device="cuda")

            for b in range(B):
                if states[b] == DONE:
                    continue

                if states[b] == EXPLICIT_PRE:
                    tok = logits[b].argmax().item()
                    if tok == eos_id:
                        states[b] = DONE
                        continue
                    next_tokens[b] = tok
                    generated[b].append(tok)
                    if tok == nl_id:
                        nl_counts[b] += 1
                    if nl_counts[b] >= insert_after:
                        states[b] = LATENT
                        latent_steps_done[b] = 0

                elif states[b] == LATENT:
                    if latent_steps_done[b] < K:
                        use_bridge_mask[b] = True
                        latent_steps_done[b] += 1
                        # Record anchor diagnostic
                        probs = F.softmax(logits[b], dim=-1)
                        sample_anchor_probs[b].append(probs.max().item())
                    if latent_steps_done[b] >= K:
                        # Record exit log-likelihood
                        exit_tok = logits[b].argmax().item()
                        sample_exit_ll[b] = F.log_softmax(logits[b], dim=-1)[exit_tok].item()
                        states[b] = EXPLICIT_POST

                elif states[b] == EXPLICIT_POST:
                    tok = logits[b].argmax().item()
                    if tok == eos_id:
                        states[b] = DONE
                        continue
                    next_tokens[b] = tok
                    generated[b].append(tok)

            # Build next input embeddings
            # Bridge for latent samples, normal embedding for explicit
            emb_normal = model.get_input_embeddings()(next_tokens.unsqueeze(1))  # (B,1,d)
            emb_bridge = bridge(h_last).unsqueeze(1)  # (B,1,d)
            bridge_3d = use_bridge_mask.view(B, 1, 1)
            next_emb = torch.where(bridge_3d, emb_bridge, emb_normal)

            # Attention: extend by 1 for active samples
            active = torch.tensor([s != DONE for s in states], dtype=torch.long, device="cuda")
            new_col = active.unsqueeze(1)
            cur_attn = torch.cat([cur_attn, new_col], dim=1)

            # Forward
            out = model(inputs_embeds=next_emb, attention_mask=cur_attn,
                       past_key_values=past_kv, use_cache=True,
                       output_hidden_states=True)
            past_kv = out.past_key_values
            h_last = out.hidden_states[-1][:, -1, :]
            logits = out.logits[:, -1, :]
            del out

        # Score results
        for b in range(B):
            text = tokenizer.decode(generated[b], skip_special_tokens=True)
            pred = extract_answer(text)
            if check_answer(pred, gts[b]):
                correct += 1
            if sample_exit_ll[b] is not None:
                all_exit_ll.append(sample_exit_ll[b])
            all_anchor_probs.extend(sample_anchor_probs[b])

        done_so_far = min(batch_start + batch_size, total)
        if done_so_far % 32 == 0 or done_so_far >= total:
            logger.info(f"    K={K}: {done_so_far}/{total}, "
                         f"acc={correct/done_so_far:.1%}")

        del past_kv, cur_attn, h_last, logits
        torch.cuda.empty_cache()

    acc = correct / max(total, 1)
    return {
        "accuracy": acc, "correct": correct, "total": total,
        "avg_exit_loglik": float(np.mean(all_exit_ll)) if all_exit_ll else 0.0,
        "avg_anchor_max_prob": float(np.mean(all_anchor_probs)) if all_anchor_probs else 0.0,
    }


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--label", default="model")
    p.add_argument("--output_dir", default="./anchor_data/eval_results")
    p.add_argument("--K_values", default="0,1,2,3,4,6,8")
    p.add_argument("--n_problems", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=400)
    p.add_argument("--insert_after", type=int, default=2)
    p.add_argument("--batch_size_k0", type=int, default=32,
                    help="Batch size for K=0 (pure generate, can be large)")
    p.add_argument("--batch_size_kn", type=int, default=16,
                    help="Batch size for K>0 (latent insertion, moderate)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    Ks = [int(k) for k in a.K_values.split(",")]
    os.makedirs(a.output_dir, exist_ok=True)

    logger.info(f"Loading: {a.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(a.checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.checkpoint, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=True).to("cuda").eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load bridge
    bridge = LatentBridge(model.config.hidden_size).to(device="cuda", dtype=torch.bfloat16)
    bp = os.path.join(a.checkpoint, "bridge.pt")
    if os.path.exists(bp):
        bridge.load_state_dict(torch.load(bp, map_location="cuda", weights_only=False))
        logger.info(f"  Loaded bridge: {bp}")
    else:
        with torch.no_grad():
            W = model.get_input_embeddings().weight
            bridge.embed_norm.fill_(W[torch.randint(0, W.shape[0], (1000,))].norm(dim=-1).mean())
        logger.info("  No bridge.pt, using calibrated default")
    bridge.eval()

    logger.info("Loading GSM8K test")
    ds = load_dataset("/fs1/home/hanrui/hanrui/drought/model/gsm8k_data", "main", split="test")
    problems = list(ds.select(range(min(a.n_problems, len(ds)))))

    logger.info(f"Evaluating [{a.label}]: {len(problems)} problems, K={Ks}")
    logger.info(f"  Batch sizes: K=0 → {a.batch_size_k0}, K>0 → {a.batch_size_kn}")

    results = {}
    for K in Ks:
        logger.info(f"\n  === K={K} ===")
        t0 = __import__('time').time()

        if K == 0:
            r = eval_k0_batched(model, tokenizer, problems,
                               batch_size=a.batch_size_k0, max_new=a.max_new_tokens)
        else:
            r = eval_kN_batched(model, bridge, tokenizer, problems, K,
                               batch_size=a.batch_size_kn, max_new=a.max_new_tokens,
                               insert_after=a.insert_after)

        elapsed = __import__('time').time() - t0
        r["time_seconds"] = elapsed
        results[K] = r
        logger.info(f"  K={K}: Acc={r['accuracy']:.4f} ({r['correct']}/{r['total']}) "
                     f"in {elapsed:.0f}s")

    # Save
    out = os.path.join(a.output_dir, f"eval_{a.label}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    logger.info("\n" + "=" * 65)
    logger.info(f"RESULTS: {a.label}")
    logger.info(f"{'K':>4} | {'Acc':>8} | {'ExitLL':>8} | {'AnchProb':>9} | {'Time':>6}")
    logger.info("-" * 50)
    total_time = 0
    for K in Ks:
        r = results[K]
        total_time += r.get("time_seconds", 0)
        logger.info(f"{K:>4} | {r['accuracy']:>7.1%} | {r['avg_exit_loglik']:>8.3f} | "
                     f"{r['avg_anchor_max_prob']:>9.3f} | {r.get('time_seconds',0):>5.0f}s")
    logger.info("-" * 50)
    logger.info(f"Total eval time: {total_time:.0f}s ({total_time/60:.1f}min)")
    logger.info("=" * 65)
    logger.info(f"Saved: {out}")


if __name__ == "__main__":
    main()