"""
Step 3: Coconut-style SFT ± Semantic Anchoring Loss.

Features:
  F1: LatentBridge (RMSNorm + norm-matching)
  F2: Truncated BPTT (detach every max_bptt_steps)
  F3: Batched Stage 0 (parallel, high GPU util)
  F4: Frozen embedding + LM Head (stable anchor target)
  F5: Resume from checkpoint
  F6: OOM fallback (halve batch on OOM)
  Fixed beta (no annealing) for clean A/B comparison
"""

import os, sys, json, time, random, argparse, logging, gc
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("anchor_train")


# ============================================================
# LatentBridge
# ============================================================

class LatentBridge(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.proj.weight)
        self.register_buffer("embed_norm", torch.tensor(2.0))

    def calibrate_embed_norm(self, model):
        with torch.no_grad():
            W = model.get_input_embeddings().weight
            idx = torch.randint(0, W.shape[0], (1000,))
            avg = W[idx].norm(dim=-1).mean().item()
            self.embed_norm.fill_(avg)
            logger.info(f"  LatentBridge: avg embedding norm = {avg:.3f}")

    def forward(self, h):
        z = self.proj(self.norm(h))
        return z * (self.embed_norm / z.norm(dim=-1, keepdim=True).clamp(min=1e-8))


# ============================================================
# Dataset
# ============================================================

class CurriculumDataset(Dataset):
    def __init__(self, all_samples, stage, pad_id, max_len=1024):
        self.samples = [s for s in all_samples if s["stage"] == stage]
        self.pad_id = pad_id
        self.max_len = max_len
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        L = min(len(s["input_ids"]), self.max_len)
        return {
            "input_ids": torch.tensor(s["input_ids"][:L], dtype=torch.long),
            "latent_mask": torch.tensor(s["latent_mask"][:L], dtype=torch.bool),
            "original_ids": torch.tensor(s["original_ids"][:L], dtype=torch.long),
            "prompt_length": s["prompt_length"],
            "seq_len": L,
        }

def collate_fn(batch, pad_id=0):
    ml = max(b["seq_len"] for b in batch)
    B = len(batch)
    d = {
        "input_ids": torch.full((B, ml), pad_id, dtype=torch.long),
        "latent_mask": torch.zeros(B, ml, dtype=torch.bool),
        "original_ids": torch.full((B, ml), pad_id, dtype=torch.long),
        "attention_mask": torch.zeros(B, ml, dtype=torch.long),
        "prompt_lengths": [],
    }
    for i, b in enumerate(batch):
        L = b["seq_len"]
        d["input_ids"][i, :L] = b["input_ids"]
        d["latent_mask"][i, :L] = b["latent_mask"]
        d["original_ids"][i, :L] = b["original_ids"]
        d["attention_mask"][i, :L] = 1
        d["prompt_lengths"].append(b["prompt_length"])
    return d


# ============================================================
# Stage 0: batched parallel forward
# ============================================================

def batched_forward_stage0(model, batch, device):
    ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    pls = batch["prompt_lengths"]
    B, L = ids.shape
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=ids, attention_mask=attn, use_cache=False)
    shift_logits = out.logits[:, :-1, :]
    shift_labels = ids[:, 1:]
    loss_mask = torch.zeros(B, L - 1, device=device)
    for i in range(B):
        sl = attn[i].sum().item()
        loss_mask[i, pls[i]:sl-1] = 1.0
    ce = F.cross_entropy(shift_logits.reshape(-1, shift_logits.shape[-1]),
                         shift_labels.reshape(-1), reduction='none').reshape(B, L - 1)
    loss = (ce * loss_mask).sum() / loss_mask.sum().clamp(min=1)
    return loss, {"ce_loss": loss.item(), "anchor_loss": 0.0, "total_loss": loss.item(),
                  "exit_loss": 0.0, "anchor_acc": 0.0, "n_latent": 0,
                  "n_explicit": int(loss_mask.sum().item())}


# ============================================================
# Stage 1+: chunked forward with latent
# ============================================================

def chunked_forward(model, bridge, batch, device,
                    use_anchor=True, beta=0.3, exit_weight=3.0, max_bptt=4):
    ids = batch["input_ids"].to(device)
    lmask = batch["latent_mask"].to(device)
    orig = batch["original_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    pls = batch["prompt_lengths"]
    B = ids.shape[0]

    total_loss = torch.tensor(0.0, device=device)
    m_ce, m_anch, m_exit = 0.0, 0.0, 0.0
    n_ce, n_anch, n_exit, anch_ok = 0, 0, 0, 0

    for b in range(B):
        sl = attn[b].sum().item()
        ids_b = ids[b, :sl]
        lm_b = lmask[b, :sl]
        orig_b = orig[b, :sl]
        pl = pls[b]

        # Build segments
        segs = []
        i = 0
        while i < sl:
            il = lm_b[i].item()
            j = i + 1
            while j < sl and lm_b[j].item() == il:
                j += 1
            segs.append((i, j, il))
            i = j

        pkv = None
        h = None
        s_loss = torch.tensor(0.0, device=device)
        s_n = 0

        for ss, se, is_lat in segs:
            slen = se - ss
            if not is_lat:
                seg_ids = ids_b[ss:se].unsqueeze(0)
                kvl = pkv[0][0].shape[2] if pkv else 0
                sa = torch.ones(1, kvl + slen, dtype=torch.long, device=device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(input_ids=seg_ids, attention_mask=sa,
                               past_key_values=pkv, use_cache=True, output_hidden_states=True)
                pkv = out.past_key_values
                h = out.hidden_states[-1][0, -1, :]
                logits = out.logits[0]
                for k in range(slen):
                    pos = ss + k
                    tgt_pos = pos + 1
                    if pos >= pl and tgt_pos < sl and not lm_b[tgt_pos].item():
                        ce = F.cross_entropy(logits[k:k+1].float(), ids_b[tgt_pos:tgt_pos+1])
                        s_loss = s_loss + ce
                        s_n += 1; m_ce += ce.item(); n_ce += 1
                if se < sl and not lm_b[se].item() and ss + slen - 1 >= pl:
                    ce = F.cross_entropy(logits[-1:].float(), ids_b[se:se+1])
                    s_loss = s_loss + ce
                    s_n += 1; m_ce += ce.item(); n_ce += 1
                del out
            else:
                lat_count = 0
                for k in range(slen):
                    pos = ss + k
                    if lat_count > 0 and lat_count % max_bptt == 0:
                        h = h.detach()
                        pkv = tuple(tuple(t.detach() for t in lkv) for lkv in pkv)
                    if h is not None:
                        emb = bridge(h).unsqueeze(0).unsqueeze(0)
                    else:
                        emb = model.get_input_embeddings()(ids_b[pos:pos+1].unsqueeze(0))
                    kvl = pkv[0][0].shape[2] if pkv else 0
                    sa = torch.ones(1, kvl + 1, dtype=torch.long, device=device)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out = model(inputs_embeds=emb, attention_mask=sa,
                                   past_key_values=pkv, use_cache=True, output_hidden_states=True)
                    pkv = out.past_key_values
                    h = out.hidden_states[-1][0, -1, :]
                    lat_logits = out.logits[0, -1, :]
                    lat_count += 1
                    if use_anchor and pos + 1 < sl:
                        atgt = orig_b[pos + 1] # 目标 Token 的 ID
                        
                        # 1. 提取目标 Token 真实的“输入 Embedding”作为绝对锚点
                        with torch.no_grad():
                            target_embed = model.get_input_embeddings()(atgt.unsqueeze(0)) # [1, hidden_size]
                        
                        # 2. 我们要求 LatentBridge 的输出 (emb) 必须指向这个真实的输入向量
                        # emb 的 shape 是 [1, 1, hidden_size]，使用 squeeze 展平
                        current_emb = emb.squeeze() # [hidden_size]
                        target_embed = target_embed.squeeze() # [hidden_size]
                        
                        # 3. 计算 Cosine Embedding Loss (1.0 表示完美同向，Loss 为 0)
                        ace = 1.0 - F.cosine_similarity(current_emb.float(), target_embed.float(), dim=-1)
                        
                        s_loss = s_loss + beta * ace
                        s_n += 1; m_anch += ace.item(); n_anch += 1
                        
                        # 4. 监控指标修改：不再使用分类准确率，而是累加余弦相似度
                        with torch.no_grad():
                            anch_ok += (1.0 - ace.item()) # 记录相似度，范围约为 [-1, 1]
                    if k == slen - 1 and se < sl and not lm_b[se].item():
                        etgt = ids_b[se]
                        ece = F.cross_entropy(lat_logits.float().unsqueeze(0), etgt.unsqueeze(0))
                        s_loss = s_loss + exit_weight * ece
                        s_n += 1; m_exit += ece.item(); n_exit += 1
                    del out

        if s_n > 0:
            total_loss = total_loss + s_loss / s_n
        del pkv
        torch.cuda.empty_cache()

    total_loss = total_loss / B
    return total_loss, {
        "ce_loss": m_ce / max(n_ce, 1), "anchor_loss": m_anch / max(n_anch, 1),
        "total_loss": total_loss.item(), "exit_loss": m_exit / max(n_exit, 1),
        "anchor_acc": anch_ok / max(n_anch, 1),
        "n_explicit": n_ce, "n_latent": n_anch, "n_exits": n_exit,
    }


# ============================================================
# Trainer with resume support
# ============================================================

class AnchorTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0")

        logger.info(f"Loading: {args.model}")
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", trust_remote_code=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.use_cache = True

        # F4: Freeze embedding + LM Head
        self.model.get_input_embeddings().requires_grad_(False)
        self.model.get_output_embeddings().requires_grad_(False)
        logger.info("  Frozen: input_embeddings + lm_head")

        self.bridge = LatentBridge(self.model.config.hidden_size).to(
            device=self.device, dtype=torch.bfloat16)
        self.bridge.calibrate_embed_norm(self.model)

        trainable_model_params = [p for p in self.model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable_model_params)
        n_bridge = sum(p.numel() for p in self.bridge.parameters())
        n_total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"  Trainable: {n_trainable/1e6:.1f}M model + {n_bridge/1e6:.1f}M bridge "
                     f"/ {n_total/1e6:.1f}M total ({(n_trainable+n_bridge)/n_total:.1%})")

        self.optimizer = AdamW([
            {"params": trainable_model_params, "lr": args.lr, "weight_decay": 0.01},
            {"params": list(self.bridge.parameters()), "lr": args.bridge_lr, "weight_decay": 0.01},
        ], betas=(0.9, 0.95), eps=1e-8)

        self.total_steps = args.steps_per_stage * args.num_stages
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, max(int(self.total_steps * 0.05), 10), self.total_steps)

        logger.info(f"Loading data: {args.data}")
        self.all_samples = torch.load(args.data, weights_only=False)
        logger.info(f"  {len(self.all_samples)} samples")
        self.use_anchor = not args.no_anchor
        os.makedirs(args.output_dir, exist_ok=True)

        # Resume state
        self.start_stage = 0
        self.start_step_in_stage = 0
        self.global_step = 0
        self.log_history = []

    def _save_checkpoint(self, stage, step_in_stage, tag=None):
        """Save full training state for resume."""
        d = os.path.join(self.args.output_dir, tag or f"stage-{stage}")
        os.makedirs(d, exist_ok=True)
        self.model.save_pretrained(d, safe_serialization=True)
        self.tokenizer.save_pretrained(d)
        torch.save(self.bridge.state_dict(), os.path.join(d, "bridge.pt"))
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "stage": stage,
            "step_in_stage": step_in_stage,
            "global_step": self.global_step,
            "log_history": self.log_history[-50:],
        }, os.path.join(d, "training_state.pt"))
        logger.info(f"  Saved: {d} (stage={stage}, step={step_in_stage}, global={self.global_step})")

    def resume_from(self, resume_dir):
        """Load training state from checkpoint."""
        # Load bridge
        bp = os.path.join(resume_dir, "bridge.pt")
        if os.path.exists(bp):
            self.bridge.load_state_dict(torch.load(bp, map_location=self.device, weights_only=False))
            logger.info(f"  Loaded bridge: {bp}")

        # Load training state
        sp = os.path.join(resume_dir, "training_state.pt")
        if os.path.exists(sp):
            state = torch.load(sp, map_location=self.device, weights_only=False)
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.start_stage = state["stage"]
            self.start_step_in_stage = state["step_in_stage"]
            self.global_step = state["global_step"]
            self.log_history = state.get("log_history", [])
            logger.info(f"  Resumed: stage={self.start_stage}, step_in_stage={self.start_step_in_stage}, "
                         f"global_step={self.global_step}")
        else:
            logger.warning(f"  No training_state.pt in {resume_dir}, starting from scratch")

    def train(self):
        a = self.args
        mode = "B (anchor)" if self.use_anchor else "A (no anchor)"
        logger.info("=" * 60)
        logger.info(f"Mode: {mode}, beta={a.beta} (fixed), bridge_lr={a.bridge_lr}")
        logger.info(f"  Stages: {a.num_stages}, Steps/stage: {a.steps_per_stage}")
        logger.info(f"  Resume from: stage={self.start_stage}, step={self.start_step_in_stage}")
        logger.info("=" * 60)

        for stage in range(self.start_stage, a.num_stages):
            logger.info(f"\n{'='*40} Stage {stage} {'='*40}")

            bs = a.batch_size_stage0 if stage == 0 else a.batch_size
            ga = max(1, (a.batch_size * a.grad_accum) // bs) if stage == 0 else a.grad_accum

            ds = CurriculumDataset(self.all_samples, stage,
                                   self.tokenizer.pad_token_id or 0, a.max_seq_len)
            if not ds.samples:
                logger.warning(f"Stage {stage} empty, skipping")
                continue
            logger.info(f"  {len(ds.samples)} samples, batch={bs}×{ga}={bs*ga}")

            dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=False,
                           collate_fn=lambda b: collate_fn(b, self.tokenizer.pad_token_id or 0))
            self.model.train()
            self.bridge.train()
            it = iter(dl)
            am, ac = {}, 0

            # Skip already-done steps if resuming mid-stage
            skip_steps = self.start_step_in_stage if stage == self.start_stage else 0
            done = skip_steps
            if skip_steps > 0:
                logger.info(f"  Skipping {skip_steps} already-done steps")

            while done < a.steps_per_stage:
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(dl)
                    batch = next(it)

                # F6: OOM fallback
                try:
                    if stage == 0:
                        loss, met = batched_forward_stage0(self.model, batch, self.device)
                    else:
                        loss, met = chunked_forward(
                            self.model, self.bridge, batch, self.device,
                            self.use_anchor, a.beta, a.exit_weight, a.max_bptt_steps)
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"  OOM at step {done}, clearing cache and retrying with batch[0:1]")
                    torch.cuda.empty_cache()
                    gc.collect()
                    self.optimizer.zero_grad()
                    # Retry with single sample
                    mini = {k: v[:1] if isinstance(v, torch.Tensor) else [v[0]]
                            for k, v in batch.items()}
                    if stage == 0:
                        loss, met = batched_forward_stage0(self.model, mini, self.device)
                    else:
                        loss, met = chunked_forward(
                            self.model, self.bridge, mini, self.device,
                            self.use_anchor, a.beta, a.exit_weight, a.max_bptt_steps)
                    am, ac = {}, 0  # Reset accumulation after OOM

                (loss / ga).backward()
                for k, v in met.items():
                    am[k] = am.get(k, 0) + v
                ac += 1

                if ac >= ga:
                    gn = torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.bridge.parameters()),
                        a.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    avg = {k: v / ac for k, v in am.items()}
                    avg["grad_norm"] = gn.item() if isinstance(gn, torch.Tensor) else gn
                    avg["lr"] = self.scheduler.get_last_lr()[0]
                    avg["stage"] = stage
                    avg["step"] = self.global_step
                    avg["step_in_stage"] = done

                    if self.global_step % a.log_interval == 0:
                        logger.info(
                            f"Step {self.global_step} [S{stage} {done}/{a.steps_per_stage}] | "
                            f"CE:{avg['ce_loss']:.4f} Anch:{avg['anchor_loss']:.4f} "
                            f"Exit:{avg['exit_loss']:.4f} Tot:{avg['total_loss']:.4f} | "
                            f"AnchAcc:{avg['anchor_acc']:.1%} GN:{avg['grad_norm']:.2f} "
                            f"LR:{avg['lr']:.2e} | GPU:{torch.cuda.max_memory_allocated()//(1024**2)}MB")

                    self.log_history.append(avg)
                    am, ac = {}, 0
                    self.global_step += 1
                    done += 1
                    gc.collect()
                    torch.cuda.empty_cache()

                    # Periodic save every 50 steps within stage (for resume)
                    if done % 50 == 0 and done < a.steps_per_stage:
                        self._save_checkpoint(stage, done, f"stage-{stage}-step-{done}")

            # End of stage save
            self._save_checkpoint(stage, a.steps_per_stage, f"stage-{stage}")
            # Reset resume offset for next stages
            self.start_step_in_stage = 0

        # Final save
        self._save_checkpoint(a.num_stages - 1, a.steps_per_stage, "final")
        with open(os.path.join(a.output_dir, "training_log.json"), "w") as f:
            json.dump(self.log_history, f, indent=2)
        logger.info("Training complete.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--data", default="./anchor_data/curriculum_data.pt")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--no_anchor", action="store_true")
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--exit_weight", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--bridge_lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--batch_size_stage0", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--steps_per_stage", type=int, default=200)
    p.add_argument("--num_stages", type=int, default=5)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--max_bptt_steps", type=int, default=4)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--log_interval", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint dir to resume from")
    a = p.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    # If resuming, load model from resume checkpoint instead of base
    if a.resume:
        logger.info(f"Resuming from: {a.resume}")
        a.model = a.resume  # Load model weights from resume dir

    trainer = AnchorTrainer(a)

    if a.resume:
        trainer.resume_from(a.resume)

    trainer.train()


if __name__ == "__main__":
    main()