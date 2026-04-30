"""
SPARK v2 - GRPO 训练器 (修复版 - 最终极品版)

核心修复点:
  1. K1-in-reward: 无偏的 sum() 估计器。
  2. Fixed Advantage: PPO 循环外预先计算 Advantage。
  3. Micro-Batching: 彻底解决 OOM，引入 mb_size 将大 Batch 拆分计算及反向传播。
  4. FSDP Safe Embedding 保护: 通过优化器 no_decay 组彻底避免触碰 shard gradient。
"""

import contextlib

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from typing import List, Dict, Optional, Tuple
from accelerate import Accelerator
import logging
import time
import os
import json
from transformers import get_cosine_schedule_with_warmup

from config import SPARKConfig, Phase1TrainConfig
from rollout import (
    generate_trajectories,
    compute_log_probs_single_no_grad,
    evaluate_accuracy,
)
from reward import compute_total_reward, compute_group_advantages

logger = logging.getLogger(__name__)


class GRPOTrainer:

    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        skip_token_id: int,
        config: SPARKConfig,
        dataset,
        eval_datasets: Optional[Dict] = None,
        accelerator: Optional[Accelerator] = None,
    ):
        self.tokenizer = tokenizer
        self.skip_token_id = skip_token_id
        self.config = config
        self.train_config = config.phase1
        self.dataset = dataset
        self.eval_datasets = eval_datasets or {}

        if accelerator is None:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=1,
                mixed_precision="bf16",
            )
        else:
            self.accelerator = accelerator

        if config.phase1.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "config"):
                model.config.use_cache = False
            logger.info("Gradient checkpointing ENABLED")

        optimizer = self._build_optimizer(model)

        # ★ 新增：余弦退火学习率调度器 (5% Warmup) ★
        warmup_steps = int(config.phase1.num_train_steps * 0.05)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=config.phase1.num_train_steps
        )

        # ★ 修复 1: 将 ref_model 也交给 accelerator.prepare 进行 FSDP 切片，极大节省显存
        self.ref_model = ref_model
        if self.ref_model is not None:
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False  # 必须在 prepare 之前设为 False

            self.model, self.ref_model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                model, self.ref_model, optimizer, lr_scheduler
            )
            logger.info("✅ Reference model is loaded and prepared (FSDP Sharded) for K1 penalty calculation.")
        else:
            self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                model, optimizer, lr_scheduler
            )
            logger.warning("🚨 WARNING: ref_model is None! K1 KL penalty will be ZERO.")


        self.global_step = 0
        self.log_history = []
        self.is_main = self.accelerator.is_main_process

        self.use_fsdp = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        if self.use_fsdp and self.is_main:
            logger.info("FSDP mode — will sync trajectory counts across ranks")

        self.loss_token_norm = float(config.model.max_new_tokens)

    def _build_optimizer(self, model) -> AdamW:
        cfg = self.train_config
        # ★ 修复 1：将 embed_tokens 加入 no_decay，天然保护 <SKIP> 不被权重衰减腐蚀
        no_decay = ["bias", "layer_norm", "layernorm", "rmsnorm", "embed_tokens", "wte", "lm_head"]
        groups = [
            {
                "params": [p for n, p in model.named_parameters()
                           if p.requires_grad and not any(nd in n.lower() for nd in no_decay)],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if p.requires_grad and any(nd in n.lower() for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(groups, lr=cfg.learning_rate,
                     betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adam_epsilon)

    def _sync_trajectory_count(self, local_count: int) -> int:
        if not self.use_fsdp or not dist.is_initialized():
            return local_count
        device = self.accelerator.device
        t = torch.tensor([local_count], dtype=torch.long, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.item()

    def _compute_log_probs_batched(self, trajs: List[Dict], model, no_grad: bool = False) -> List[torch.Tensor]:
        if not trajs:
            return []

        device = self.accelerator.device
        max_len = max(t["full_ids"].shape[-1] for t in trajs)

        padded_ids, padded_masks = [], []
        for t in trajs:
            seq_len = t["full_ids"].shape[-1]
            pad_len = max_len - seq_len
            p_ids = F.pad(t["full_ids"].squeeze(0), (0, pad_len), value=self.tokenizer.pad_token_id)
            p_mask = F.pad(t["full_attention_mask"].squeeze(0), (0, pad_len), value=0)
            padded_ids.append(p_ids)
            padded_masks.append(p_mask)

        input_ids = torch.stack(padded_ids).to(device)
        attention_mask = torch.stack(padded_masks).to(device)

        context = torch.no_grad() if no_grad else torch.enable_grad()
        with context:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            log_probs_all = F.log_softmax(outputs.logits.float(), dim=-1)

        result_lps = []
        for i, t in enumerate(trajs):
            prompt_len = t["prompt_length"]
            seq_len = t["full_ids"].shape[-1]
            response_ids = t["response_ids"].to(device)
            resp_lp_all = log_probs_all[i, prompt_len - 1 : seq_len - 1, :]
            lp = resp_lp_all.gather(dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1)
            result_lps.append(lp)

        return result_lps


    def train_step(self) -> Dict:
        cfg = self.train_config
        self.model.train()
        step_start = time.time()

        # ========================================
        # Phase 1 & 2: 生成轨迹与初始 Reward
        # ========================================
        batch = self.dataset.sample_batch(cfg.batch_size)
        grouped_trajectories = generate_trajectories(
            model=self.model, tokenizer=self.tokenizer, prompts=batch,
            group_size=cfg.group_size, temperature=cfg.temperature,
            top_p=cfg.top_p, max_new_tokens=self.config.model.max_new_tokens,
            data_config=self.config.data, blocked_token_ids=[self.skip_token_id]
        )
        torch.cuda.empty_cache()

        all_trajs, all_rewards, group_boundaries = [], [], []
        batch_correct, batch_total, offset = 0, 0, 0

        for group in grouped_trajectories:
            for traj in group:
                r = compute_total_reward(traj, phase=1, config=self.config.reward)
                all_trajs.append(traj)
                all_rewards.append(r)
                if traj["is_correct"]: batch_correct += 1
                batch_total += 1
            group_boundaries.append((offset, offset + len(group)))
            offset += len(group)

        local_total_num = len(all_trajs)
        synced_total_num = self._sync_trajectory_count(local_total_num)
        if synced_total_num == 0:
            return {"step": self.global_step, "loss": 0.0, "skipped": True}

            # =========================================================
            # ★ 修复：防止极端情况下某张卡全部 OOM 导致 all_trajs[0] 越界崩溃 ★
            # =========================================================
        if not all_trajs:
            # 造一条极简的假轨迹用于 FSDP 占位通信
            device = self.accelerator.device
            dummy_traj = {
                "full_ids": torch.tensor([[0, 0]], device=device),
                "full_attention_mask": torch.tensor([[1, 1]], device=device),
                "prompt_length": 1,
                "response_ids": torch.tensor([0], device=device),
                "response_length": 1,
                "is_correct": False
            }
            safe_dummy_pool = [dummy_traj]
        else:
            safe_dummy_pool = all_trajs

            # 使用 safe_dummy_pool[0] 补齐
        compute_trajs = all_trajs + [safe_dummy_pool[0]] * (synced_total_num - local_total_num)
            # =========================================================


        # ========================================
        # Phase 3: Rollout 推断 (仅计算 Log Probs)
        # ========================================
        old_log_probs_list = []
        ref_log_probs_list = []
        mb_size_rollout = 2  # Rollout 没有反向传播，Batch 可以稍大

        # 1. 纯粹的推断循环，只负责生成 log_probs
        for i in range(0, synced_total_num, mb_size_rollout):
            chunk = compute_trajs[i:i + mb_size_rollout]
            old_log_probs_list.extend(self._compute_log_probs_batched(chunk, self.model, no_grad=True))

            if self.ref_model is not None and cfg.kl_coeff > 0:
                ref_log_probs_list.extend(self._compute_log_probs_batched(chunk, self.ref_model, no_grad=True))
            else:
                ref_log_probs_list.extend([None] * len(chunk))

        # ========================================
        # Phase 3.5: ★ Trajectory-level K1 KL & Advantage ★
        # ========================================
        total_kl_sum = 0.0
        adjusted_rewards = list(all_rewards)  # 默认不调整

        # ★ 修复 Bug 2 & Bug 3：计算整条轨迹的平均 KL，并直接在 Reward 上扣除 ★
        if self.ref_model is not None and cfg.kl_coeff > 0:
            for idx in range(local_total_num):
                old_lp = old_log_probs_list[idx]
                ref_lp = ref_log_probs_list[idx]
                if ref_lp is not None:
                    # 计算整条轨迹的平均 KL (标量) -> 这是 Dr. GRPO K1 的核心
                    traj_kl = (old_lp - ref_lp.to(old_lp.device)).sum().item()
                    total_kl_sum += traj_kl
                    # R_adj = R - β * K1_traj
                    adjusted_rewards[idx] = all_rewards[idx] - cfg.kl_coeff * traj_kl

        # ★ 基于扣除 KL 后的 Adjusted Reward 计算 Group Advantage ★
        fixed_advantages = []
        for start, end in group_boundaries:
            g_rewards = adjusted_rewards[start:end]
            fixed_advantages.extend(compute_group_advantages(g_rewards, use_std_norm=False))
        
        # Dummy 轨迹补齐
        fixed_advantages.extend([0.0] * (synced_total_num - local_total_num))

        torch.cuda.empty_cache()

        # ========================================
        # Phase 4: Multi-Epoch PPO (使用严格 Micro-Batching 防 OOM)
        # ========================================
        #T_norm = self.loss_token_norm
        total_abs_loss = 0.0
        total_clip_frac = 0.0
        last_grad_norm = 0.0
        num_nonzero_adv = sum(1 for a in fixed_advantages[:local_total_num] if abs(a) > 1e-6)

        mb_size_ppo = 2  # ★ 修复 2：极其关键的 PPO Micro-batch Size，防止带梯度的计算图爆掉显存

        for epoch in range(cfg.ppo_epochs):
            self.optimizer.zero_grad()
            ep_abs_loss, ep_clipped, ep_tokens = 0.0, 0, 0

            # 分批次计算 Forward + 立即 Backward 以释放显存
            # ★ 修复 3 (终极性能优化): 计算微批次总数，利用 no_sync 消除 FSDP 通信风暴 ★
            num_chunks = (synced_total_num + mb_size_ppo - 1) // mb_size_ppo

            # 分批次计算 Forward + 立即 Backward 以释放显存
            for chunk_idx, i in enumerate(range(0, synced_total_num, mb_size_ppo)):
                is_last_chunk = (chunk_idx == num_chunks - 1)

                # 前 N-1 个 chunk 禁用梯度同步，最后 1 个 chunk 触发一次全局同步
                sync_context = contextlib.nullcontext() if is_last_chunk else self.accelerator.no_sync(self.model)

                with sync_context:
                    chunk_trajs = compute_trajs[i:i + mb_size_ppo]
                    curr_lps_chunk = self._compute_log_probs_batched(chunk_trajs, self.model, no_grad=False)

                    chunk_loss = 0.0
                    for j, curr_lp in enumerate(curr_lps_chunk):
                        idx = i + j
                        is_valid = idx < local_total_num
                        old_lp = old_log_probs_list[idx].to(curr_lp.device)
                        
                        # ★ 直接拿 Phase 3.5 算好的标量 Advantage ★
                        adv_val = fixed_advantages[idx]
                        
                        # 构建与 ratio 形状相同的 Advantage Tensor，并限制极值防 NaN
                        adv_tensor = torch.full_like(old_lp, adv_val)
                        adv_tensor = torch.clamp(adv_tensor, min=-10.0, max=10.0)

                        # 计算 PPO 概率比率
                        log_ratio = curr_lp - old_lp
                        # 防护：避免极端的策略偏离导致 exp(log_ratio) 产生 inf
                        log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
                        ratio = torch.exp(log_ratio)

                        surr1 = ratio * adv_tensor
                        surr2 = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv_tensor

                        # ★ 修复 Bug 1：使用全局常数 MAX_TOKENS 进行绝对的 K1 归一化 ★
                        # 绝对不能用 ratio.numel()，否则会奖励冗余废话！
                        T_norm = self.loss_token_norm 
                        policy_loss = -torch.min(surr1, surr2).sum() / T_norm
                        
                        traj_loss = policy_loss / max(local_total_num, 1)

                        if not is_valid:
                            traj_loss = traj_loss * 0.0

                        chunk_loss = chunk_loss + traj_loss

                        if is_valid:
                            ep_abs_loss += abs(policy_loss.item())
                            n_tok = ratio.numel()
                            n_clip = ((ratio < 1.0 - cfg.clip_range) | (ratio > 1.0 + cfg.clip_range)).sum().item()
                            ep_clipped += n_clip
                            ep_tokens += n_tok

                        if is_valid:
                            ep_abs_loss += abs(policy_loss.item())
                            n_tok = ratio.numel()
                            n_clip = ((ratio < 1.0 - cfg.clip_range) | (ratio > 1.0 + cfg.clip_range)).sum().item()
                            ep_clipped += n_clip
                            ep_tokens += n_tok

                    # 立即反向传播释放计算图 (前 N-1 次本地累加无通信，最后 1 次全网同步)
                    self.accelerator.backward(chunk_loss)


            gn = self.accelerator.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
            if isinstance(gn, torch.Tensor):
                gn = gn.item()
            last_grad_norm = gn

            # 删除手改梯度的代码，因为已通过 no_decay 免除了权重衰减腐蚀
            self.optimizer.step()

            total_abs_loss += ep_abs_loss
            total_clip_frac += ep_clipped / max(ep_tokens, 1)

        torch.cuda.empty_cache()

        # ★ 新增：推进一步学习率调度器 ★
        self.lr_scheduler.step()

        # ========================================
        # Metrics
        # ========================================
        t_elapsed = time.time() - step_start
        n_ep = cfg.ppo_epochs
        valid_advs = fixed_advantages[:local_total_num]
        mean_abs_adv = sum(abs(a) for a in valid_advs) / max(len(valid_advs), 1)
        valid_rewards = all_rewards[:local_total_num]
        mean_kl = total_kl_sum / max(local_total_num, 1)

        metrics = {
            "step": self.global_step,
            "learning_rate": self.lr_scheduler.get_last_lr()[0],
            "loss": total_abs_loss / max(local_total_num * n_ep, 1),
            "kl": mean_kl,
            "grad_norm": last_grad_norm,
            "clip_frac": total_clip_frac / n_ep,
            "mean_reward": sum(valid_rewards) / len(valid_rewards) if valid_rewards else 0.0,
            "batch_accuracy": batch_correct / batch_total if batch_total > 0 else 0.0,
            "mean_abs_advantage": mean_abs_adv,
            "nonzero_adv_ratio": num_nonzero_adv / max(local_total_num, 1),
            "mean_response_length": sum(t["response_length"] for t in all_trajs) / max(local_total_num, 1),
            "num_trajectories": local_total_num,
            "step_time": t_elapsed,
        }

        # ==============================================================
        # ★ 修复 1：分布式同步关键 Metrics，确保所有的控制流判断在各卡完全一致 ★
        # ==============================================================
        if self.use_fsdp and dist.is_initialized():
            # 提取需要同步的四个关键指标
            keys_to_sync = ["batch_accuracy", "mean_reward", "loss", "kl"]
            sync_tensor = torch.tensor([metrics[k] for k in keys_to_sync], device=self.accelerator.device)
            # 全局求和后取平均
            dist.all_reduce(sync_tensor, op=dist.ReduceOp.SUM)
            sync_tensor = sync_tensor / self.accelerator.num_processes
            # 写回 metrics
            for i, k in enumerate(keys_to_sync):
                metrics[k] = sync_tensor[i].item()
        # ==============================================================

        self.global_step += 1
        return metrics


    def train(self, resume_step: int = 0) -> Dict:
        cfg = self.train_config

        if self.is_main:
            if resume_step > 0:
                logger.info(f"Resuming from step {resume_step}")
            else:
                logger.info(f"Starting Phase 1 GRPO: {cfg.num_train_steps} steps")
            logger.info(f"  B={cfg.batch_size}, G={cfg.group_size}, lr={cfg.learning_rate}")
            logger.info(f"  ppo_epochs={cfg.ppo_epochs}, clip={cfg.clip_range}")
            logger.info(f"  kl_coeff={cfg.kl_coeff} (K1-in-reward)")
            logger.info(f"  max_grad_norm={cfg.max_grad_norm}")
            logger.info(f"  loss_norm=Dr.GRPO K1 (T_max={self.loss_token_norm})")
            logger.info(f"  advantage=Dr.GRPO (group-level, no std)")
            logger.info(f"  GPUs: {self.accelerator.num_processes}, FSDP: {self.use_fsdp}")
            os.makedirs(cfg.output_dir, exist_ok=True)
            os.makedirs(cfg.log_dir, exist_ok=True)

        best_accuracy = getattr(self, '_best_accuracy', 0.0)
        self._best_accuracy = best_accuracy
        self.optimizer.zero_grad()

        baseline_accuracy = None
        patience_counter = 0
        patience_limit = 3
        consecutive_zero_acc = 0

        for step in range(resume_step, cfg.num_train_steps):
            metrics = self.train_step()
            if metrics.get("skipped"):
                continue

            gn = metrics.get('grad_norm', 0.0)
            if gn > 10.0 and self.is_main:
                logger.warning(f"  ⚠ GradNorm spike: {gn:.1f} at step {step}")

            if metrics['batch_accuracy'] == 0.0 and metrics['mean_reward'] <= -0.9:
                consecutive_zero_acc += 1
            else:
                consecutive_zero_acc = 0
            if consecutive_zero_acc >= 5:
                if self.is_main:
                    logger.error(f"  ✖ Model collapsed at step {step}. Emergency save.")
                self._save(step, tag="emergency-collapse")
                break

            if self.is_main and step % cfg.log_interval == 0:
                logger.info(
                    f"Step {step}/{cfg.num_train_steps} | "
                    f"Loss:{metrics['loss']:.4f} "
                    f"KL:{metrics['kl']:.4f} "
                    f"Clip:{metrics['clip_frac']:.3f} "
                    f"GN:{metrics['grad_norm']:.3f} "
                    f"Rew:{metrics['mean_reward']:.3f} "
                    f"Acc:{metrics['batch_accuracy']:.3f} "
                    f"|Adv|:{metrics['mean_abs_advantage']:.3f} "
                    f"Len:{metrics['mean_response_length']:.0f} "
                    f"T:{metrics['step_time']:.0f}s"
                )
                self.log_history.append(metrics)

            if step > 0 and step % cfg.eval_interval == 0:
                eval_results = self._run_evaluation()
                gsm8k_acc = eval_results.get("gsm8k", {}).get("accuracy", 0.0)

                # ==============================================================
                # ★ 修复 2：分布式同步评估分数，防止触发 Save 时不同卡状态不一致死锁 ★
                # ==============================================================
                if self.use_fsdp and dist.is_initialized():
                    acc_tensor = torch.tensor([gsm8k_acc], device=self.accelerator.device)
                    dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
                    # 获得全集群平均准确率
                    gsm8k_acc = acc_tensor.item() / self.accelerator.num_processes
                # ==============================================================


                if self.is_main:
                    for name, result in eval_results.items():
                        logger.info(f"  Eval {name}: {result['accuracy']:.4f}")

                if baseline_accuracy is None:
                    baseline_accuracy = gsm8k_acc
                    if self.is_main:
                        logger.info(f"  Baseline: {baseline_accuracy:.4f}")

                if gsm8k_acc > best_accuracy:
                    best_accuracy = gsm8k_acc
                    self._best_accuracy = best_accuracy
                    self._save(step, tag="best", extra={"accuracy": gsm8k_acc})
                    patience_counter = 0

                if baseline_accuracy and baseline_accuracy > 0:
                    if gsm8k_acc < baseline_accuracy * 0.7:
                        patience_counter += 1
                        if self.is_main:
                            logger.warning(
                                f"  ⚠ Degraded: {gsm8k_acc:.3f} < "
                                f"{baseline_accuracy*0.7:.3f}. "
                                f"Patience {patience_counter}/{patience_limit}"
                            )
                        if patience_counter >= patience_limit:
                            if self.is_main:
                                logger.error(f"  ✖ Early stop. Best={best_accuracy:.4f}")
                            self._save(step, tag="early-stop")
                            break
                    else:
                        patience_counter = 0

            if step > 0 and step % cfg.save_interval == 0:
                self._save(step)

        self._save(cfg.num_train_steps, tag="final")
        final_eval = self._run_evaluation()

        if self.is_main:
            logger.info("=" * 60)
            logger.info("Phase 1 Complete!")
            for name, result in final_eval.items():
                logger.info(f"  {name}: {result['accuracy']:.4f}")
            logger.info(f"  Best GSM8K: {best_accuracy:.4f}")
            logger.info("=" * 60)
            log_path = os.path.join(cfg.log_dir, "training_log.json")
            with open(log_path, "w") as f:
                json.dump(self.log_history, f, indent=2)
            return {"final_eval": final_eval, "best_accuracy": best_accuracy}
        return {}

    def _run_evaluation(self) -> Dict:
        results = {}
        for name, eval_data in self.eval_datasets.items():
            problems = eval_data.problems if hasattr(eval_data, 'problems') else eval_data
            result = evaluate_accuracy(
                model=self.model, tokenizer=self.tokenizer,
                problems=problems, data_config=self.config.data,
                temperature=0.0, max_problems=200,
            )
            results[name] = result
        return results

    def _save(self, step, tag=None, extra=None):
            save_dir = os.path.join(self.train_config.output_dir, tag or f"step-{step}")
            os.makedirs(save_dir, exist_ok=True)

            unwrapped = self.accelerator.unwrap_model(self.model)
            # 获取全局聚合后的完整参数字典
            state_dict = self.accelerator.get_state_dict(self.model, unwrap=False)

            if self.is_main and state_dict is not None:
                # 1. 保存模型和 Tokenizer
                unwrapped.save_pretrained(
                    save_dir, state_dict=state_dict, safe_serialization=True,
                )
                self.tokenizer.save_pretrained(save_dir)

                # 2. ★ 修复 Bug 4：从完整安全的 state_dict 中提取 skip_adapter 权重 ★
                adapter_state = {k.replace("skip_adapter.", ""): v.cpu() 
                                for k, v in state_dict.items() if "skip_adapter." in k}
                
                if adapter_state:
                    adapter_path = os.path.join(save_dir, "skip_adapter.pt")
                    torch.save(adapter_state, adapter_path)

                # 3. 保存训练元数据
                meta = {
                    "step": step,
                    "global_step": self.global_step,
                    "best_accuracy": getattr(self, '_best_accuracy', 0.0),
                    **(extra or {}),
                }
                with open(os.path.join(save_dir, "training_state.json"), "w") as f:
                    json.dump(meta, f, indent=2)
                logger.info(f"Checkpoint saved: {save_dir}")

            self.accelerator.wait_for_everyone()

    def set_resume_state(self, resume_step: int, best_accuracy: float = 0.0):
        self.global_step = resume_step
        self._best_accuracy = best_accuracy

        # ==========================================================
        # ★ 修复：让学习率调度器“追赶”到中断时的进度，防止重新触发 Warmup ★
        # ==========================================================
        if hasattr(self, "lr_scheduler") and self.lr_scheduler is not None:
            for _ in range(resume_step):
                self.lr_scheduler.step()
        # ==========================================================


        if self.is_main:
            logger.info(f"Resume: step={resume_step}, best_acc={best_accuracy:.4f}")