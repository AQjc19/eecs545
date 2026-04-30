"""
SPARK v2 - 轨迹生成 + Log Prob 计算

FSDP 兼容策略:
  - generate(): 
    1) summon_full_params 恢复完整参数
    2) 临时把所有 FSDP(layer) 替换为裸 layer → forward 不经过 FSDP hooks
    3) 各 rank 独立生成，不触发任何 all_gather → 不死锁
  - compute_log_probs: 用 FSDP-wrapped model（所有 rank 调用次数已同步）
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from contextlib import contextmanager, nullcontext
from transformers import AutoTokenizer
import logging

from config import DataConfig
from data_utils import extract_model_answer, check_answer, build_prompt

logger = logging.getLogger(__name__)


# ============================================================
# FSDP 工具函数
# ============================================================

def _is_fsdp_model(model) -> bool:
    """检测模型是否被 FSDP 包裹"""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        if isinstance(model, FSDP):
            return True
        inner = getattr(model, "module", None)
        if inner is not None and isinstance(inner, FSDP):
            return True
        return False
    except ImportError:
        return False


def _get_fsdp_module(model):
    """获取最外层的 FSDP 模块"""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    if isinstance(model, FSDP):
        return model
    inner = getattr(model, "module", None)
    if inner is not None and isinstance(inner, FSDP):
        return inner
    return model


def _get_inner_model(model):
    """
    获取最内层的原始 CausalLM 模型（例如 Qwen2ForCausalLM）。
    只剥掉顶层的 wrapper（Accelerate → FSDP → module）。
    注意：内部的 decoder layers 可能仍然被 FSDP 包裹。
    """
    inner = model
    for _ in range(10):
        if hasattr(inner, '_fsdp_wrapped_module'):
            inner = inner._fsdp_wrapped_module
            continue
        if hasattr(inner, 'module'):
            inner = inner.module
            continue
        break
    return inner


def _get_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda")


# ============================================================
# ★ FSDP 核心修复：生成时临时解除所有 FSDP 包裹 ★
# ============================================================

@contextmanager
def _fsdp_generate_context(model):
    """
    FSDP 安全的 generate 上下文管理器。
    
    问题：Accelerate FSDP 不仅包裹最外层模型，还单独包裹每个 decoder layer：
    
        FSDP(Qwen2ForCausalLM)
          └→ model.layers = [
               FSDP(Qwen2DecoderLayer_0),   ← 每层都有 FSDP！
               FSDP(Qwen2DecoderLayer_1),
               ...
             ]
    
    generate() 里每生成一个 token 就做一次 forward，每次 forward 经过
    每个 FSDP(layer) 都触发 all_gather（集合通信）。
    不同 rank 生成不同长度序列 → all_gather 次数不同 → NCCL 死锁。
    
    修复：
    1) summon_full_params: 恢复所有参数到完整形状（集合操作，所有 rank 同步）
    2) 临时把模块树中所有 FSDP(child) 替换为 child._fsdp_wrapped_module
       → forward 不经过 FSDP.forward() → 不触发 all_gather
    3) 各 rank 可以独立生成不同长度的序列
    4) 退出时恢复 FSDP 包裹
    5) 退出 summon_full_params（集合操作，所有 rank 同步）
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    fsdp_module = _get_fsdp_module(model)

    # Step 1: summon_full_params (集合操作 — 所有 rank 必须同时进入)
    with FSDP.summon_full_params(fsdp_module, writeback=False):

        # Step 2: 获取内部的 CausalLM 模型
        inner_model = _get_inner_model(model)

        # Step 3: 递归替换所有 FSDP 子模块
        swapped = []  # [(parent_module, attr_name, original_fsdp_child)]

        def _swap_fsdp_children(module):
            """递归遍历模块树，把所有 FSDP(child) 替换为 child 本身"""
            for name, child in list(module.named_children()):
                if isinstance(child, FSDP):
                    # 取出 FSDP 内部的原始模块
                    unwrapped_child = child._fsdp_wrapped_module
                    # 记录以便恢复
                    swapped.append((module, name, child))
                    # 替换：通过 _modules dict 直接操作
                    # 这对 nn.Module, nn.ModuleList, nn.ModuleDict 都有效
                    module._modules[name] = unwrapped_child
                    # 继续递归处理 unwrapped 的子模块（它们内部可能也有 FSDP）
                    _swap_fsdp_children(unwrapped_child)
                else:
                    _swap_fsdp_children(child)

        _swap_fsdp_children(inner_model)

        if swapped:
            logger.debug(
                f"Temporarily unwrapped {len(swapped)} FSDP sub-modules for generation"
            )

        try:
            # Step 4: yield 干净的模型（无 FSDP hooks）
            yield inner_model
        finally:
            # Step 5: 恢复所有 FSDP 包裹（逆序恢复）
            for parent, name, original_fsdp in reversed(swapped):
                parent._modules[name] = original_fsdp

            if swapped:
                logger.debug(f"Restored {len(swapped)} FSDP sub-modules")


# ============================================================
# 轨迹生成
# ============================================================

@torch.no_grad()
def generate_trajectories(
    model,
    tokenizer: AutoTokenizer,
    prompts: List[Dict],
    group_size: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
    data_config: Optional[DataConfig] = None,
    blocked_token_ids: Optional[List[int]] = None,
) -> List[List[Dict]]:
    """为一批 prompt 生成多条轨迹

    Args:
        blocked_token_ids: 禁止生成的 token ID 列表（如 <SKIP>）
    """
    model.eval()
    if data_config is None:
        data_config = DataConfig()

    grouped_trajectories = []
    is_fsdp = _is_fsdp_model(model)

    if is_fsdp:
        # FSDP: 使用安全的生成上下文（summon_full_params + 解除子模块 FSDP）
        ctx = _fsdp_generate_context(model)
    else:
        # 单卡 / DDP: 直接用模型
        ctx = nullcontext(model)

    with ctx as gen_model:
        # 关闭 autocast 避免 bf16 NaN 累积
        with torch.amp.autocast('cuda', enabled=False):
            device = _get_device(gen_model)

            for prompt_data in prompts:
                question = prompt_data["question"]
                ground_truth = prompt_data["answer"]

                prompt_text = build_prompt(question, tokenizer, data_config)
                prompt_encoded = tokenizer(
                    prompt_text, return_tensors="pt", add_special_tokens=False,
                )
                prompt_ids = prompt_encoded["input_ids"].to(device)
                prompt_length = prompt_ids.shape[1]

                trajectories = []
                for g in range(group_size):
                    try:
                        gen_kwargs = dict(
                            input_ids=prompt_ids,
                            attention_mask=torch.ones_like(prompt_ids),
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=temperature,
                            top_p=top_p,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                        # Phase 1: 禁止生成 <SKIP>（保护其 embedding 不被梯度污染）
                        if blocked_token_ids:
                            gen_kwargs["bad_words_ids"] = [[t] for t in blocked_token_ids]
                        output = gen_model.generate(**gen_kwargs)
                        full_ids = output[0]
                    except torch.cuda.OutOfMemoryError:
                        logger.warning(f"OOM generating trajectory {g}, skipping")
                        torch.cuda.empty_cache()
                        continue
                    except RuntimeError as e:
                        if "inf" in str(e).lower() or "nan" in str(e).lower():
                            logger.warning(f"NaN/Inf in trajectory {g}, skipping")
                            continue
                        raise

                    response_ids = full_ids[prompt_length:]

                    # 截断到 EOS
                    eos_pos = (response_ids == tokenizer.eos_token_id).nonzero(
                        as_tuple=True
                    )[0]
                    if len(eos_pos) > 0:
                        cut = eos_pos[0].item() + 1
                        response_ids = response_ids[:cut]
                        full_ids = full_ids[:prompt_length + cut]

                    response_text = tokenizer.decode(
                        response_ids, skip_special_tokens=True
                    )
                    predicted_answer = extract_model_answer(response_text)
                    is_correct = check_answer(predicted_answer, ground_truth)

                    trajectories.append({
                        "prompt_ids": prompt_ids.squeeze(0).cpu(),
                        "prompt_length": prompt_length,
                        "response_ids": response_ids.cpu(),
                        "response_text": response_text,
                        "full_ids": full_ids.cpu(),
                        "full_attention_mask": torch.ones_like(full_ids).cpu(),
                        "predicted_answer": predicted_answer,
                        "ground_truth": ground_truth,
                        "is_correct": is_correct,
                        "response_length": len(response_ids),
                    })

                grouped_trajectories.append(trajectories)

    model.train()
    return grouped_trajectories


# ============================================================
# Log Prob 计算 — 单条
# 这里用 FSDP-wrapped model（所有 rank 调用次数已同步）
# FSDP forward hook 自动处理 all_gather/reduce_scatter
# ============================================================

def compute_log_probs_single(
    model,
    full_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_length: int,
) -> torch.Tensor:
    """单条轨迹 log probs（带梯度）"""
    if full_ids.dim() == 1:
        full_ids = full_ids.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    device = _get_device(model)
    full_ids = full_ids.to(device)
    attention_mask = attention_mask.to(device)

    outputs = model(
        input_ids=full_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )

    logits = outputs.logits
    log_probs_all = F.log_softmax(logits.float(), dim=-1)

    response_ids = full_ids[0, prompt_length:]
    response_log_probs = log_probs_all[0, prompt_length - 1:-1, :]

    log_probs = response_log_probs.gather(
        dim=-1,
        index=response_ids.unsqueeze(-1),
    ).squeeze(-1)

    return log_probs


@torch.no_grad()
def compute_log_probs_single_no_grad(
    model,
    full_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_length: int,
) -> torch.Tensor:
    """单条 log probs（无梯度，用于 ref model）"""
    if full_ids.dim() == 1:
        full_ids = full_ids.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    device = _get_device(model)
    full_ids = full_ids.to(device)
    attention_mask = attention_mask.to(device)

    outputs = model(
        input_ids=full_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )

    logits = outputs.logits
    log_probs_all = F.log_softmax(logits.float(), dim=-1)

    response_ids = full_ids[0, prompt_length:]
    response_log_probs = log_probs_all[0, prompt_length - 1:-1, :]

    log_probs = response_log_probs.gather(
        dim=-1,
        index=response_ids.unsqueeze(-1),
    ).squeeze(-1)

    return log_probs


# ============================================================
# 评估
# ============================================================

def evaluate_accuracy(
    model,
    tokenizer: AutoTokenizer,
    problems: List[Dict],
    data_config: DataConfig,
    temperature: float = 0.0,
    max_problems: int = 500,
    max_new_tokens: int = 512,
    **kwargs,
) -> Dict:
    """评估 Pass@1"""
    model.eval()

    if len(problems) > max_problems:
        import random
        problems = random.sample(problems, max_problems)

    correct = 0
    total = 0
    is_fsdp = _is_fsdp_model(model)

    if is_fsdp:
        ctx = _fsdp_generate_context(model)
    else:
        ctx = nullcontext(model)

    with ctx as gen_model:
        with torch.amp.autocast('cuda', enabled=False):
            device = _get_device(gen_model)

            for i, problem in enumerate(problems):
                question = problem["question"]
                ground_truth = problem["answer"]

                prompt_text = build_prompt(question, tokenizer, data_config)
                encoded = tokenizer(
                    prompt_text, return_tensors="pt", add_special_tokens=False,
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}

                with torch.no_grad():
                    gen_kwargs = dict(
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    if temperature > 0:
                        gen_kwargs.update(
                            do_sample=True, temperature=temperature, top_p=0.95
                        )
                    else:
                        gen_kwargs.update(do_sample=False, top_k=None, top_p=None)

                    try:
                        outputs = gen_model.generate(**encoded, **gen_kwargs)
                    except RuntimeError as e:
                        if "inf" in str(e).lower() or "nan" in str(e).lower():
                            logger.warning(
                                f"NaN/Inf in eval problem {i}, skipping"
                            )
                            total += 1
                            continue
                        raise

                response_ids = outputs[0, encoded["input_ids"].shape[1]:]
                response_text = tokenizer.decode(
                    response_ids, skip_special_tokens=True
                )
                predicted = extract_model_answer(response_text)

                if check_answer(predicted, ground_truth):
                    correct += 1
                total += 1

                if (i + 1) % 50 == 0:
                    logger.info(
                        f"Eval: {i+1}/{len(problems)}, acc={correct/total:.4f}"
                    )

    model.train()
    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"Evaluation: {correct}/{total} = {accuracy:.4f}")

    return {"accuracy": accuracy, "correct": correct, "total": total}