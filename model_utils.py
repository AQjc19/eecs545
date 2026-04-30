"""
SPARK v2 - 模型工具

负责:
1. 加载 Qwen-2.5-7B-Instruct 基座模型
2. 词表扩展：添加 <SKIP> token
3. <SKIP> Embedding 语义初始化
4. 挂载 Skip Adapter 模块
5. 冻结/解冻参数管理
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional
import logging

from config import ModelConfig, SPARKConfig
from skip_adapter import SkipAdapter

logger = logging.getLogger(__name__)


def load_base_model(
    config: ModelConfig,
    dtype: str = "bfloat16",
    device: str = "cuda",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    加载基座模型和 tokenizer

    Returns:
        model: Qwen-2.5-7B-Instruct
        tokenizer: 对应的 tokenizer
    """
    torch_dtype = getattr(torch, dtype)

    logger.info(f"Loading model: {config.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="sdpa",  # PyTorch 2.4 原生 SDPA，无需额外安装
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        padding_side="left",  # GRPO 需要 left padding
    )

    # 确保有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Model loaded. Vocab size: {len(tokenizer)}, "
                f"Hidden size: {model.config.hidden_size}")

    return model, tokenizer


def add_skip_token(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: ModelConfig,
) -> int:
    """
    向词表添加 <SKIP> token 并初始化其 embedding

    初始化策略:
    1. 找到语义相近 token ("skip", "pass", "omit", "therefore") 的 embedding
    2. 取算术平均值 e_mean
    3. 加入微小随机噪声: e_SKIP = e_mean + N(0, (0.01 × ||e_mean||)² I)

    Args:
        model: 基座模型
        tokenizer: tokenizer
        config: 模型配置

    Returns:
        skip_token_id: <SKIP> 的 token ID
    """
    # Step 1: 添加 <SKIP> 到词表
    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": [config.skip_token_str]
    })
    assert num_added == 1, f"Expected to add 1 token, got {num_added}"

    skip_token_id = tokenizer.convert_tokens_to_ids(config.skip_token_str)
    logger.info(f"Added {config.skip_token_str} with token_id = {skip_token_id}")

    # Step 2: Resize model embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Step 3: 收集语义相近 token 的 embedding
    embedding_layer = model.get_input_embeddings()
    semantic_embeddings = []

    for word in config.skip_semantic_tokens:
        # tokenize 可能产生多个子词，取第一个
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(token_ids) > 0:
            emb = embedding_layer.weight.data[token_ids[0]].clone()
            semantic_embeddings.append(emb)
            logger.info(f"  Semantic token '{word}' → id={token_ids[0]}, "
                        f"norm={emb.norm().item():.4f}")

    if len(semantic_embeddings) == 0:
        logger.warning("No semantic tokens found, using random initialization")
        return skip_token_id

    # Step 4: 计算均值
    e_mean = torch.stack(semantic_embeddings).mean(dim=0)
    mean_norm = e_mean.norm().item()

    # Step 5: 加入噪声
    noise_std = config.skip_embedding_noise_scale * mean_norm
    noise = torch.randn_like(e_mean) * noise_std
    e_skip = e_mean + noise

    # Step 6: 写入 embedding
    with torch.no_grad():
        embedding_layer.weight.data[skip_token_id] = e_skip

    # 同时更新 lm_head（输出层）如果是 tied weights
    output_layer = model.get_output_embeddings()
    if output_layer is not None and not _is_tied(model):
        with torch.no_grad():
            output_layer.weight.data[skip_token_id] = e_skip

    logger.info(f"<SKIP> embedding initialized: norm={e_skip.norm().item():.4f}, "
                f"noise_std={noise_std:.6f}")

    return skip_token_id


def _is_tied(model) -> bool:
    """检查 embedding 和 lm_head 是否共享权重"""
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()
    if output_emb is None:
        return False
    return input_emb.weight.data_ptr() == output_emb.weight.data_ptr()


def attach_skip_adapter(
    model: AutoModelForCausalLM,
    config: ModelConfig,
) -> SkipAdapter:
    """
    创建并挂载 Skip Adapter 到模型上

    Skip Adapter 作为独立模块存在，不修改原始模型结构。
    通过 model.skip_adapter 访问。

    Args:
        model: 基座模型
        config: 模型配置

    Returns:
        skip_adapter: 创建的 Skip Adapter 实例
    """
    # ★ 修改 1: 移除 bottleneck_ratio 参数，启用全维度残差网络
    adapter = SkipAdapter(
        hidden_size=config.hidden_size,
    )

    # 将 adapter 移到与模型相同的设备和精度
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    adapter = adapter.to(device=device, dtype=dtype)

    # 挂载到模型上
    model.skip_adapter = adapter

    param_info = adapter.get_param_count()
    
    # ★ 修改 2: 移除日志中对 bottleneck_dim 的引用，改为打印全维度
    logger.info(f"Skip Adapter attached: {param_info['total_M']:.2f}M params "
                f"(full_dim={config.hidden_size})")

    return adapter


def get_last_hidden_state_before_norm(
    model: AutoModelForCausalLM,
    input_ids: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[tuple] = None,
    position_ids: Optional[torch.Tensor] = None,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[tuple]]:
    """
    前向传播，返回最后一层隐状态（RMSNorm 之前）和 logits

    对于 Qwen2 架构:
        model.model(...) 返回 last_hidden_state（经过所有 Transformer Block 但在最终 RMSNorm 之前）
        model.model.norm(...) 是最终 RMSNorm
        model.lm_head(...) 是输出投影

    Args:
        model: 模型
        input_ids: token IDs (batch_size, seq_len)
        inputs_embeds: 直接输入 embedding（用于 <SKIP> 路径）
        attention_mask: 注意力掩码
        past_key_values: KV cache
        position_ids: 位置 ID
        use_cache: 是否返回 KV cache

    Returns:
        h_last: RMSNorm 之前的 hidden state (batch_size, seq_len, hidden_size)
        logits: 输出 logits (batch_size, seq_len, vocab_size)
        new_past_key_values: 更新后的 KV cache（如果 use_cache=True）
    """
    # 获取 Transformer Block 输出（RMSNorm 之前）
    outputs = model.model(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
        use_cache=use_cache,
        output_hidden_states=False,
    )

    h_last = outputs.last_hidden_state  # RMSNorm 之前

    # 手动应用 RMSNorm + lm_head 得到 logits
    h_normed = model.model.norm(h_last)
    logits = model.lm_head(h_normed)

    new_past = outputs.past_key_values if use_cache else None

    return h_last, logits, new_past


def setup_model_for_phase1(config: SPARKConfig):
    """
    Phase 1 完整模型准备

    执行顺序:
    1. 加载基座模型
    2. 添加 <SKIP> token 并初始化 embedding
    3. 挂载 Skip Adapter（W_down 零初始化）
    4. 返回准备好的模型、tokenizer、skip_token_id

    注意: Phase 1 中不启用 <SKIP>，这些组件只是预先安装好。

    Returns:
        model: 准备好的模型
        tokenizer: tokenizer
        skip_token_id: <SKIP> 的 token ID
        skip_adapter: Skip Adapter 实例
    """
    # 1. 加载基座模型
    model, tokenizer = load_base_model(
        config.model,
        dtype=config.dtype,
        device=config.device,
    )

    # 2. 添加 <SKIP> token
    skip_token_id = add_skip_token(model, tokenizer, config.model)

    # 3. 挂载 Skip Adapter
    skip_adapter = attach_skip_adapter(model, config.model)

    # 4. 设置模型为训练模式
    model.train()

    # 5. Skip Adapter 在 Phase 1 中冻结（不需要梯度）
    for param in skip_adapter.parameters():
        param.requires_grad = False
    logger.info("Skip Adapter frozen for Phase 1 (will be unfrozen in Phase 1.5)")

    # 6. <SKIP> embedding 在 Phase 1 中也冻结
    # （实际上模型很少会生成它，冻结只是额外保险）
    # 注意：由于 embedding layer 整体参与训练，我们通过 hook 排除 <SKIP> 梯度
    # 这在 grpo_trainer.py 中处理

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params / 1e9:.2f}B, "
                f"Trainable: {trainable_params / 1e9:.2f}B")

    return model, tokenizer, skip_token_id, skip_adapter


def save_checkpoint(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    skip_adapter: SkipAdapter,
    step: int,
    output_dir: str,
    extra_state: Optional[dict] = None,
):
    """保存 checkpoint（模型 + tokenizer + Skip Adapter + 额外状态）"""
    import os
    save_path = os.path.join(output_dir, f"checkpoint-step-{step}")
    os.makedirs(save_path, exist_ok=True)

    # 保存模型和 tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # 单独保存 Skip Adapter（因为它不是模型原有结构的一部分）
    adapter_path = os.path.join(save_path, "skip_adapter.pt")
    torch.save(skip_adapter.state_dict(), adapter_path)

    # 保存额外状态
    if extra_state is not None:
        state_path = os.path.join(save_path, "training_state.pt")
        torch.save(extra_state, state_path)

    logger.info(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    config: SPARKConfig,
):
    """加载 checkpoint"""
    torch_dtype = getattr(torch, config.dtype)

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch_dtype,
        device_map=config.device,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        padding_side="left",
    )

    # 重建 Skip Adapter
    skip_adapter = attach_skip_adapter(model, config.model)

    # 加载 Skip Adapter 权重
    import os
    adapter_path = os.path.join(checkpoint_path, "skip_adapter.pt")
    if os.path.exists(adapter_path):
        try:
            # 尝试加载，消除 weights_only 警告
            state_dict = torch.load(adapter_path, map_location=config.device, weights_only=True)
            skip_adapter.load_state_dict(state_dict, strict=True)
            logger.info("Skip Adapter weights loaded successfully.")
        except RuntimeError as e:
            # 捕获因架构升级（如移除了 bottleneck）导致的尺寸不匹配
            logger.warning("="*60)
            logger.warning("⚠️ Architecture Mismatch Detected in Skip Adapter!")
            logger.warning("This is EXPECTED if you upgraded to the v2 Full-Dimension Adapter.")
            logger.warning("Discarding old weights and using fresh residual initialization.")
            logger.warning("="*60)

    skip_token_id = tokenizer.convert_tokens_to_ids(config.model.skip_token_str)

    # 加载额外状态
    state_path = os.path.join(checkpoint_path, "training_state.pt")
    extra_state = None
    if os.path.exists(state_path):
        extra_state = torch.load(state_path, map_location=config.device)

    return model, tokenizer, skip_token_id, skip_adapter, extra_state