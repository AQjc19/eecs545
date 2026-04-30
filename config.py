"""
SPARK v2 - 全局配置
适配: Qwen2.5-1.5B-Instruct + v3_simple prompt + 1~8卡
"""
from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class ModelConfig:
    model_name_or_path: str = "/fs1/home/hanrui/hanrui/drought/model/Qwen2.5-1.5B-Instruct"
    hidden_size: int = 1536
    num_attention_heads: int = 12
    head_dim: int = 128
    skip_token_str: str = "<SKIP>"
    skip_semantic_tokens: list = field(default_factory=lambda: [
        "skip", "pass", "omit", "therefore"
    ])
    skip_embedding_noise_scale: float = 0.01
    adapter_bottleneck_ratio: int = 4
    max_seq_length: int = 2048
    max_new_tokens: int = 512
    k_max: int = 8


@dataclass
class KVIGConfig:
    alpha: float = 1.03
    beta: float = 1.06
    power_iter_steps: int = 3
    eps: float = 1e-8
    d_eff_threshold: Optional[float] = None
    kvig_mean: Optional[float] = None
    kvig_std: Optional[float] = None
    t_ref: Optional[float] = None


@dataclass
class Phase1TrainConfig:
    group_size: int = 8
    batch_size: int = 4
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    # ── PPO/GRPO 核心参数 ──
    kl_coeff: float = 0.04       # KL 惩罚系数（即使没有 ref model 也生效）
    clip_range: float = 0.2      # PPO ratio clip 范围 [1-ε, 1+ε]
    ppo_epochs: int = 1          # ★ 每步 PPO epoch 数（≥2 才能激活 clip）★
    temperature: float = 1.0
    top_p: float = 1.0
    num_train_steps: int = 200
    eval_interval: int = 50
    save_interval: int = 100
    log_interval: int = 5
    output_dir: str = "./checkpoints/phase1"
    log_dir: str = "./logs/phase1"
    gradient_checkpointing: bool = True


@dataclass
class CalibrationConfig:
    num_problems: int = 500
    num_trajectories_per_problem: int = 16
    temperature: float = 0.6
    min_auc: float = 0.65
    min_p_value_threshold: float = 0.001
    min_d_eff_median: float = 3.0
    alpha_search_range: tuple = (0.72, 1.34)
    beta_search_range: tuple = (0.74, 1.38)
    search_grid_size: int = 5
    calibration_data_dir: str = "./data/calibration"
    calibration_output_path: str = "./checkpoints/phase1/calibration_stats.json"


@dataclass
class RewardConfig:
    lambda1_init: float = 0.10
    lambda1_full: float = 0.30
    lambda2_init: float = 0.0
    lambda2_full: float = 0.15
    lambda_mod_init: float = 0.0
    lambda_mod_full: float = 0.3
    gamma_tanh: float = 5.0
    mu_skip: float = 0.5
    beta_eff: float = 1.0
    lambda1_incorrect_ratio: float = 0.1


@dataclass
class DataConfig:
    gsm8k_path: str = "/fs1/home/hanrui/hanrui/drought/model/gsm8k_data"
    math_path: str = "/fs1/home/hanrui/hanrui/drought/model/math_data"
    use_chat_template: bool = True
    answer_format: str = "hash"
    math500_path: str = "/fs1/home/hanrui/hanrui/drought/model/math500"
    max_prompt_length: int = 512


@dataclass
class SPARKConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    kvig: KVIGConfig = field(default_factory=KVIGConfig)
    phase1: Phase1TrainConfig = field(default_factory=Phase1TrainConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: int = 42
    device: str = "cuda"
    dtype: str = "float16"
    num_gpus: int = 1


def get_config() -> SPARKConfig:
    return SPARKConfig()