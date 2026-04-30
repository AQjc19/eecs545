"""
SPARK v2 - 数据工具

关键修复:
- 本地数据集用 load_from_disk（不是 load_dataset）
- 自动检测字段名（problem/question, solution/answer 等）
- v3_simple prompt 格式 + #### 答案提取
"""

import re
import random
import logging
import os
import math
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
from transformers import AutoTokenizer

from config import DataConfig

logger = logging.getLogger(__name__)


# ============================================================
# 通用数据集加载：自动判断 load_from_disk vs load_dataset
# ============================================================

def smart_load_dataset(path: str, split: Optional[str] = None, **kwargs) -> Dataset:
    """
    智能加载数据集，自动处理三种本地格式：

    格式 A: save_to_disk 保存的 Arrow 目录
        → 特征: 含 state.json 或 dataset_info.json + *.arrow 文件
        → 方法: load_from_disk(path)

    格式 B: 带加载脚本的本地目录（如 gsm8k clone）
        → 特征: 含 *.py 或 *.json 加载脚本，需要 config name
        → 方法: load_dataset(path, "main", split=...)

    格式 C: HuggingFace Hub ID
        → 方法: load_dataset(path, split=...)
    """
    kwargs.pop("trust_remote_code", None)

    if not os.path.isdir(path):
        # 不是本地目录 → Hub 加载
        logger.info(f"Loading from Hub: {path}")
        if split:
            return load_dataset(path, split=split, **kwargs)
        ds = load_dataset(path, **kwargs)
        return _pick_split(ds, split)

    # ── 本地目录：判断是 save_to_disk (格式A) 还是脚本式 (格式B) ──

    dir_files = os.listdir(path)

    # 格式 A 特征: state.json 或 (dataset_info.json + arrow 文件)
    has_state = "state.json" in dir_files
    has_arrow = any(f.endswith(".arrow") for f in dir_files)
    # save_to_disk 的 DatasetDict 会有子目录（train/, test/）里带 state.json
    subdirs = [f for f in dir_files if os.path.isdir(os.path.join(path, f))]
    subdir_has_state = any(
        os.path.exists(os.path.join(path, sd, "state.json"))
        for sd in subdirs
    )

    is_save_to_disk = has_state or has_arrow or subdir_has_state

    if is_save_to_disk:
        logger.info(f"Loading from disk (save_to_disk format): {path}")
        ds = load_from_disk(path)
        return _pick_split(ds, split)

    # 格式 B: 脚本式本地目录 → load_dataset + 自动猜 config name
    logger.info(f"Loading from local script directory: {path}")

    # 先直接尝试
    try:
        if split:
            return load_dataset(path, split=split, **kwargs)
        ds = load_dataset(path, **kwargs)
        return _pick_split(ds, split)
    except ValueError as e:
        err_msg = str(e)
        # "Config name is missing. Please pick one among: ['main', 'socratic']"
        if "Config name is missing" in err_msg:
            # 从错误信息中解析可用的 config names
            import ast
            match = re.search(r"\[([^\]]+)\]", err_msg)
            if match:
                try:
                    configs = ast.literal_eval(f"[{match.group(1)}]")
                except:
                    configs = ["main"]
            else:
                configs = ["main"]

            # 用第一个 config name 重试
            config_name = configs[0]
            logger.info(f"  Retrying with config name: '{config_name}'")
            if split:
                return load_dataset(path, config_name, split=split, **kwargs)
            ds = load_dataset(path, config_name, **kwargs)
            return _pick_split(ds, split)
        else:
            raise


def _pick_split(ds, split: Optional[str]) -> Dataset:
    """从 DatasetDict 中选择合适的 split"""
    if isinstance(ds, Dataset):
        return ds
    if not isinstance(ds, DatasetDict):
        return ds

    if split and split in ds:
        return ds[split]
    for try_split in [split, "train", "test", "validation"]:
        if try_split and try_split in ds:
            logger.info(f"  Using split: '{try_split}' ({len(ds[try_split])} rows)")
            return ds[try_split]
    first_split = list(ds.keys())[0]
    logger.warning(f"  Split '{split}' not found, using '{first_split}'")
    return ds[first_split]


def detect_field(item: dict, candidates: List[str], field_desc: str) -> Optional[str]:
    """
    自动检测字段名

    Args:
        item: 数据集中的一行
        candidates: 候选字段名列表（按优先级）
        field_desc: 字段描述（用于日志）

    Returns:
        匹配到的字段名，或 None
    """
    for name in candidates:
        if name in item:
            return name
    logger.warning(f"Cannot find {field_desc} field. "
                   f"Tried: {candidates}. "
                   f"Available: {list(item.keys())}")
    return None


# ============================================================
# 答案提取 — 多策略级联
# ============================================================

def extract_model_answer(text: str) -> Optional[str]:
    """
    从模型生成文本中提取最终答案
    优先级: #### > "the answer is" > \\boxed{} > 最后 "=" > 最后数字
    """
    text = text.strip()

    # Pattern 1: ####
    match = re.search(r'####\s*(.+?)(?:\n|$)', text)
    if match:
        ans = match.group(1).strip()
        ans = ans.replace(',', '').replace('$', '').replace('%', '').rstrip('.')
        if ans:
            return ans

    # Pattern 2: "the answer is"
    match = re.search(
        r'(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*([+-]?[\d,]+\.?\d*)',
        text, re.IGNORECASE
    )
    if match:
        return match.group(1).replace(',', '')

    # Pattern 3: \boxed{...}
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).replace(',', '').strip()

    # Pattern 4: 最后 "= number"
    matches = re.findall(r'=\s*([+-]?[\d,]+\.?\d*)', text)
    if matches:
        return matches[-1].replace(',', '')

    # Pattern 5: 最后一个数字
    matches = re.findall(r'([+-]?\d[\d,]*\.?\d*)', text)
    if matches:
        return matches[-1].replace(',', '')

    return None


def normalize_answer(answer: str) -> str:
    """标准化答案字符串"""
    if answer is None:
        return ""
    ans = answer.strip()
    ans = ans.replace("$", "").replace("%", "").replace(",", "")
    ans = ans.replace("\\$", "").replace("\\%", "")
    ans = re.sub(r'\\text\{([^}]*)\}', r'\1', ans)
    ans = re.sub(r'\s+', ' ', ans).strip()

    try:
        val = float(ans)
        if not math.isfinite(val):
            # inf / nan → 当作无法解析的字符串
            return ans.lower()
        if val == int(val):
            return str(int(val))
        return f"{val:.6f}"
    except (ValueError, TypeError, OverflowError):
        pass

    frac_match = re.match(r'\\frac\{([^}]+)\}\{([^}]+)\}', ans)
    if frac_match:
        try:
            val = float(frac_match.group(1)) / float(frac_match.group(2))
            if not math.isfinite(val):
                return ans.lower()
            if val == int(val):
                return str(int(val))
            return f"{val:.6f}"
        except (ValueError, ZeroDivisionError, OverflowError):
            pass

    return ans.lower()


def check_answer(prediction: str, ground_truth: str) -> bool:
    """检查预测是否正确（精确 + 数值近似）"""
    if prediction is None:
        return False

    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)

    if pred_norm == gt_norm:
        return True

    try:
        pred_val = float(pred_norm)
        gt_val = float(gt_norm)
        if not (math.isfinite(pred_val) and math.isfinite(gt_val)):
            return False
        if gt_val == int(gt_val):
            return abs(pred_val - gt_val) < 0.5
        return abs(pred_val - gt_val) < max(0.01, abs(gt_val) * 0.001)
    except (ValueError, TypeError, OverflowError):
        pass

    return False


# ============================================================
# GSM8K 数据处理
# ============================================================

def extract_gsm8k_answer(answer_text: str) -> str:
    """从 GSM8K 答案文本中提取数值: "step1\\n#### 42" → "42" """
    match = re.search(r'####\s*([\-\d,\.]+)', answer_text)
    if match:
        return match.group(1).replace(",", "").strip()
    return answer_text.strip()


def load_gsm8k(config: DataConfig) -> List[Dict]:
    """加载 GSM8K 训练集"""
    logger.info(f"Loading GSM8K from {config.gsm8k_path}...")
    dataset = smart_load_dataset(config.gsm8k_path, split="train")

    # 自动检测字段名
    sample = dataset[0]
    q_field = detect_field(sample, ["question", "problem", "input", "text"], "question")
    a_field = detect_field(sample, ["answer", "solution", "output", "target"], "answer")

    if q_field is None or a_field is None:
        logger.error(f"GSM8K fields not recognized. Available: {list(sample.keys())}")
        return []

    logger.info(f"  GSM8K fields: question='{q_field}', answer='{a_field}'")

    problems = []
    for item in dataset:
        answer = extract_gsm8k_answer(str(item[a_field]))
        problems.append({
            "question": item[q_field],
            "answer": answer,
            "source": "gsm8k",
        })

    logger.info(f"GSM8K train: {len(problems)} problems")
    return problems


def load_gsm8k_test(config: DataConfig) -> List[Dict]:
    """加载 GSM8K 测试集"""
    logger.info(f"Loading GSM8K test from {config.gsm8k_path}...")
    dataset = smart_load_dataset(config.gsm8k_path, split="test")

    sample = dataset[0]
    q_field = detect_field(sample, ["question", "problem", "input", "text"], "question")
    a_field = detect_field(sample, ["answer", "solution", "output", "target"], "answer")

    if q_field is None or a_field is None:
        logger.error(f"GSM8K test fields not recognized. Available: {list(sample.keys())}")
        return []

    problems = []
    for item in dataset:
        answer = extract_gsm8k_answer(str(item[a_field]))
        problems.append({
            "question": item[q_field],
            "answer": answer,
            "source": "gsm8k_test",
        })

    logger.info(f"GSM8K test: {len(problems)} problems")
    return problems


# ============================================================
# MATH 数据处理
# ============================================================

def extract_boxed_answer(solution: str) -> Optional[str]:
    """从 solution 中提取 \\boxed{...} 答案（处理嵌套花括号）"""
    pattern = r'\\boxed\{'
    matches = list(re.finditer(pattern, solution))
    if not matches:
        return None
    last_match = matches[-1]
    start = last_match.end()
    depth = 1
    i = start
    while i < len(solution) and depth > 0:
        if solution[i] == '{':
            depth += 1
        elif solution[i] == '}':
            depth -= 1
        i += 1
    if depth == 0:
        return solution[start:i - 1].strip()
    return None


def extract_math_answer(solution: str) -> str:
    """从 MATH solution 中提取答案：优先 \\boxed{}，备选 #### ，再备选最后一行"""
    # 尝试 \boxed{}
    ans = extract_boxed_answer(solution)
    if ans:
        return ans

    # 尝试 ####
    match = re.search(r'####\s*(.+?)(?:\n|$)', solution)
    if match:
        return match.group(1).strip()

    # 最后一行
    lines = solution.strip().split('\n')
    return lines[-1].strip()


def load_math(config: DataConfig) -> List[Dict]:
    """
    加载 MATH 训练数据集

    自动检测字段名，兼容:
    - hendrycks/MATH: problem, solution, level, type
    - HuggingFaceH4/MATH-500: problem, solution, answer
    - competition_math: problem, solution
    - 其他变体
    """
    logger.info(f"Loading MATH from {config.math_path}...")

    try:
        dataset = smart_load_dataset(config.math_path, split="train")
    except Exception as e1:
        logger.warning(f"Failed to load MATH train split: {e1}")
        try:
            # MATH-500 只有 test split
            dataset = smart_load_dataset(config.math_path, split="test")
            logger.info("  Loaded 'test' split (MATH-500 style)")
        except Exception as e2:
            # 最后尝试不指定 split
            try:
                dataset = smart_load_dataset(config.math_path)
            except Exception as e3:
                logger.warning(f"Cannot load MATH dataset: {e3}")
                logger.warning("Proceeding with GSM8K only")
                return []

    # 自动检测字段名
    sample = dataset[0]
    available = list(sample.keys())
    logger.info(f"  MATH dataset columns: {available}")

    # 问题字段
    q_field = detect_field(sample,
        ["problem", "question", "input", "text", "prompt"],
        "question")

    # 答案/解法字段
    sol_field = detect_field(sample,
        ["solution", "answer", "output", "target", "response"],
        "solution")

    if q_field is None or sol_field is None:
        logger.error(f"MATH fields not recognized. Available: {available}")
        return []

    logger.info(f"  MATH fields: question='{q_field}', solution='{sol_field}'")

    # 检查是否有独立的 answer 字段（MATH-500 同时有 solution 和 answer）
    ans_field = None
    if "answer" in sample and sol_field != "answer":
        ans_field = "answer"
        logger.info(f"  Also found separate answer field: '{ans_field}'")

    problems = []
    for item in dataset:
        # 优先用独立 answer 字段，否则从 solution 提取
        if ans_field and item.get(ans_field):
            answer = str(item[ans_field]).strip()
        else:
            answer = extract_math_answer(str(item[sol_field]))

        problems.append({
            "question": item[q_field],
            "answer": answer,
            "source": "math",
            "level": item.get("level", "unknown"),
            "type": item.get("type", item.get("subject", "unknown")),
        })

    logger.info(f"MATH: {len(problems)} problems loaded")
    return problems


def load_math500(config: DataConfig) -> List[Dict]:
    """加载 MATH-500 评估集"""
    logger.info(f"Loading MATH-500 from {config.math500_path}...")
    try:
        dataset = smart_load_dataset(config.math500_path, split="test")
    except Exception as e:
        logger.warning(f"Failed to load MATH-500: {e}")
        return []

    sample = dataset[0]
    q_field = detect_field(sample,
        ["problem", "question", "input", "text"], "question")
    sol_field = detect_field(sample,
        ["solution", "answer", "output", "target"], "solution")
    ans_field = None
    if "answer" in sample and sol_field != "answer":
        ans_field = "answer"

    if q_field is None or sol_field is None:
        logger.error(f"MATH-500 fields not recognized: {list(sample.keys())}")
        return []

    problems = []
    for item in dataset:
        if ans_field and item.get(ans_field):
            answer = str(item[ans_field]).strip()
        else:
            answer = extract_math_answer(str(item[sol_field]))
        problems.append({
            "question": item[q_field],
            "answer": answer,
            "source": "math500",
        })

    logger.info(f"MATH-500: {len(problems)} problems")
    return problems


# ============================================================
# Prompt 构造 — v3_simple 格式
# ============================================================

def build_prompt(
    question: str,
    tokenizer: AutoTokenizer,
    config: DataConfig,
) -> str:
    """
    v3_simple 格式 (Qwen2.5-1.5B 上实测最优):
      user: "{question}\n\nPlease solve step by step and put your final answer after ####."
    无 system prompt
    """
    if config.use_chat_template:
        messages = [
            {
                "role": "user",
                "content": f"{question}\n\nPlease solve step by step and put your final answer after ####."
            }
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = (
            f"Question: {question}\n\n"
            f"Please solve step by step and put your final answer after ####.\n\n"
            f"Solution:"
        )
    return prompt


def build_prompt_ids(
    question: str,
    tokenizer: AutoTokenizer,
    config: DataConfig,
    max_length: Optional[int] = None,
) -> Dict:
    """构造 prompt 并 tokenize"""
    prompt_text = build_prompt(question, tokenizer, config)
    max_len = max_length or config.max_prompt_length

    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "prompt_length": encoded["input_ids"].shape[1],
        "prompt_text": prompt_text,
    }


# ============================================================
# 数据集类
# ============================================================

class MathProblemDataset:
    """合并 GSM8K + MATH，提供批次采样"""

    def __init__(self, config: DataConfig, tokenizer: AutoTokenizer, seed: int = 42):
        self.config = config
        self.tokenizer = tokenizer

        gsm8k = load_gsm8k(config)
        math_data = load_math(config)

        self.problems = gsm8k + math_data
        random.seed(seed)
        random.shuffle(self.problems)

        logger.info(f"Total training problems: {len(self.problems)} "
                    f"(GSM8K: {len(gsm8k)}, MATH: {len(math_data)})")
        self.epoch_counter = 0

    def __len__(self):
        return len(self.problems)

    def sample_batch(self, batch_size: int, seed: Optional[int] = None) -> List[Dict]:
        """
        采样一个 batch。
        seed: 如果提供，用固定种子采样（确保多 GPU 各 rank 采样相同 batch）
        """
        if seed is not None:
            rng = random.Random(seed + self.epoch_counter)
            indices = rng.sample(range(len(self.problems)), min(batch_size, len(self.problems)))
            self.epoch_counter += 1
        else:
            indices = random.sample(range(len(self.problems)), min(batch_size, len(self.problems)))
        batch = []
        for idx in indices:
            problem = self.problems[idx]
            prompt_data = build_prompt_ids(problem["question"], self.tokenizer, self.config)
            batch.append({
                "question": problem["question"],
                "answer": problem["answer"],
                "source": problem["source"],
                "prompt_ids": prompt_data["input_ids"].squeeze(0),
                "prompt_mask": prompt_data["attention_mask"].squeeze(0),
                "prompt_length": prompt_data["prompt_length"],
            })
        return batch

    def get_calibration_subset(self, n: int, source: str = "math") -> List[Dict]:
        filtered = [p for p in self.problems if p["source"] == source]
        if len(filtered) < n:
            logger.warning(f"Only {len(filtered)} {source} problems, adding from other sources")
            remaining = [p for p in self.problems if p["source"] != source]
            filtered.extend(remaining[:n - len(filtered)])
        if len(filtered) < n:
            n = len(filtered)
        return random.sample(filtered, n)


class GSM8KEvalDataset:
    """GSM8K 评估数据集"""

    def __init__(self, config: DataConfig, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.config = config
        self.problems = load_gsm8k_test(config)

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        return self.problems[idx]