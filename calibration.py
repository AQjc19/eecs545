"""
SPARK v2 - KVIG 校准模块

Phase 1 训练完成后执行 KVIG 基线校准:
1. 生成校准数据：500题 × 16轨迹 = 8000条轨迹
2. 计算每条轨迹每步的 KVIG
3. 统计分析：正确组 vs 错误组
4. 验证 KVIG 信号有效性（AUC ≥ 0.65, p < 0.001）
5. 输出关键常数：d_eff_threshold, T_ref, KVIG_mean, KVIG_std

如果验证失败，自动搜索更优的 α, β 参数
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score
import logging
import json
import os
import time
import gc

import ray
from ray import tune, train
from ray.tune.search.optuna import OptunaSearch
import optuna

from config import SPARKConfig, CalibrationConfig, KVIGConfig, ModelConfig
from kvig import KVIGComputer
from rollout import generate_trajectories
from data_utils import build_prompt

logger = logging.getLogger(__name__)


class KVIGCalibrator:
    """
    KVIG 校准器

    执行完整的校准流程并输出统计基线
    """

    def __init__(self, config: SPARKConfig):
        self.config = config
        self.cal_config = config.calibration
        self.kvig_computer = KVIGComputer(config.kvig, config.model)

    def run_calibration(
        self,
        model,
        tokenizer,
        calibration_problems: List[Dict],
    ) -> Dict:
        """
        执行完整校准流程

        Args:
            model: Phase 1 训练完成的 checkpoint
            tokenizer: tokenizer
            calibration_problems: 校准题目列表

        Returns:
            calibration_results: {
                "d_eff_threshold": float,
                "t_ref": float,
                "kvig_mean": float,
                "kvig_std": float,
                "auc": float,
                "p_value": float,
                "correct_mean_kvig": float,
                "incorrect_mean_kvig": float,
                "d_eff_median": float,
                "d_eff_mean": float,
                "alpha": float,
                "beta": float,
                "num_correct_trajectories": int,
                "num_incorrect_trajectories": int,
                "validation_passed": bool,
            }
        """
        logger.info("=" * 60)
        logger.info("Starting KVIG Calibration")
        logger.info(f"  Problems: {len(calibration_problems)}")
        logger.info(f"  Trajectories per problem: {self.cal_config.num_trajectories_per_problem}")
        logger.info("=" * 60)

        cal_start = time.time()


        # +++ 替换为 +++
        # ========================================
        # Step 1 & 2: 断点续跑生成与 KVIG 计算
        # ========================================
        trajectories_data, kvig_results = self._generate_and_compute_kvig_with_resume(
            model, tokenizer, calibration_problems
        )
        
        num_correct = sum(1 for t in trajectories_data if t["is_correct"])
        num_incorrect = len(trajectories_data) - num_correct
        logger.info(f"  Total Valid Trajectories: {len(trajectories_data)}")
        logger.info(f"  Correct: {num_correct}, Incorrect: {num_incorrect}")

        if num_correct < 50 or num_incorrect < 50:
            logger.warning("Too few correct/incorrect trajectories for reliable calibration!")

        # ========================================
        # Step 3: 统计分析 + 验证
        # ========================================
        logger.info("Step 3: Statistical analysis and validation...")
        stats = self._analyze_statistics(trajectories_data, kvig_results)

        # ========================================
        # Step 4: 验证是否通过
        # ========================================
        passed = self._validate(stats)

        if not passed:
            logger.warning("Calibration validation FAILED. Searching for better α, β...")
            stats, passed = self._search_alpha_beta(model, trajectories_data)

        # ========================================
        # 强力保全 - 导出最终 SFT 数据供 Phase 1.5 使用
        # ========================================
        self._export_final_sft_data(trajectories_data, kvig_results)

        # ========================================
        # Step 5: 输出结果
        # ========================================
        results = {
            "d_eff_threshold": stats["d_eff_median"],
            "t_ref": stats["correct_mean_length"],
            "kvig_mean": stats["overall_mean_kvig"],
            "kvig_std": stats["overall_std_kvig"],
            "auc": stats["auc"],
            "p_value": stats["p_value"],
            "correct_mean_kvig": stats["correct_mean_kvig"],
            "incorrect_mean_kvig": stats["incorrect_mean_kvig"],
            "d_eff_median": stats["d_eff_median"],
            "d_eff_mean": stats["d_eff_mean"],
            "alpha": self.kvig_computer.alpha,
            "beta": self.kvig_computer.beta,
            "num_correct_trajectories": num_correct,
            "num_incorrect_trajectories": num_incorrect,
            "validation_passed": passed,
        }

        cal_time = time.time() - cal_start
        logger.info(f"Calibration completed in {cal_time / 60:.1f} minutes")
        self._log_results(results)

        # 保存结果
        self._save_results(results)

        return results


    @torch.no_grad()
    @torch.no_grad()
    def _compute_all_kvig(
        self,
        model,
        trajectories: List[Dict],
        log_prefix: str = ""   # ★ 这里加上 log_prefix 参数
    ) -> List[Dict]:
        """
        计算所有轨迹的 KVIG
        """
        model.eval()
        kvig_results = []

        for i, traj in enumerate(trajectories):
            full_ids = traj["full_ids"].to(model.device)
            attention_mask = traj["full_attention_mask"].to(model.device)
            prompt_length = traj["prompt_length"]

            result = self.kvig_computer.compute_trajectory_from_model(
                model=model,
                input_ids=full_ids,
                attention_mask=attention_mask,
                prompt_length=prompt_length,
                d_eff_threshold_override=None,  # 校准阶段不用阈值
            )

            kvig_results.append(result)

            # ★ 把这里的 500 改成 100，并加上 log_prefix，方便断点续跑时看进度
            if (i + 1) % 100 == 0:
                logger.info(f"  {log_prefix} Computed KVIG for {i+1}/{len(trajectories)} trajectories")

        return kvig_results


    # +++ 新增代码 (直接粘贴到类里面) +++
    def _export_final_sft_data(self, trajectories, kvig_results):
        """★ 导出完整的原始数据，供 Phase 1.5 SFT 训练读取"""
        final_path = os.path.join(
            os.path.dirname(self.cal_config.calibration_output_path),
            "calibration_raw_data.pt"
        )
        logger.info(f"Step 5: Exporting full dataset for Phase 1.5 SFT to {final_path}...")
        
        sft_dataset = []
        for traj, kvig_res in zip(trajectories, kvig_results):
            sft_dataset.append({
                "prompt_length": traj["prompt_length"],
                "response_length": traj["response_length"],
                "full_ids": traj["full_ids"], # 已经是 CPU Tensor
                "is_correct": traj["is_correct"],
                "kvig_values": kvig_res["kvig_values"],
                "mean_kvig": kvig_res["mean_kvig"],
                "d_eff_values": kvig_res["d_eff_values"]
            })
            
        torch.save(sft_dataset, final_path)
        logger.info("  ✓ Final SFT data successfully compiled and saved.")


    #新增
    # +++ 新增代码 (直接粘贴到类里面) +++
    def _generate_and_compute_kvig_with_resume(
        self, model, tokenizer, problems: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """分块生成轨迹并立刻计算 KVIG，支持断点续跑"""
        
        # 创建缓存目录
        raw_data_dir = os.path.join(
            os.path.dirname(self.cal_config.calibration_output_path), 
            "raw_chunks"
        )
        os.makedirs(raw_data_dir, exist_ok=True)

        all_trajectories = []
        all_kvig_results = []

        # ★ 榨干显存优化：从 4 提升到 16（每次处理 16题 * 16轨迹 = 256 条序列）
        batch_size = 16 
        
        for batch_start in range(0, len(problems), batch_size):
            batch_end = min(batch_start + batch_size, len(problems))
            chunk_file = os.path.join(raw_data_dir, f"chunk_{batch_start}_{batch_end}.pt")

            # ★ 断点续跑逻辑
            if os.path.exists(chunk_file):
                try:
                    logger.info(f"  [Resume] Loading completed chunk {batch_start}-{batch_end} from {chunk_file}")
                    chunk_data = torch.load(chunk_file, map_location="cpu")
                    all_trajectories.extend(chunk_data["trajectories"])
                    all_kvig_results.extend(chunk_data["kvig_results"])
                    continue
                except Exception as e:
                    logger.warning(f"  [Resume] Failed to load {chunk_file}: {e}. Recomputing this chunk...")

            logger.info(f"Processing problems {batch_start}-{batch_end}/{len(problems)}...")
            batch_problems = problems[batch_start:batch_end]

            # 1. 生成轨迹
            grouped = generate_trajectories(
                model=model,
                tokenizer=tokenizer,
                prompts=batch_problems,
                group_size=self.cal_config.num_trajectories_per_problem,
                temperature=self.cal_config.temperature,
                max_new_tokens=self.config.model.max_new_tokens,
                data_config=self.config.data,
            )

            batch_trajs = []
            for group in grouped:
                batch_trajs.extend(group)

            # 2. 立即计算 KVIG
            batch_kvig = self._compute_all_kvig(model, batch_trajs, log_prefix=f"[{batch_start}-{batch_end}]")

            # 3. 剥离到 CPU 并持久化落盘
            for t in batch_trajs:
                if isinstance(t.get("full_ids"), torch.Tensor):
                    t["full_ids"] = t["full_ids"].cpu()
                if isinstance(t.get("full_attention_mask"), torch.Tensor):
                    t["full_attention_mask"] = t["full_attention_mask"].cpu()

            chunk_data = {
                "trajectories": batch_trajs,
                "kvig_results": batch_kvig
            }
            torch.save(chunk_data, chunk_file)
            logger.info(f"  [Save] Chunk {batch_start}-{batch_end} safely appended to disk.")

            all_trajectories.extend(batch_trajs)
            all_kvig_results.extend(batch_kvig)

            # 强行释放当前批次的显存
            torch.cuda.empty_cache()
            gc.collect()

        return all_trajectories, all_kvig_results

    def _analyze_statistics(
        self,
        trajectories: List[Dict],
        kvig_results: List[Dict],
    ) -> Dict:
        """
        统计分析

        计算:
        - 正确组 vs 错误组的 mean_KVIG 差异
        - t 检验 p 值
        - AUC（用 mean_KVIG 区分正确/错误）
        - d_eff 分布
        - T_ref（正确轨迹平均长度）
        """
        correct_mean_kvigs = []
        incorrect_mean_kvigs = []
        all_mean_kvigs = []
        all_d_effs = []
        correct_lengths = []
        labels = []  # 1 = correct, 0 = incorrect

        for i, (traj, kvig_res) in enumerate(zip(trajectories, kvig_results)):
            mean_kvig = kvig_res["mean_kvig"]
            all_mean_kvigs.append(mean_kvig)

            # 收集 d_eff 值
            all_d_effs.extend(kvig_res["d_eff_values"])

            if traj["is_correct"]:
                correct_mean_kvigs.append(mean_kvig)
                correct_lengths.append(traj["response_length"])
                labels.append(1)
            else:
                incorrect_mean_kvigs.append(mean_kvig)
                labels.append(0)

        # t 检验
        if len(correct_mean_kvigs) > 1 and len(incorrect_mean_kvigs) > 1:
            t_stat, p_value = scipy_stats.ttest_ind(
                correct_mean_kvigs,
                incorrect_mean_kvigs,
                alternative="two-sided",  # 正确组应该更高
            )
        else:
            t_stat, p_value = 0.0, 1.0

        # AUC
        if len(set(labels)) > 1:
            auc = roc_auc_score(labels, all_mean_kvigs)
        else:
            auc = 0.5

        # d_eff 统计
        d_eff_array = np.array(all_d_effs)
        d_eff_valid = d_eff_array[np.isfinite(d_eff_array)]

        # 整体 KVIG 统计
        all_kvigs_flat = []
        for kvig_res in kvig_results:
            all_kvigs_flat.extend(kvig_res["kvig_values"])

        stats = {
            "correct_mean_kvig": np.mean(correct_mean_kvigs) if correct_mean_kvigs else 0.0,
            "incorrect_mean_kvig": np.mean(incorrect_mean_kvigs) if incorrect_mean_kvigs else 0.0,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "auc": float(auc),
            "d_eff_median": float(np.median(d_eff_valid)) if len(d_eff_valid) > 0 else 5.0,
            "d_eff_mean": float(np.mean(d_eff_valid)) if len(d_eff_valid) > 0 else 5.0,
            "d_eff_std": float(np.std(d_eff_valid)) if len(d_eff_valid) > 0 else 1.0,
            "overall_mean_kvig": float(np.mean(all_kvigs_flat)) if all_kvigs_flat else 0.0,
            "overall_std_kvig": float(np.std(all_kvigs_flat)) if all_kvigs_flat else 1.0,
            "correct_mean_length": float(np.mean(correct_lengths)) if correct_lengths else 300.0,
            "cohen_d": self._cohens_d(correct_mean_kvigs, incorrect_mean_kvigs),
        }

        return stats

    def _cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """计算 Cohen's d 效应量"""
        if len(group1) < 2 or len(group2) < 2:
            return 0.0
        n1, n2 = len(group1), len(group2)
        m1, m2 = np.mean(group1), np.mean(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        return float((m1 - m2) / pooled_std)

    def _validate(self, stats: Dict) -> bool:
        """
        验证 KVIG 信号有效性

        必须同时满足:
        1. p < 0.001（正确组 mean_KVIG 显著高于错误组）
        2. AUC ≥ 0.65
        3. d_eff 中位数 > 3
        """
        cal_cfg = self.cal_config

        checks = {
            "p_value": stats["p_value"] < cal_cfg.min_p_value_threshold,
            "auc": stats["auc"] >= cal_cfg.min_auc,
            "d_eff_median": stats["d_eff_median"] > cal_cfg.min_d_eff_median,
        }

        for name, passed in checks.items():
            status = "✓" if passed else "✗"
            logger.info(f"  Validation {name}: {status} "
                       f"(value={stats.get(name, 'N/A')}, "
                       f"threshold={getattr(cal_cfg, f'min_{name}' if f'min_{name}' in dir(cal_cfg) else name, 'N/A')})")

        all_passed = all(checks.values())

        if all_passed:
            logger.info("  ✓ All calibration checks PASSED")
        else:
            logger.warning("  ✗ Some calibration checks FAILED")

        return all_passed

    def _search_alpha_beta(self, model, trajectories: List[Dict]) -> Tuple[Dict, bool]:
        """★ 终极版：使用纯 Optuna 进行贝叶斯优化，完美避开 2GB 序列化崩溃 ★"""
        cal_cfg = self.cal_config
        original_alpha = self.kvig_computer.alpha
        original_beta = self.kvig_computer.beta

        logger.info("Pre-computing hidden states to CPU memory for Optuna...")
        all_hidden_states = self._precompute_hidden_states(model, trajectories)

        # 设置 Optuna 的日志级别，方便在终端看进度
        optuna.logging.set_verbosity(optuna.logging.INFO)

        def objective(trial):
            # 1. 贝叶斯算法建议参数
            alpha = trial.suggest_float("alpha", cal_cfg.alpha_search_range[0], cal_cfg.alpha_search_range[1])
            beta = trial.suggest_float("beta", cal_cfg.beta_search_range[0], cal_cfg.beta_search_range[1])
            
            self.kvig_computer.alpha = alpha
            self.kvig_computer.beta = beta

            # 2. 高速矩阵计算
            kvig_results = []
            for hs in all_hidden_states:
                hs_gpu = hs.to(model.device)
                result = self.kvig_computer.compute_trajectory(hs_gpu)
                kvig_results.append(result)
                del hs_gpu

            # 3. 计算 AUC
            stats = self._analyze_statistics(trajectories, kvig_results)
            
            # ★ 把当前 trial 算出来的全套指标挂载保存下来，方便选 best 时提取
            trial.set_user_attr("best_stats", stats)
            
            return stats["auc"]

        logger.info(f"Starting Bayesian Optimization (Pure Optuna in-process)...")
        logger.info(f"Search Space: α ∈ {cal_cfg.alpha_search_range}, β ∈ {cal_cfg.beta_search_range}")

        # 创建一个旨在 最大化 AUC 的 Study
        study = optuna.create_study(direction="maximize")
        
        # 开始搜索！30次迭代（因为单进程无通信开销，速度会非常快）
        study.optimize(objective, n_trials=30)

        # 提取最优结果
        best_trial = study.best_trial
        best_alpha = best_trial.params["alpha"]
        best_beta = best_trial.params["beta"]
        best_auc = best_trial.value
        best_stats = best_trial.user_attrs["best_stats"]

        logger.info(f"Optuna Search Finished! Best found: α={best_alpha:.3f}, β={best_beta:.3f}, AUC={best_auc:.4f}")

        # 恢复最优参数
        self.kvig_computer.alpha = best_alpha
        self.kvig_computer.beta = best_beta
        passed = self._validate(best_stats) if best_stats else False

        if not passed:
            logger.error("Calibration still FAILED AUC threshold after Bayesian Opt. Proceeding with caution.")

        # 及时清理这 40GB 的隐状态内存
        del all_hidden_states

        return best_stats, passed

    @torch.no_grad()
    def _precompute_hidden_states(
        self,
        model,
        trajectories: List[Dict],
    ) -> List[torch.Tensor]:
        """预计算所有轨迹的 hidden states（用于 α,β 搜索）"""
        model.eval()
        all_hidden = []

        for i, traj in enumerate(trajectories):
            full_ids = traj["full_ids"].unsqueeze(0).to(model.device)
            attention_mask = traj["full_attention_mask"].unsqueeze(0).to(model.device)
            prompt_length = traj["prompt_length"]

            outputs = model.model(
                input_ids=full_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

            # 取最后一层 hidden state 的 response 部分
            last_hidden = outputs.last_hidden_state[0, prompt_length:]  # (resp_len, d)
            all_hidden.append(last_hidden.cpu())

            if (i + 1) % 500 == 0:
                logger.info(f"  Pre-computed hidden states: {i+1}/{len(trajectories)}")

            # 清理 GPU 内存
            del outputs
            if i % 100 == 0:
                torch.cuda.empty_cache()

        return all_hidden

    def _log_results(self, results: Dict):
        """打印校准结果"""
        logger.info("=" * 60)
        logger.info("KVIG Calibration Results:")
        logger.info(f"  d_eff_threshold (d_eff median): {results['d_eff_threshold']:.4f}")
        logger.info(f"  T_ref (avg correct length): {results['t_ref']:.1f}")
        logger.info(f"  KVIG_mean: {results['kvig_mean']:.6f}")
        logger.info(f"  KVIG_std: {results['kvig_std']:.6f}")
        logger.info(f"  AUC: {results['auc']:.4f}")
        logger.info(f"  p-value: {results['p_value']:.2e}")
        logger.info(f"  Correct group mean KVIG: {results['correct_mean_kvig']:.6f}")
        logger.info(f"  Incorrect group mean KVIG: {results['incorrect_mean_kvig']:.6f}")
        logger.info(f"  α: {results['alpha']:.3f}, β: {results['beta']:.3f}")
        logger.info(f"  Validation: {'PASSED ✓' if results['validation_passed'] else 'FAILED ✗'}")
        logger.info("=" * 60)

    def _save_results(self, results: Dict):
        """保存校准结果到 JSON 文件"""
        output_path = self.cal_config.calibration_output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 确保所有值都是 JSON 可序列化的
        serializable = {}
        for k, v in results.items():
            if isinstance(v, (np.floating, np.integer)):
                serializable[k] = float(v)
            elif isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            else:
                serializable[k] = v

        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"Calibration results saved to {output_path}")


def load_calibration_results(path: str) -> Dict:
    """加载校准结果"""
    with open(path, "r") as f:
        return json.load(f)


def apply_calibration_to_config(
    calibration_results: Dict,
    kvig_config: KVIGConfig,
) -> KVIGConfig:
    """
    将校准结果应用到 KVIG 配置

    Args:
        calibration_results: 校准输出
        kvig_config: KVIG 配置

    Returns:
        更新后的 KVIG 配置
    """
    kvig_config.d_eff_threshold = calibration_results["d_eff_threshold"]
    kvig_config.t_ref = calibration_results["t_ref"]
    kvig_config.kvig_mean = calibration_results["kvig_mean"]
    kvig_config.kvig_std = calibration_results["kvig_std"]
    kvig_config.alpha = calibration_results["alpha"]
    kvig_config.beta = calibration_results["beta"]

    logger.info("Calibration results applied to KVIG config:")
    logger.info(f"  d_eff_threshold = {kvig_config.d_eff_threshold}")
    logger.info(f"  T_ref = {kvig_config.t_ref}")
    logger.info(f"  α = {kvig_config.alpha}, β = {kvig_config.beta}")

    return kvig_config