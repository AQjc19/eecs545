import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from transformers import AutoTokenizer

# ==========================================
# 模拟 W&B 风格，设置高级学术绘图样式
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14, 
    'axes.labelsize': 16, 
    'axes.titlesize': 18,
    'legend.fontsize': 12,
    'pdf.fonttype': 42 # 保证导出 pdf 时的文字可编辑
})

def load_data(pt_path):
    print(f"[*] Loading raw trajectory data from {pt_path}...")
    return torch.load(pt_path, map_location='cpu', weights_only=False)

def plot_trajectory_kde(sft_data, out_dir):
    """【W&B Histogram 平替】: 轨迹级别分布重叠度分析"""
    print("[*] Generating Trajectory-Level KDE Plot...")
    correct_means = [d.get('mean_mcig', d.get('mean_mcig', 0)) for d in sft_data if d['is_correct']]
    incorrect_means = [d.get('mean_mcig', d.get('mean_mcig', 0)) for d in sft_data if not d['is_correct']]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(correct_means, fill=True, color='#2ecc71', label=f'Correct (n={len(correct_means)})', alpha=0.5, ax=ax)
    sns.kdeplot(incorrect_means, fill=True, color='#e74c3c', label=f'Incorrect (n={len(incorrect_means)})', alpha=0.5, ax=ax)
    
    # 画出均值线
    ax.axvline(np.mean(correct_means), color='darkgreen', linestyle='--', linewidth=2)
    ax.axvline(np.mean(incorrect_means), color='darkred', linestyle='--', linewidth=2)

    ax.set_title("Trajectory-Level Mean MCIG Distribution\n(Explains Low AUC)")
    ax.set_xlabel("Mean MCIG per Trajectory")
    ax.set_ylabel("Density")
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "1_trajectory_distribution.png"), dpi=300)
    plt.close()

def plot_token_mcig_curve(sft_data, tokenizer, out_dir, sample_idx=0):
    """【W&B Line Chart 平替】: Token 级别波动折线图"""
    print("[*] Generating Token-Level Fluctuation Curve...")
    
    # 挑选一条长度适中的正确轨迹
    valid_trajs = [d for d in sft_data if d['is_correct'] and 100 < d['response_length'] < 200]
    if not valid_trajs: valid_trajs = sft_data
    
    traj = valid_trajs[sample_idx]
    # 兼容旧数据的读取
    mcig_vals = traj.get('mcig_values', traj.get('kvig_values', []))
    tokens = tokenizer.convert_ids_to_tokens(traj['full_ids'][traj['prompt_length']:].tolist())
    
    # 截取前 80 个 token 保证肉眼可见
    plot_len = min(80, len(mcig_vals))
    x = np.arange(plot_len)
    y = mcig_vals[:plot_len]
    
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(x, y, marker='o', linestyle='-', color='#3498db', markersize=6, linewidth=2)
    
    # 标记 Skip 阈值线 (后 20%)
    threshold = np.percentile(mcig_vals, 20)
    
    # 把应该被 skip 的点标红
    for i in range(plot_len):
        if y[i] <= threshold:
            ax.plot(i, y[i], marker='X', color='#e74c3c', markersize=10)
            
    ax.axhline(threshold, color='#e74c3c', linestyle='--', linewidth=2, label=f'Skip Threshold (Bottom 20% = {threshold:.2f})')
    
    # 替换 tokenizer 特殊字符，显示单词
    clean_tokens = [t.replace('Ġ', '').replace('Ċ', '\\n') for t in tokens[:plot_len]]
    ax.set_xticks(x)
    ax.set_xticklabels(clean_tokens, rotation=60, ha='right', fontsize=10)
    
    # 算一下这条轨迹的 CV
    mean_k = np.mean(mcig_vals)
    std_k = np.std(mcig_vals)
    cv = std_k / (abs(mean_k) + 1e-8)

    # 打印在标题里
    ax.set_title(f"Token-Level MCIG Signal (Trajectory CV = {cv:.2f})")
    ax.set_ylabel("MCIG Score")
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "2_token_fluctuation.png"), dpi=300)
    plt.close()

def generate_semantic_heatmap_html(sft_data, tokenizer, out_dir, num_samples=10):
    """【W&B HTML Table 平替】: 语义对齐文本高亮热力图"""
    print("[*] Generating Semantic Text Heatmap (HTML)...")
    valid_trajs = [d for d in sft_data if d['is_correct']]
    
    html = """
    <html><head><meta charset="utf-8"><style>
        body { font-family: 'Courier New', monospace; padding: 30px; background: #f8f9fa; color: #2c3e50;}
        h1 { text-align: center; }
        .traj-box { background: white; padding: 20px; margin-bottom: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .token { padding: 3px 2px; border-radius: 4px; border: 1px solid transparent; font-size: 16px; transition: 0.2s;}
        .token:hover { border: 1px solid #34495e; cursor: help; transform: scale(1.1);}
        .legend { background: white; padding: 15px; border-radius: 10px; margin-bottom: 20px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
        .color-box { display: inline-block; width: 30px; height: 15px; margin: 0 10px; vertical-align: middle; border: 1px solid #ccc;}
    </style></head><body>
    <h1>SPARK v2: Token-Level Semantic Alignment</h1>
    <div class="legend">
        <strong>Color Legend:</strong><br><br>
        <span class="color-box" style="background:rgba(231, 76, 60, 0.8);"></span> High MCIG (Crucial Step / Keep)<br>
        <span class="color-box" style="background:rgba(52, 152, 219, 0.8);"></span> Low MCIG (Redundant / <b>SKIP</b> Candidate)
    </div>
    """
    
    for i, traj in enumerate(valid_trajs[:num_samples]):
        mcig_vals = np.array(traj.get('mcig_values', traj.get('kvig_values', [])))
        ids = traj['full_ids'][traj['prompt_length']:]
        
        if len(mcig_vals) == 0: continue
        
        # 归一化 MCIG 到 0-1 (截断极端值)
        p10, p90 = np.percentile(mcig_vals, 10), np.percentile(mcig_vals, 90)
        norm_mcig = np.clip((mcig_vals - p10) / (p90 - p10 + 1e-8), 0, 1)
        
        html += f"<div class='traj-box'><h3>Trajectory #{i+1} (Tokens: {len(ids)}, Answer: Correct)</h3><p>"
        
        for j, token_id in enumerate(ids):
            if j >= len(norm_mcig): break
            token_str = tokenizer.decode([token_id]).replace('<', '&lt;').replace('>', '&gt;').replace('\n', '↵<br>')
            
            val = norm_mcig[j]
            # 生成红(高分) -> 白(均分) -> 蓝(低分) 的渐变色
            if val > 0.5:
                intensity = (val - 0.5) * 2
                color = f"rgba(231, 76, 60, {intensity:.2f})" # Red
            else:
                intensity = (0.5 - val) * 2
                color = f"rgba(52, 152, 219, {intensity:.2f})" # Blue
                
            html += f"<span class='token' style='background-color:{color};' title='MCIG: {mcig_vals[j]:.4f}'>{token_str}</span>"
            
        html += "</p></div>"
        
    html += "</body></html>"
    
    with open(os.path.join(out_dir, "3_semantic_heatmap.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print("[*] Saved 3_semantic_heatmap.html (Open this in your browser!)")

if __name__ == "__main__":
    DATA_PATH = "./checkpoints/phase1/calibration_raw_data.pt"
    OUT_DIR = "./wandb_offline_viz"
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Please run the calibration script first.")
        exit(1)
        
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 加载数据与 Tokenizer
    sft_data = load_data(DATA_PATH)
    tokenizer = AutoTokenizer.from_pretrained("/fs1/home/hanrui/hanrui/drought/model/Qwen2.5-7B-Instruct") # 根据你的模型路径修改
    
    # 绘制图表
    plot_trajectory_kde(sft_data, OUT_DIR)
    plot_token_mcig_curve(sft_data, tokenizer, OUT_DIR, sample_idx=5)
    generate_semantic_heatmap_html(sft_data, tokenizer, OUT_DIR, num_samples=15)
    
    print(f"\n✅ All visualizations generated in '{OUT_DIR}/'")
    print("Download this folder to your local machine using SCP or VS Code to view the results.")