import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

from config import KVIGConfig, ModelConfig

@dataclass
class KVIGState:
    h_prev: Optional[torch.Tensor] = None       
    delta_h_prev: Optional[torch.Tensor] = None 
    entropy_prev: Optional[float] = None        
    mcig_env_prev: float = 0.0                  
    step: int = 0                               

    def is_first_step(self) -> bool:
        return self.step == 0

class KVIGComputer:
    """隐流形动力学计算器 (MCIG Engine)"""
    def __init__(self, kvig_config: KVIGConfig, model_config: ModelConfig):
        self.eps = getattr(kvig_config, "eps", 1e-8)
        self.hidden_size = model_config.hidden_size
        self.decay = 0.85  # 包络衰减系数 γ

    @torch.no_grad()
    def compute_step(self, h_t: torch.Tensor, logits_t: torch.Tensor, state: KVIGState, **kwargs) -> Tuple[float, KVIGState]:
        h_t = h_t.detach().float().squeeze()
        logits_t = logits_t.detach().float().squeeze()
        
        probs = F.softmax(logits_t, dim=-1).clamp(min=self.eps)
        H_t = -(probs * probs.log()).sum().item()
        norm_t = h_t.norm().item()

        if state.is_first_step() or state.h_prev is None or state.entropy_prev is None:
            mcig_raw = 0.0
            new_delta_h = None
        else:
            causal_drop = max(0.0, state.entropy_prev - H_t)
            delta_h_t = h_t - state.h_prev
            if state.delta_h_prev is None:
                cos_sim = F.cosine_similarity(h_t.unsqueeze(0), state.h_prev.unsqueeze(0), eps=self.eps).item()
                kappa = max(0.0, 1.0 - cos_sim)
            else:
                cos_sim = F.cosine_similarity(delta_h_t.unsqueeze(0), state.delta_h_prev.unsqueeze(0), eps=self.eps).item()
                kappa = max(0.0, 1.0 - cos_sim)
            mcig_raw = kappa * causal_drop * norm_t
            new_delta_h = delta_h_t.cpu()

        mcig_env = max(mcig_raw, state.mcig_env_prev * self.decay)
        new_state = KVIGState(h_prev=h_t.cpu(), delta_h_prev=new_delta_h, entropy_prev=H_t, mcig_env_prev=mcig_env, step=state.step + 1)
        return mcig_env, new_state

    @torch.no_grad()
    def compute_trajectory_from_model(self, model, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, prompt_length: int = 0, **kwargs) -> Dict:
        device = next(model.parameters()).device
        if input_ids.dim() == 1: input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1: attention_mask = attention_mask.unsqueeze(0)
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        resp_len = input_ids.shape[1] - prompt_length
        
        if resp_len <= 1:
            empty = [0.0] * max(0, resp_len)
            return {"kvig_values": empty, "mean_kvig": 0.0, "std_kvig": 0.0}

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
        last_hidden = outputs.hidden_states[-1][0].float()
        logits = outputs.logits[0].float()
        
        h_resp, logits_resp = last_hidden[prompt_length:], logits[prompt_length:]
        h_prev_first, logits_prev_first = last_hidden[prompt_length - 1], logits[prompt_length - 1]
        
        h_all = torch.cat([h_prev_first.unsqueeze(0), h_resp], dim=0)
        all_logits = torch.cat([logits_prev_first.unsqueeze(0), logits_resp], dim=0)

        log_probs = F.log_softmax(all_logits, dim=-1)
        probs = log_probs.exp()
        entropies = -(probs * log_probs).sum(dim=-1).cpu().numpy()
        causal_drop = np.maximum(0.0, entropies[:-1] - entropies[1:])
        
        h_norms = h_resp.norm(dim=-1).cpu().numpy()
        mcig_raw = np.zeros(resp_len, dtype=np.float32)
        
        if resp_len >= 2:
            delta_h = h_all[1:] - h_all[:-1]
            cos_t0 = F.cosine_similarity(h_resp[0].unsqueeze(0), h_prev_first.unsqueeze(0), eps=self.eps).item()
            mcig_raw[0] = causal_drop[0] * h_norms[0] * max(0.0, 1.0 - cos_t0)
            
            kappa_vec = 1.0 - F.cosine_similarity(delta_h[1:], delta_h[:-1], dim=-1, eps=self.eps)
            kappa_vec = F.relu(kappa_vec).cpu().numpy()
            mcig_raw[1:] = kappa_vec * causal_drop[1:] * h_norms[1:]

        mcig_env = np.zeros_like(mcig_raw)
        if len(mcig_raw) > 0:
            mcig_env[0] = mcig_raw[0]
            for t in range(1, len(mcig_raw)):
                mcig_env[t] = max(mcig_raw[t], mcig_env[t-1] * self.decay)

        del outputs, logits, last_hidden, h_resp, logits_resp, h_all, all_logits, log_probs, probs
        if resp_len >= 2: del delta_h
        torch.cuda.empty_cache()

        mcig_env = np.nan_to_num(mcig_env, nan=0.0, posinf=0.0, neginf=0.0)
        kvig_values = mcig_env.tolist()
        return {
            "kvig_values": kvig_values,
            "mean_kvig": float(np.mean(kvig_values)) if kvig_values else 0.0,
            "std_kvig": float(np.std(kvig_values)) if kvig_values else 0.0,
            # 兼容遗留返回
            "d_eff_values": [0.0] * resp_len, "V_values": [0.0] * resp_len, "delta_S_values": [0.0] * resp_len
        }

def compute_kvig_batch(kvig_computer, model, batch_input_ids, batch_attention_masks, prompt_lengths, **kwargs):
    return [kvig_computer.compute_trajectory_from_model(model, input_ids=ids, attention_mask=mask, prompt_length=pl) 
            for ids, mask, pl in zip(batch_input_ids, batch_attention_masks, prompt_lengths)]