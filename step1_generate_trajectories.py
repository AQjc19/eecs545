"""
Step 1: Generate correct GSM8K trajectories from Qwen-2.5-1.5B-Instruct.
Produces ~500 correct trajectories for curriculum construction.
"""

import os, json, re, random, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def build_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": "Solve the math problem step by step. Put the final answer in \\boxed{}."},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def extract_answer(text: str) -> str:
    # Try \boxed{...}
    m = re.findall(r'\\boxed\{([^}]*)\}', text)
    if m:
        return m[-1].strip().replace(",", "")
    # Try "#### answer"
    m = re.findall(r'####\s*([\-\d,.]+)', text)
    if m:
        return m[-1].strip().replace(",", "")
    # Last number
    nums = re.findall(r'[\-\d,.]+', text)
    if nums:
        return nums[-1].replace(",", "")
    return ""

def normalize_answer(ans: str) -> str:
    ans = ans.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        return str(int(float(ans)))
    except:
        return ans

def check_answer(pred: str, gt: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gt)

def extract_gt_answer(solution: str) -> str:
    m = re.findall(r'####\s*([\-\d,.]+)', solution)
    if m:
        return m[-1].strip().replace(",", "")
    return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", type=str, default="./anchor_data/trajectories.jsonl")
    parser.add_argument("--target_correct", type=int, default=500)
    parser.add_argument("--max_attempts", type=int, default=3000)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=True,
    ).to("cuda").eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Loading GSM8K dataset")
    ds = load_dataset("/fs1/home/hanrui/hanrui/drought/model/gsm8k_data", "main", split="train")

    correct_trajs = []
    attempted = 0
    problem_indices = list(range(len(ds)))
    random.shuffle(problem_indices)

    print(f"Generating trajectories (target: {args.target_correct} correct)")

    idx_ptr = 0
    while len(correct_trajs) < args.target_correct and attempted < args.max_attempts:
        # Build batch
        batch_problems = []
        batch_gts = []
        for _ in range(args.batch_size):
            if idx_ptr >= len(problem_indices):
                idx_ptr = 0
                random.shuffle(problem_indices)
            pi = problem_indices[idx_ptr]
            idx_ptr += 1
            batch_problems.append(ds[pi])
            batch_gts.append(extract_gt_answer(ds[pi]["answer"]))

        prompts = [build_prompt(p["question"], tokenizer) for p in batch_problems]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=1024).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        for i in range(len(batch_problems)):
            attempted += 1
            prompt_len = inputs["input_ids"][i].ne(tokenizer.pad_token_id).sum().item()
            resp_ids = outputs[i][prompt_len:]
            resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True)
            pred = extract_answer(resp_text)
            gt = batch_gts[i]

            if check_answer(pred, gt):
                traj = {
                    "question": batch_problems[i]["question"],
                    "ground_truth": gt,
                    "response": resp_text,
                    "response_ids": resp_ids.tolist(),
                    "prompt": prompts[i],
                    "predicted": pred,
                }
                correct_trajs.append(traj)

        if attempted % 100 == 0:
            print(f"  Attempted: {attempted}, Correct: {len(correct_trajs)}, "
                  f"Rate: {len(correct_trajs)/max(attempted,1):.1%}")

        if len(correct_trajs) >= args.target_correct:
            break

    print(f"\nDone. {len(correct_trajs)} correct trajectories from {attempted} attempts "
          f"({len(correct_trajs)/max(attempted,1):.1%})")

    with open(args.output, "w") as f:
        for t in correct_trajs:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()