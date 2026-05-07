"""ARC v9 Step 1: K-sample stochastic consistency generator.

Per claude's diagnosis (claude_note.md, 2026-04-13): on HallusionBench,
first-order text confidence (margin, seq_lp, agree) carries near-zero
signal about correctness because hallucinations are *high-confidence wrong*.
Selective classification therefore cannot beat fair_fixed at any coverage
with the existing v4_1 features (best CV-AUC = 0.666).

This script generates K stochastic samples per (image, question) for a
benchmark + model, exactly matching gen_actions.py's prompts and parsing.
The K samples are used downstream to compute uncertainty signals that are
*orthogonal* to first-order confidence:

  - self_consistency       = max_a count(samples that parsed as a) / K
  - predictive_entropy_t1  = -sum_y p1[y] log p1[y]    over first-token softmax
  - semantic_entropy       = -sum_a (n_a/K) log(n_a/K) over parsed answers
  - sample_disagreement    = 1 - self_consistency
  - mean_sample_margin     = mean of (p1 - p2) across the K samples
  - margin_std             = std of  (p1 - p2) across the K samples
  - mean_seq_logprob       = mean across samples
  - seq_logprob_std        = std across samples

These are exactly the quantities Kuhn et al. NeurIPS 2023 / Farquhar et al.
Nature 2024 found to be the strongest hallucination indicators in NLG.

Output:
  ./data/consistency_v9/{bench}__{model_tag}.json

CLI:
  python gen_consistency_v9.py --bench HallusionBench --model 3B --K 10
  python gen_consistency_v9.py --bench POPE --model 3B --K 10 --max_samples 600
"""
import argparse, json, math, os, sys, time

MODEL_3B = "./models/Qwen2.5-VL-3B-Instruct"
MODEL_7B = "./models/Qwen2.5-VL-7B-Instruct"
OUT_DIR  = "./data/consistency_v9"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_loader import iter_bench  # noqa
from gen_actions import build_prompt_direct, parse_answer, build_inputs, SYSTEM_VL, SYSTEM_TEXT  # noqa


def stochastic_generate_K(model, tokenizer, inputs, K, max_new, temperature, top_p, seed_base=12345):
    """Generate K stochastic samples in one call via num_return_sequences.

    Uses top_k=50 + top_p + temperature with a guard rewrite of any rows that
    would yield NaN/Inf (Qwen2.5-VL in float16 occasionally emits one extreme
    logit row; without top_k=50 the sampler can hit a CUDA assert)."""
    import torch
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    il = inputs['input_ids'].shape[1]
    torch.manual_seed(seed_base)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=50,
            num_return_sequences=K,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            renormalize_logits=True,
        )
    seqs = out.sequences[:, il:]                         # (K, T)
    texts = []
    for k in range(K):
        texts.append(tokenizer.decode(seqs[k], skip_special_tokens=True).strip())

    # First-token softmax distribution: out.scores[0] has shape (K, V)
    p1_per_k = []
    p2_per_k = []
    margin_per_k = []
    seq_lp_per_k = []
    if len(out.scores) > 0:
        first = out.scores[0].float()                    # (K, V)
        probs0 = torch.softmax(first, dim=-1)
        topv, _ = torch.topk(probs0, 2, dim=-1)
        for k in range(K):
            p1_per_k.append(float(topv[k, 0]))
            p2_per_k.append(float(topv[k, 1]))
            margin_per_k.append(float(topv[k, 0] - topv[k, 1]))

        # mean per-token logprob
        eos = tokenizer.eos_token_id
        T = seqs.shape[1]
        for k in range(K):
            lp = 0.0; n = 0
            for t in range(min(T, len(out.scores))):
                tok = int(seqs[k, t].item())
                if tok == eos:
                    break
                ls = torch.log_softmax(out.scores[t][k].float(), dim=-1)
                lp += float(ls[tok])
                n += 1
            seq_lp_per_k.append(lp / max(1, n))
    else:
        p1_per_k = [0.0]*K; p2_per_k = [0.0]*K
        margin_per_k = [0.0]*K; seq_lp_per_k = [0.0]*K
    return texts, p1_per_k, p2_per_k, margin_per_k, seq_lp_per_k


def shannon_entropy_dist(counts):
    s = sum(counts.values())
    if s == 0: return 0.0
    h = 0.0
    for v in counts.values():
        if v > 0:
            p = v / s
            h -= p * math.log(p)
    return h


def run(bench, model_tag, K, temperature, top_p, max_samples):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f'{bench}__{model_tag}__K{K}.json')
    if os.path.exists(out_path):
        print(f'[skip exists] {out_path}'); return

    path = MODEL_3B if model_tag == '3B' else MODEL_7B
    print(f'Loading {model_tag} from {path}')
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    # bfloat16 is more numerically stable than float16 for the sampler
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path, torch_dtype=dtype, device_map='cuda', trust_remote_code=True)
    model.eval()
    print(f'Model loaded in {time.time()-t0:.1f}s')

    out_rows = {}
    n = 0
    t0 = time.time()
    for r in iter_bench(bench):
        if max_samples and n >= max_samples: break
        try:
            system = SYSTEM_TEXT if r['task'] == 'fever' else SYSTEM_VL
            prompt = build_prompt_direct(r)
            inp = build_inputs(processor, r['image'], system, prompt)
            texts, p1, p2, margins, seq_lps = stochastic_generate_K(
                model, tokenizer, inp, K=K, max_new=8,
                temperature=temperature, top_p=top_p,
                seed_base=12345 + (n * 7919))
            parsed = [parse_answer(t, r['task'], r['label_set'], r.get('choices')) for t in texts]
            counts = {}
            for a in parsed:
                counts[a] = counts.get(a, 0) + 1
            top_a, top_n = max(counts.items(), key=lambda kv: kv[1])
            self_consistency = top_n / K
            semantic_entropy = shannon_entropy_dist(counts)
            mean_p1 = sum(p1) / K
            mean_margin = sum(margins) / K
            margin_std = (sum((x - mean_margin) ** 2 for x in margins) / K) ** 0.5
            mean_lp = sum(seq_lps) / K
            lp_std = (sum((x - mean_lp) ** 2 for x in seq_lps) / K) ** 0.5
            # First-token predictive entropy averaged across K samples
            t_pred_ent = 0.0
            for k in range(K):
                p_yes = p1[k]; p_no = p2[k]
                # approximate two-class entropy from top-2
                if p_yes > 0: t_pred_ent -= p_yes * math.log(max(p_yes, 1e-12))
                if p_no  > 0: t_pred_ent -= p_no  * math.log(max(p_no,  1e-12))
            t_pred_ent /= K
            out_rows[r['sample_id']] = {
                'sample_id': r['sample_id'],
                'benchmark': bench,
                'category':  r.get('category'),
                'task':      r['task'],
                'samples':   parsed,
                'top_action':       top_a,
                'self_consistency': self_consistency,
                'semantic_entropy': semantic_entropy,
                'sample_disagreement': 1.0 - self_consistency,
                'mean_p1':           mean_p1,
                'mean_margin':       mean_margin,
                'margin_std':        margin_std,
                'mean_seq_logprob':  mean_lp,
                'seq_logprob_std':   lp_std,
                'pred_entropy_t1':   t_pred_ent,
                'gt': r['gt'],
            }
            n += 1
            if n % 50 == 0:
                dt = time.time() - t0
                print(f'  [{n}] {dt:.0f}s ({dt/n:.2f}s/it)')
        except Exception as e:
            print(f'  [err sid={r.get("sample_id")}] {type(e).__name__}: {e}')
            # If CUDA is poisoned, abort hard so we get a clean restart
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception as e2:
                print(f'  [fatal cuda] {e2}')
                break

    json.dump(out_rows, open(out_path, 'w'), indent=1)
    print(f'\nSaved {len(out_rows)} rows to {out_path}')
    print(f'Total time: {time.time()-t0:.0f}s')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bench', required=True, choices=['POPE', 'HallusionBench', 'A-OKVQA', 'FEVER'])
    p.add_argument('--model', default='3B', choices=['3B', '7B'])
    p.add_argument('--K', type=int, default=10)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top_p', type=float, default=0.95)
    p.add_argument('--max_samples', type=int, default=0)
    args = p.parse_args()
    run(args.bench, args.model, args.K, args.temperature, args.top_p, args.max_samples)


if __name__ == '__main__':
    main()
