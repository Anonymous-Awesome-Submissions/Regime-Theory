"""ARC Step 2: action-set generator.

Produces one JSON per (benchmark, action, model) — a flat list of rows that
the unified action table will join on `(benchmark, sample_id, action)`.

Actions (Section 5):

  direct    — single greedy forward.  cost = 1 forward pass.
              records: first-token softmax p1/p2 over the benchmark's label
              set as `confidence`, and full-sequence mean per-token logprob
              as `seq_logprob`.
  recover   — 2-turn grounded recovery. 1st pass asks for context (object
              list for VL, self-verify sketch for text). 2nd pass prepends
              that context and asks for the final answer.  cost = 2 forward
              passes. records same signals as direct, plus the raw context
              string in `recover_context`.
  escalate  — same as direct but with the bigger model.

`answer_direct / recover_then_answer / escalate_to_stronger_model` map to
(direct, 3B) / (recover, 3B) / (direct, 7B) respectively.
`abstain` is handled offline in Step 3 — no generation needed.

Usage:
  python gen_actions.py --bench POPE --action direct --model 3B
  python gen_actions.py --bench A-OKVQA --action recover --model 3B
  python gen_actions.py --bench HallusionBench --action direct --model 7B
"""
import argparse, json, os, re, sys, time

MODEL_3B = "./models/Qwen2.5-VL-3B-Instruct"
MODEL_7B = "./models/Qwen2.5-VL-7B-Instruct"
OUT_DIR  = "./data/action_tables"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_loader import iter_bench  # noqa

SYSTEM_VL   = "You are a careful visual question answering assistant. Answer briefly and factually."
SYSTEM_TEXT = "You are a careful fact-checking assistant."

PROMPT_YESNO = "{q}\n\nAnswer with only 'yes' or 'no'."
PROMPT_MC    = "{q}\n\nChoose exactly one option from: {opts}\nReply with only the chosen option, no explanation."
PROMPT_FEVER = ("Claim: {q}\n\nDecide whether the claim is SUPPORTS, REFUTES, or NEI "
                "(not enough info). Reply with only one of: supports / refutes / nei.")
PROMPT_FOLIO = ("{q}\n\nReply with only one of: true / false / uncertain.")

# Recover context prompts (turn 1)
CTX_VL_YESNO = "List all objects, people, animals, and salient elements visible in the image. Output only a short comma-separated list."
CTX_VL_MC    = "Briefly describe what is visible in the image that is relevant to this question: {q}\nKeep it to 2 short sentences."
CTX_FEVER    = ("Claim: {q}\n\nIn 2 short sentences, list the key entities and facts in this claim and "
                "what would need to be verified to judge it.")
CTX_FOLIO    = ("{q}\n\nIn 2 short sentences, list the key facts asserted by the premises that bear on whether the conclusion follows.")

# Recover answer prompts (turn 2)
ANS_VL_YESNO = "Context (what is visible): {ctx}\n\nQuestion: {q}\nAnswer with only 'yes' or 'no'."
ANS_VL_MC    = "Context: {ctx}\n\nQuestion: {q}\nChoose exactly one option from: {opts}\nReply with only the chosen option."
ANS_FEVER    = ("Context (verification sketch): {ctx}\n\nClaim: {q}\n"
                "Decide whether the claim is SUPPORTS, REFUTES, or NEI.\n"
                "Reply with only one of: supports / refutes / nei.")
ANS_FOLIO    = ("Context (key facts): {ctx}\n\n{q}\n"
                "Reply with only one of: true / false / uncertain.")


def build_prompt_direct(row):
    if row['task'] == 'yesno':
        return PROMPT_YESNO.format(q=row['question'])
    if row['task'] == 'mc':
        return PROMPT_MC.format(q=row['question'], opts=' / '.join(row['choices']))
    if row['task'] == 'fever':
        return PROMPT_FEVER.format(q=row['question'])
    if row['task'] == 'folio':
        return PROMPT_FOLIO.format(q=row['question'])
    raise ValueError(row['task'])


def build_prompt_recover_ctx(row):
    if row['task'] == 'yesno':
        return CTX_VL_YESNO
    if row['task'] == 'mc':
        return CTX_VL_MC.format(q=row['question'])
    if row['task'] == 'fever':
        return CTX_FEVER.format(q=row['question'])
    if row['task'] == 'folio':
        return CTX_FOLIO.format(q=row['question'])
    raise ValueError(row['task'])


def build_prompt_recover_ans(row, ctx):
    if row['task'] == 'yesno':
        return ANS_VL_YESNO.format(ctx=ctx, q=row['question'])
    if row['task'] == 'mc':
        return ANS_VL_MC.format(ctx=ctx, q=row['question'], opts=' / '.join(row['choices']))
    if row['task'] == 'fever':
        return ANS_FEVER.format(ctx=ctx, q=row['question'])
    if row['task'] == 'folio':
        return ANS_FOLIO.format(ctx=ctx, q=row['question'])
    raise ValueError(row['task'])


def parse_answer(pred, task, label_set, choices=None):
    tl = pred.strip().lower()
    if task == 'yesno':
        m = re.search(r'\b(yes|no)\b', tl)
        return m.group(1) if m else 'no'
    if task == 'fever':
        for lab in ('supports', 'refutes', 'nei', 'support', 'refute'):
            if lab in tl:
                return 'supports' if lab.startswith('support') else ('refutes' if lab.startswith('refute') else 'nei')
        return 'nei'
    if task == 'folio':
        # Look for true/false/uncertain in any order; "true" can be a substring of "uncertain", check uncertain first
        if 'uncertain' in tl or 'unknown' in tl or 'cannot' in tl or 'insufficient' in tl:
            return 'uncertain'
        if 'false' in tl:
            return 'false'
        if 'true' in tl:
            return 'true'
        return 'uncertain'
    if task == 'mc':
        clean = re.sub(r'[^a-z0-9 ]', ' ', tl).strip()
        matches = [c.lower() for c in choices if c.lower() in clean]
        if len(matches) == 1:
            return matches[0]
        if matches:
            return max(matches, key=len)
        # fallback: first word token
        tok = clean.split()[0] if clean.split() else ''
        for c in choices:
            if c.lower().startswith(tok):
                return c.lower()
        return choices[0].lower() if choices else clean
    raise ValueError(task)


def score(pred_parsed, gt):
    return int(pred_parsed.strip().lower() == gt.strip().lower())


def build_inputs(processor, image, system, user):
    """VL input with optional image. For FEVER, image is None -> text-only path."""
    msgs = [{"role": "system", "content": system}]
    if image is not None:
        msgs.append({"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user},
        ]})
    else:
        msgs.append({"role": "user", "content": [{"type": "text", "text": user}]})
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if image is not None:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(msgs)
        return processor(text=[text], images=image_inputs, videos=video_inputs,
                         padding=True, return_tensors='pt')
    else:
        return processor(text=[text], padding=True, return_tensors='pt')


def generate(model, tokenizer, inputs, max_new=48):
    import torch
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    il = inputs['input_ids'].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True, output_scores=True,
        )
    seq = out.sequences[0, il:]
    text = tokenizer.decode(seq, skip_special_tokens=True).strip()
    # first-token softmax margin
    if len(out.scores) > 0:
        logits0 = out.scores[0][0].float()
        probs0 = torch.softmax(logits0, dim=-1)
        topk = torch.topk(probs0, 2)
        p1 = float(topk.values[0]); p2 = float(topk.values[1])
        margin = p1 - p2
        # mean per-token log-prob for whole generated sequence
        lp = 0.0; n = 0
        for t, s in enumerate(out.scores):
            tok = seq[t].item()
            if tok == tokenizer.eos_token_id: break
            ls = torch.log_softmax(s[0].float(), dim=-1)
            lp += float(ls[tok]); n += 1
        seq_logprob = lp / max(1, n)
    else:
        p1 = p2 = margin = 0.0; seq_logprob = 0.0
    return text, {'p1': p1, 'p2': p2, 'margin': margin, 'seq_logprob': seq_logprob,
                  'n_gen_tokens': int(seq.shape[0])}


def run(bench, action, model_tag, slice_str=None):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = f'__slice_{slice_str.replace(":", "_")}' if slice_str else ''
    out_path = os.path.join(OUT_DIR, f'{bench}__{action}__{model_tag}{suffix}.json')
    if os.path.exists(out_path):
        print(f'[skip exists] {out_path}'); return
    s_start, s_end = (None, None)
    if slice_str:
        a, b = slice_str.split(':')
        s_start = int(a) if a else None
        s_end = int(b) if b else None

    path = MODEL_3B if model_tag == '3B' else MODEL_7B
    print(f'Loading {model_tag} from {path}')
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.float16, device_map='cuda', trust_remote_code=True)
    model.eval()
    print(f'Model loaded in {time.time()-t0:.1f}s')

    rows_iter = iter_bench(bench)
    if slice_str:
        rows_iter = (r for i, r in enumerate(iter_bench(bench))
                     if (s_start is None or i >= s_start)
                     and (s_end is None or i < s_end))
    results = []
    n = 0
    t0 = time.time()
    for r in rows_iter:
        try:
            system = SYSTEM_TEXT if r['task'] in ('fever', 'folio') else SYSTEM_VL
            if action == 'recover':
                # Turn 1: get context (max 96 tokens)
                ctx_prompt = build_prompt_recover_ctx(r)
                inp1 = build_inputs(processor, r['image'], system, ctx_prompt)
                ctx_text, sig1 = generate(model, tokenizer, inp1, max_new=96)
                # Turn 2: answer with context
                ans_prompt = build_prompt_recover_ans(r, ctx_text)
                inp2 = build_inputs(processor, r['image'], system, ans_prompt)
                pred_text, sig2 = generate(model, tokenizer, inp2, max_new=32)
                parsed = parse_answer(pred_text, r['task'], r['label_set'], r.get('choices'))
                correct = score(parsed, r['gt'])
                row_out = {
                    'sample_id': r['sample_id'], 'benchmark': r['benchmark'],
                    'category':  r['category'], 'action': 'recover_then_answer',
                    'model': model_tag, 'raw_answer': pred_text, 'answer': parsed,
                    'gt': r['gt'], 'correct': correct,
                    'confidence_margin': sig2['margin'],
                    'seq_logprob': sig2['seq_logprob'],
                    'cost_tokens': sig1['n_gen_tokens'] + sig2['n_gen_tokens'],
                    'cost_passes': 2,
                    'recover_context': ctx_text[:500],
                    'ctx_margin': sig1['margin'],
                }
            else:  # direct or escalate
                prompt = build_prompt_direct(r)
                inp = build_inputs(processor, r['image'], system, prompt)
                pred_text, sig = generate(model, tokenizer, inp, max_new=32)
                parsed = parse_answer(pred_text, r['task'], r['label_set'], r.get('choices'))
                correct = score(parsed, r['gt'])
                action_name = 'escalate_to_stronger_model' if action == 'escalate' else 'answer_direct'
                row_out = {
                    'sample_id': r['sample_id'], 'benchmark': r['benchmark'],
                    'category':  r['category'], 'action': action_name,
                    'model': model_tag, 'raw_answer': pred_text, 'answer': parsed,
                    'gt': r['gt'], 'correct': correct,
                    'confidence_margin': sig['margin'],
                    'seq_logprob': sig['seq_logprob'],
                    'cost_tokens': sig['n_gen_tokens'], 'cost_passes': 1,
                }
            results.append(row_out)
            n += 1
            if n % 100 == 0:
                dt = time.time() - t0
                acc = sum(x['correct'] for x in results) / len(results)
                print(f'  [{n}] {dt:.0f}s acc={acc:.3f}')
        except Exception as e:
            print(f'  ERR {r.get("sample_id")}: {e}')
            continue
    dt = time.time() - t0
    acc = sum(x['correct'] for x in results) / max(1, len(results))
    print(f'Done: {len(results)} samples in {dt:.0f}s, acc={acc:.3f}')
    json.dump({'bench': bench, 'action': action, 'model': model_tag,
               'n': len(results), 'acc': acc, 'results': results},
              open(out_path, 'w'), indent=2)
    print(f'Saved {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bench', required=True,
                   choices=['POPE', 'POPE_ext', 'HallusionBench', 'A-OKVQA', 'FEVER', 'FOLIO'])
    p.add_argument('--action', required=True, choices=['direct', 'recover', 'escalate'])
    p.add_argument('--model', required=True, choices=['3B', '7B'])
    p.add_argument('--slice', default=None,
                   help='Optional A:B slice into iter_bench (e.g. 0:4500)')
    args = p.parse_args()
    # Sanity: escalate must use 7B; direct/recover in this plan use 3B.
    if args.action == 'escalate' and args.model != '7B':
        print('warn: escalate with non-7B model')
    run(args.bench, args.action, args.model, args.slice)


if __name__ == '__main__':
    main()
