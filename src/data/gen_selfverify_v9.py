"""ARC v9 backup: visual self-verification probe.

For each (image, question, greedy_answer) we re-prompt the SAME model with:

  "Question: <q>
   Proposed answer: <a>
   Look carefully at the image. Is the proposed answer correct?
   Answer with only 'yes' or 'no'."

We record: P(yes), P(no), margin, predicted verdict. The 'yes' probability
under self-verification is a known orthogonal signal to first-order
confidence (Manakul et al. SelfCheckGPT 2023; Kadavath et al. 2022 P(IK)).
On HallusionBench, the hypothesis is that confidently-wrong answers will
elicit a non-trivially lowered P(yes) under explicit re-grounding.

This is computed off the GREEDY answer from gen_actions.py output, so the
script reads the action_tables for the direct/3B answers and only re-prompts
the (image, q, a) triples.

Output: ./data/consistency_v9/{bench}__selfverify__3B.json
"""
import argparse, json, os, sys, time

MODEL_3B = "./models/Qwen2.5-VL-3B-Instruct"
MODEL_7B = "./models/Qwen2.5-VL-7B-Instruct"
OUT_DIR = "./data/consistency_v9"
ACTION_DIR = "./data/action_tables"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_loader import iter_bench  # noqa
from gen_actions import build_inputs, SYSTEM_VL, SYSTEM_TEXT  # noqa

PROMPT_VERIFY_VL = ("Question: {q}\n"
                    "Proposed answer: {a}\n\n"
                    "Look carefully at the image. Is the proposed answer correct? "
                    "Answer with only 'yes' or 'no'.")
PROMPT_VERIFY_FEVER = ("Claim: {q}\n"
                       "Proposed verdict: {a}\n\n"
                       "Is the proposed verdict correct? Answer with only 'yes' or 'no'.")


def yes_no_probs(model, tokenizer, inputs):
    import torch
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    il = inputs['input_ids'].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=2, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id,
                             return_dict_in_generate=True, output_scores=True)
    seq = out.sequences[0, il:]
    text = tokenizer.decode(seq, skip_special_tokens=True).strip().lower()
    if len(out.scores) == 0:
        return text, 0.0, 0.0
    logits = out.scores[0][0].float()
    probs = torch.softmax(logits, dim=-1)
    yes_ids = [tokenizer.encode('yes', add_special_tokens=False)[0],
               tokenizer.encode(' yes', add_special_tokens=False)[0],
               tokenizer.encode('Yes', add_special_tokens=False)[0]]
    no_ids = [tokenizer.encode('no', add_special_tokens=False)[0],
              tokenizer.encode(' no', add_special_tokens=False)[0],
              tokenizer.encode('No', add_special_tokens=False)[0]]
    p_yes = float(sum(probs[i] for i in set(yes_ids)))
    p_no = float(sum(probs[i] for i in set(no_ids)))
    return text, p_yes, p_no


def run(bench, model_tag):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f'{bench}__selfverify__{model_tag}.json')
    if os.path.exists(out_path):
        print(f'[skip exists] {out_path}'); return

    # Load greedy answers from action_tables
    greedy_path = os.path.join(ACTION_DIR, f'{bench}__direct__{model_tag}.json')
    greedy = json.load(open(greedy_path))
    sid_to_ans = {r['sample_id']: r['answer'] for r in greedy['results']}
    print(f'Loaded {len(sid_to_ans)} greedy answers from {greedy_path}')

    path = MODEL_3B if model_tag == '3B' else MODEL_7B
    print(f'Loading {model_tag} from {path}')
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path, torch_dtype=dtype, device_map='cuda', trust_remote_code=True)
    model.eval()

    out_rows = {}
    n = 0; t0 = time.time()
    for r in iter_bench(bench):
        sid = r['sample_id']
        ans = sid_to_ans.get(sid)
        if ans is None: continue
        try:
            system = SYSTEM_TEXT if r['task'] == 'fever' else SYSTEM_VL
            template = PROMPT_VERIFY_FEVER if r['task'] == 'fever' else PROMPT_VERIFY_VL
            prompt = template.format(q=r['question'], a=ans)
            inp = build_inputs(processor, r['image'], system, prompt)
            text, p_yes, p_no = yes_no_probs(model, tokenizer, inp)
            out_rows[sid] = {
                'sample_id': sid, 'benchmark': bench, 'task': r['task'],
                'greedy_answer': ans, 'verify_text': text,
                'verify_p_yes': p_yes, 'verify_p_no': p_no,
                'verify_margin_yes_minus_no': p_yes - p_no,
                'gt': r['gt'],
            }
            n += 1
            if n % 100 == 0:
                dt = time.time() - t0
                print(f'  [{n}] {dt:.0f}s ({dt/n:.2f}s/it)')
        except Exception as e:
            print(f'  [err sid={sid}] {type(e).__name__}: {e}')
            try:
                import torch
                torch.cuda.synchronize()
            except Exception:
                break
    json.dump(out_rows, open(out_path, 'w'), indent=1)
    print(f'\nSaved {len(out_rows)} rows to {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bench', required=True)
    p.add_argument('--model', default='3B')
    args = p.parse_args()
    run(args.bench, args.model)


if __name__ == '__main__':
    main()
