"""ARC v9 backup signal: modality-bypass / visual-vs-blind probe.

Diagnosis (claude_note.md): on Hallu, semantic entropy from K stochastic
samples gives only AUC=0.658 — essentially the same as first-order text
margin. Hallucinations are stable across decoding seeds because they are
*systematic visual misreadings*, not stochastic noise.

A genuinely orthogonal signal: run the same question through the same model
with (a) the real image and (b) a black 224x224 image. If the answer is the
same, the model is not using visual evidence — i.e. it's relying on prior
language knowledge, which is the defining failure mode of visual
hallucination on HallusionBench (Liu et al. 2023, Sec. 3 — "language prior
overrides visual content").

Records (per sample):
  blind_answer            - answer when image is blacked out
  greedy_answer           - answer with the real image (from action_tables)
  blind_disagrees         - 1 if blind_answer != greedy_answer (good sign)
  blind_p_first_token     - first-token probability of blind answer
  blind_seq_logprob       - mean per-token logprob of blind answer
  blind_yes_minus_no      - p(yes) - p(no) under blind run

This is a known debiasing signal in VQA (Niu et al. CVPR 2021
"Counterfactual VQA: A cause-effect look at language bias") used here for
the opposite purpose: as a *feature* to detect when the model is biased.

Output: ./data/consistency_v9/{bench}__blind__3B.json
"""
import argparse, json, os, sys, time

MODEL_3B = "./models/Qwen2.5-VL-3B-Instruct"
MODEL_7B = "./models/Qwen2.5-VL-7B-Instruct"
OUT_DIR  = "./data/consistency_v9"
ACTION_DIR = "./data/action_tables"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_loader import iter_bench  # noqa
from gen_actions import (build_inputs, build_prompt_direct, parse_answer,  # noqa
                         SYSTEM_VL, SYSTEM_TEXT)


def make_blank_image():
    from PIL import Image
    return Image.new('RGB', (224, 224), (0, 0, 0))


def generate_with_signals(model, tokenizer, inputs, max_new=8):
    import torch
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    il = inputs['input_ids'].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id,
                             return_dict_in_generate=True, output_scores=True)
    seq = out.sequences[0, il:]
    text = tokenizer.decode(seq, skip_special_tokens=True).strip()
    p1 = p2 = margin = 0.0; lp = 0.0
    p_yes = p_no = 0.0
    if len(out.scores) > 0:
        first = out.scores[0][0].float()
        probs = torch.softmax(first, dim=-1)
        topv, _ = torch.topk(probs, 2)
        p1, p2 = float(topv[0]), float(topv[1])
        margin = p1 - p2
        # Yes/no token IDs
        yes_ids = list({tokenizer.encode('yes', add_special_tokens=False)[0],
                        tokenizer.encode(' yes', add_special_tokens=False)[0],
                        tokenizer.encode('Yes', add_special_tokens=False)[0]})
        no_ids = list({tokenizer.encode('no', add_special_tokens=False)[0],
                       tokenizer.encode(' no', add_special_tokens=False)[0],
                       tokenizer.encode('No', add_special_tokens=False)[0]})
        p_yes = float(sum(probs[i] for i in yes_ids))
        p_no  = float(sum(probs[i] for i in no_ids))
        nT = 0
        for t in range(min(len(out.scores), seq.shape[0])):
            tok = int(seq[t].item())
            if tok == tokenizer.eos_token_id: break
            ls = torch.log_softmax(out.scores[t][0].float(), dim=-1)
            lp += float(ls[tok]); nT += 1
        lp = lp / max(1, nT)
    return text, {'p1': p1, 'p2': p2, 'margin': margin, 'seq_lp': lp,
                  'p_yes': p_yes, 'p_no': p_no}


def run(bench, model_tag):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f'{bench}__blind__{model_tag}.json')
    if os.path.exists(out_path):
        print(f'[skip exists] {out_path}'); return

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

    blank = make_blank_image()
    out_rows = {}
    n = 0; t0 = time.time()
    for r in iter_bench(bench):
        sid = r['sample_id']
        greedy_ans = sid_to_ans.get(sid)
        if greedy_ans is None: continue
        try:
            system = SYSTEM_TEXT if r['task'] == 'fever' else SYSTEM_VL
            prompt = build_prompt_direct(r)
            # FEVER has no image — use as a sanity check (greedy_ans should == blind_ans)
            blind_image = blank if r['image'] is not None else None
            inp = build_inputs(processor, blind_image, system, prompt)
            text, sig = generate_with_signals(model, tokenizer, inp, max_new=8)
            parsed = parse_answer(text, r['task'], r['label_set'], r.get('choices'))
            out_rows[sid] = {
                'sample_id': sid, 'benchmark': bench, 'task': r['task'],
                'greedy_answer': greedy_ans, 'blind_answer': parsed,
                'blind_disagrees': int(parsed != greedy_ans),
                'blind_margin': sig['margin'], 'blind_p1': sig['p1'],
                'blind_seq_lp': sig['seq_lp'],
                'blind_p_yes': sig['p_yes'], 'blind_p_no': sig['p_no'],
                'blind_yes_minus_no': sig['p_yes'] - sig['p_no'],
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
