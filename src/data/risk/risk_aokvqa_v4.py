"""ARC v4 Step 1: A-OKVQA image-grounded VLM judge (replaces rationale NLI).

v3 failure: rationale-NLI risk had `corr(risk, wrong) = +0.72` because human
rationales encode the answer, so risk(action) ≈ 1 − correct(action).

v4 fix: judge groundedness against the IMAGE, not the rationale text.
A-OKVQA samples have images (COCO 2014). The judge sees:

    image  +  question  +  candidate answer

and answers whether the answer is visually / contextually supported by the
image, with the same proposition-level structure used for HallusionBench:

  Proposition: <one short sentence>
  Verdict: <grounded / contradicted / unverifiable>

Risk:  grounded → 0.0  ;  unverifiable → 0.5  ;  contradicted → 1.0

Output: ./data/risk/A-OKVQA_risk.json
"""
import argparse, json, os, re, sys, time
from PIL import Image
sys.path.insert(0, ".")
from internvl_judge import load_judge, judge_image

ACTION_TABLE = "./data/unified/A-OKVQA_action_table.json"
AOKVQA_SAMP  = "./data/AOKVQA/samples_val.json"
OUT_DIR = "./data/risk"

ACTIONS = ('answer_direct', 'recover_then_answer', 'escalate_to_stronger_model')

SYS_JUDGE = ("You are a strict visual grounding judge. For each candidate answer to a "
             "visual question, FIRST state the single most specific visual proposition the "
             "answer commits to, then classify it against the image. Reply on two lines:\n"
             "Proposition: <one short sentence stating the visual claim>\n"
             "Verdict: <one of: grounded / contradicted / unverifiable>\n"
             "Be strict: only answer 'grounded' if the image literally shows what the "
             "proposition asserts. World-knowledge plausibility alone is NOT enough.")

PROMPT_AOKVQA = ("Question: {q}\n"
                 "Candidate answer: {a}\n\n"
                 "What concrete visual proposition does this answer commit to, and "
                 "does the image support it?")


def parse_judge(text: str):
    t = text.strip()
    verdict = 'unverifiable'; risk = 0.5
    v_match = re.search(r'verdict\s*:?\s*(grounded|contradicted|unverifiable|supported|unsupported)', t, re.I)
    if v_match:
        v = v_match.group(1).lower()
        if v in ('grounded', 'supported'):       verdict = 'grounded';     risk = 0.0
        elif v in ('contradicted', 'unsupported'): verdict = 'contradicted'; risk = 1.0
        else:                                       verdict = 'unverifiable'; risk = 0.5
    else:
        tl = t.lower()
        if 'contradict' in tl or 'unsupported' in tl: verdict = 'contradicted'; risk = 1.0
        elif 'unverifiable' in tl or 'unclear' in tl: verdict = 'unverifiable'; risk = 0.5
        elif 'grounded' in tl or 'supported' in tl:    verdict = 'grounded';     risk = 0.0
    return verdict, risk


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--worker_id', type=int, default=0)
    p.add_argument('--num_workers', type=int, default=1)
    args = p.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.num_workers > 1:
        out_path = os.path.join(OUT_DIR, f'A-OKVQA_risk_chunk{args.worker_id}.json')
    else:
        out_path = os.path.join(OUT_DIR, 'A-OKVQA_risk.json')

    model, tokenizer = load_judge()

    rows_all = json.load(open(ACTION_TABLE))
    rows = [r for i, r in enumerate(rows_all) if i % args.num_workers == args.worker_id]
    print(f'worker {args.worker_id}/{args.num_workers}: {len(rows)}/{len(rows_all)} samples')
    samp_map = {s['sample_id']: s for s in json.load(open(AOKVQA_SAMP))}

    out = {}
    n_label = {'grounded': 0, 'unverifiable': 0, 'contradicted': 0}
    t0 = time.time(); n = 0
    for r in rows:
        sid = r['sample_id']
        meta = samp_map.get(sid)
        if meta is None:
            continue
        try:
            img = Image.open(meta['image_path']).convert('RGB')
        except Exception:
            continue
        per_action = {}
        for a in ACTIONS:
            ans = (r[a].get('answer') or '').strip()
            if not ans:
                per_action[a] = {'risk_score': 0.5, 'judge_label': 'unverifiable', 'judge_raw': '(empty)'}
                n_label['unverifiable'] += 1
                continue
            prompt = PROMPT_AOKVQA.format(q=meta['question'], a=ans)
            raw = judge_image(model, tokenizer, img, SYS_JUDGE, prompt, max_new=48)
            verdict, risk = parse_judge(raw)
            per_action[a] = {'risk_score': risk, 'judge_label': verdict, 'judge_raw': raw[:200]}
            n_label[verdict] += 1
        per_action['abstain'] = {'risk_score': 0.0, 'judge_label': 'no_commitment'}
        out[sid] = per_action
        n += 1
        if n % 100 == 0:
            print(f'  [{n}] {time.time()-t0:.0f}s  labels={n_label}')

    json.dump(out, open(out_path, 'w'), indent=1)
    print(f'\nDone A-OKVQA: {n} samples × 3 actions in {time.time()-t0:.0f}s')
    print(f'  verdict distribution: {n_label}')


if __name__ == '__main__':
    main()
