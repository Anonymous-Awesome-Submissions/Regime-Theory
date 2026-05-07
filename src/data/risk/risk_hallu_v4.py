"""ARC v4 Step 1: HallusionBench proposition-level grounded VLM judge.

v3 failure: the v3 prompt scored at the answer-string level
("is this answer supported by the image?") and returned 95% grounded on
short yes/no answers. ~5% high-risk → loss dominated by accuracy → all
solvers fail Section 5 criterion 1.

v4 fix: two-step proposition-level scoring (single forward, one prompt).

  1. The judge is asked to FIRST state the concrete visual proposition the
     answer commits to (presence / attribute / counting / relation), then
     classify support against the image.
  2. Output is parsed into (proposition_type, support_label).
     support_label ∈ {grounded, contradicted, unverifiable}.
  3. Risk score:
        grounded       → 0.0
        unverifiable   → 0.5
        contradicted   → 1.0

This is closer to what codex calls "proposition-level grounding": even a
"yes" answer commits to a proposition (e.g. "the object X is present"),
and the judge must score that proposition against image evidence.

Output: ./data/risk/HallusionBench_risk.json
        {sample_id: {action: {risk_score, judge_label, prop_type, judge_raw}}}
"""
import json, os, re, sys, time
sys.path.insert(0, ".")
from internvl_judge import load_judge, judge_image

HALLU_DS = "./datasets/HallusionBench"
ACTION_TABLE = "./data/unified/HallusionBench_action_table.json"
HALLU_CANDS  = "./data/GRACE/HallusionBench/candidates.json"
OUT_DIR = "./data/risk"

ACTIONS = ('answer_direct', 'recover_then_answer', 'escalate_to_stronger_model')

SYS_JUDGE = ("You are a strict visual proposition judge. For each answer you must "
             "FIRST state the single most specific visual proposition the answer commits to "
             "(presence / attribute / count / relation), and THEN classify it against the image. "
             "Reply on two lines:\n"
             "Proposition: <one short sentence stating the visual claim>\n"
             "Verdict: <one of: grounded / contradicted / unverifiable>\n"
             "Be strict: if the image does not literally show what the proposition asserts, "
             "answer 'contradicted' or 'unverifiable'.")

PROMPT_HALLU = ("Question: {q}\n"
                "Candidate answer: {a}\n\n"
                "What concrete visual proposition does this answer commit to, "
                "and does the image support it?")


def parse_judge(text: str):
    t = text.strip()
    prop_type = 'unknown'
    verdict = 'unverifiable'
    risk = 0.5
    # Grab proposition line
    prop_match = re.search(r'proposition\s*:?\s*(.+?)(?:\n|verdict|$)', t, re.I | re.S)
    prop_text = prop_match.group(1).strip() if prop_match else ''
    pt_low = prop_text.lower()
    if 'how many' in pt_low or any(w in pt_low for w in [' count ', ' number ', ' two ', ' three ', ' four ', ' five ']):
        prop_type = 'count'
    elif any(w in pt_low for w in [' on ', ' next to ', ' near ', ' between ', ' behind ', ' in front ', ' above ', ' below ', ' beside ']):
        prop_type = 'relation'
    elif any(w in pt_low for w in [' color ', ' colour ', ' size ', ' shape ', ' is wearing ', ' has ', ' is ', ' looks ']):
        prop_type = 'attribute'
    else:
        prop_type = 'presence'
    # Verdict line
    v_match = re.search(r'verdict\s*:?\s*(grounded|contradicted|unverifiable|supported|unsupported)', t, re.I)
    if v_match:
        v = v_match.group(1).lower()
        if v in ('grounded', 'supported'):       verdict = 'grounded';     risk = 0.0
        elif v in ('contradicted', 'unsupported'): verdict = 'contradicted'; risk = 1.0
        else:                                       verdict = 'unverifiable'; risk = 0.5
    else:
        # fallback: look for keywords anywhere
        tl = t.lower()
        if 'contradict' in tl or 'unsupported' in tl: verdict = 'contradicted'; risk = 1.0
        elif 'unverifiable' in tl or 'unclear' in tl: verdict = 'unverifiable'; risk = 0.5
        elif 'grounded' in tl or 'supported' in tl:    verdict = 'grounded';     risk = 0.0
    return verdict, prop_type, risk


def main():
    from datasets import load_from_disk
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'HallusionBench_risk.json')

    model, tokenizer = load_judge()

    rows = json.load(open(ACTION_TABLE))
    ds = load_from_disk(HALLU_DS)
    sid_to_img = {}
    for i in range(len(ds)):
        r = ds[i]
        note = (r.get('sample_note') or '').strip()
        note = ''.join(ch if ch.isalnum() else '_' for ch in note)[:30]
        sid = f"hallu_{r['set_id']}_{r['figure_id']}_{r['question_id']}_{r['visual_input']}_{note}"
        sid_to_img[sid] = r['image']
    cand_q = {c['sample_id']: c['question'] for c in json.load(open(HALLU_CANDS))}

    out = {}
    n_label = {'grounded': 0, 'unverifiable': 0, 'contradicted': 0}
    n_prop  = {'presence': 0, 'attribute': 0, 'count': 0, 'relation': 0, 'unknown': 0}
    t0 = time.time(); n = 0
    for r in rows:
        sid = r['sample_id']
        img = sid_to_img.get(sid)
        question = cand_q.get(sid, '')
        if img is None or not question:
            continue
        per_action = {}
        for a in ACTIONS:
            ans = (r[a].get('answer') or '').strip()
            if not ans:
                per_action[a] = {'risk_score': 0.5, 'judge_label': 'unverifiable',
                                 'prop_type': 'unknown', 'judge_raw': '(empty)'}
                n_label['unverifiable'] += 1; n_prop['unknown'] += 1
                continue
            prompt = PROMPT_HALLU.format(q=question, a=ans)
            raw = judge_image(model, tokenizer, img, SYS_JUDGE, prompt, max_new=64)
            verdict, prop_type, risk = parse_judge(raw)
            per_action[a] = {'risk_score': risk, 'judge_label': verdict,
                             'prop_type': prop_type, 'judge_raw': raw[:200]}
            n_label[verdict] += 1; n_prop[prop_type] += 1
        per_action['abstain'] = {'risk_score': 0.0, 'judge_label': 'no_commitment'}
        out[sid] = per_action
        n += 1
        if n % 50 == 0:
            print(f'  [{n}] {time.time()-t0:.0f}s  labels={n_label}  prop={n_prop}')

    json.dump(out, open(out_path, 'w'), indent=1)
    print(f'\nDone HallusionBench: {n} samples × 3 actions in {time.time()-t0:.0f}s')
    print(f'  verdict distribution: {n_label}')
    print(f'  proposition type distribution: {n_prop}')


if __name__ == '__main__':
    main()
