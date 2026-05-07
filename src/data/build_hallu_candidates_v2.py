"""Rebuild HallusionBench candidates.json with the corrected 951-unique-sample IDs.

Old key (set,fig,qid,vis) collapsed to 383; new key includes sample_note for
the full 951 unique samples. Schema mirrors the existing format minimally:
  sample_id, question_id, category, subcategory, question, gt_answer.
We don't regenerate K=4 sampling signals (token_conf, self_consistency etc.)
because nothing downstream needs them at this size.
"""
import json, os
from datasets import load_from_disk

HF = "./datasets/HallusionBench"
OUT = "./data/GRACE/HallusionBench/candidates.json"
BAK = OUT.replace('.json', '_old_383.json')

if os.path.exists(OUT) and not os.path.exists(BAK):
    os.rename(OUT, BAK)
    print(f'[backup] {OUT} -> {BAK}')

ds = load_from_disk(HF)
print(f'[load] HF n={len(ds)}')

def _sid(r):
    note = (r.get('sample_note') or '').strip()
    note = ''.join(ch if ch.isalnum() else '_' for ch in note)[:30]
    return f"hallu_{r['set_id']}_{r['figure_id']}_{r['question_id']}_{r['visual_input']}_{note}"

GT_MAP = {'0': 'no', '1': 'yes', 'no': 'no', 'yes': 'yes'}
cands = []
seen = set()
for i in range(len(ds)):
    r = ds[i]
    sid = _sid(r)
    if sid in seen:
        continue
    seen.add(sid)
    gt_raw = (r['gt_answer'] or '').strip().lower()
    cands.append({
        'sample_id': sid,
        'question_id': r.get('question_id'),
        'category': r.get('category'),
        'subcategory': r.get('subcategory'),
        'visual_input': r.get('visual_input'),
        'sample_note': r.get('sample_note'),
        'set_id': r.get('set_id'),
        'figure_id': r.get('figure_id'),
        'question': r['question'],
        'gt_answer': GT_MAP.get(gt_raw, gt_raw),
    })

with open(OUT, 'w') as f:
    json.dump(cands, f, indent=1)
print(f'[wrote] {OUT}: n={len(cands)}')
