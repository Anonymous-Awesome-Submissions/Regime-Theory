"""ARC Step 2d + Step 3 head: unified action table + fixed-action baselines.

Inputs: ./data/action_tables/
        {bench}__{direct,recover}__3B.json
        {bench}__direct__7B.json   (this is the escalate action)

Output: ./data/unified/
        {bench}_action_table.json   one row per sample, all 4 actions joined
        summary.json                per-bench fixed-action baselines + hard-subset analysis

Per sample schema:
    sample_id, benchmark, category,
    answer_direct:    {answer, correct, confidence, cost, ...}
    recover_then_answer: {...}
    escalate_to_stronger_model: {...}
    abstain:          {correct: 0, cost: 0}           # first-class no-op
    features:         {direct_margin, direct_seq_lp, recover_margin, ...}

Hard-subset analysis per benchmark:
  - full_acc of each fixed action
  - best fixed action
  - hard_subset = {samples where answer_direct wrong}
  - on hard_subset: acc(recover), acc(escalate), acc(union), overlap(recover,escalate)
  - oracle any-action correct (upper bound for offline controller)
"""
import json, os, sys
from collections import defaultdict
import numpy as np

SRC = "./data/action_tables"
DST = "./data/unified"
BENCHES = ['POPE', 'HallusionBench', 'A-OKVQA', 'FEVER']


def load(bench, action, model):
    p = os.path.join(SRC, f'{bench}__{action}__{model}.json')
    return {r['sample_id']: r for r in json.load(open(p))['results']}


def join_benchmark(bench):
    direct3 = load(bench, 'direct', '3B')   # answer_direct
    recov3  = load(bench, 'recover', '3B')  # recover_then_answer
    direct7 = load(bench, 'direct', '7B')   # escalate_to_stronger_model
    # Use direct3's sample_id set as anchor; require presence in all three
    ids = sorted(set(direct3) & set(recov3) & set(direct7))
    rows = []
    for sid in ids:
        d3 = direct3[sid]; r3 = recov3[sid]; d7 = direct7[sid]
        rows.append({
            'sample_id':  sid,
            'benchmark':  bench,
            'category':   d3.get('category'),
            'gt':         d3.get('gt'),
            'answer_direct': {
                'answer':  d3['answer'], 'correct': int(d3['correct']),
                'confidence_margin': d3['confidence_margin'],
                'seq_logprob':       d3['seq_logprob'],
                'cost_passes': 1, 'cost_tokens': d3.get('cost_tokens', 0),
            },
            'recover_then_answer': {
                'answer':  r3['answer'], 'correct': int(r3['correct']),
                'confidence_margin': r3['confidence_margin'],
                'seq_logprob':       r3['seq_logprob'],
                'cost_passes': 2, 'cost_tokens': r3.get('cost_tokens', 0),
                'ctx_margin':        r3.get('ctx_margin', 0.0),
                'recover_context':   r3.get('recover_context', '')[:200],
            },
            'escalate_to_stronger_model': {
                'answer':  d7['answer'], 'correct': int(d7['correct']),
                'confidence_margin': d7['confidence_margin'],
                'seq_logprob':       d7['seq_logprob'],
                'cost_passes': 1, 'cost_tokens': d7.get('cost_tokens', 0),
            },
            'abstain': {
                'answer': None, 'correct': 0, 'confidence_margin': 0.0,
                'seq_logprob': 0.0, 'cost_passes': 0, 'cost_tokens': 0,
            },
        })
    return rows


def summarize(bench, rows):
    n = len(rows)
    cd = np.array([r['answer_direct']['correct'] for r in rows])
    cr = np.array([r['recover_then_answer']['correct'] for r in rows])
    ce = np.array([r['escalate_to_stronger_model']['correct'] for r in rows])
    ca = np.zeros(n, dtype=int)  # abstain

    full_acc = {
        'answer_direct':               float(cd.mean()),
        'recover_then_answer':         float(cr.mean()),
        'escalate_to_stronger_model':  float(ce.mean()),
        'abstain':                     0.0,
    }
    best_fixed = max(full_acc, key=full_acc.get)
    best_fixed_acc = full_acc[best_fixed]

    # Hard subset vs direct
    hard = (cd == 0)
    n_hard = int(hard.sum())
    hard_r_acc = float(cr[hard].mean()) if n_hard else float('nan')
    hard_e_acc = float(ce[hard].mean()) if n_hard else float('nan')
    # Overlap: recover & escalate correct on same hard sample
    hard_both = ((cr == 1) & (ce == 1) & hard).sum()
    hard_r_only = ((cr == 1) & (ce == 0) & hard).sum()
    hard_e_only = ((cr == 0) & (ce == 1) & hard).sum()
    hard_neither = ((cr == 0) & (ce == 0) & hard).sum()
    # Oracle any-correct (answer_direct / recover / escalate)
    any_correct = ((cd == 1) | (cr == 1) | (ce == 1)).astype(int)
    oracle_acc = float(any_correct.mean())

    # Cost accounting (mean across samples) — relative to direct=1 in cost_passes units
    cost = {
        'answer_direct': 1.0,
        'recover_then_answer': float(np.mean([r['recover_then_answer']['cost_passes'] for r in rows])),
        'escalate_to_stronger_model': float(np.mean([
            r['escalate_to_stronger_model']['cost_passes'] * 2.3 for r in rows])),
        # 2.3x factor = empirical wall-clock cost ratio 7B/3B on A100 fp16
        'abstain': 0.0,
    }

    # Action differentiation diagnostics
    # recover_extra = samples where recover correct but direct wrong
    recover_extra = int(((cr == 1) & (cd == 0)).sum())
    escalate_extra = int(((ce == 1) & (cd == 0)).sum())
    # Jaccard overlap on hard subset (of the complementary recovered sets)
    r_set = set(np.where((cr == 1) & hard)[0].tolist())
    e_set = set(np.where((ce == 1) & hard)[0].tolist())
    union = r_set | e_set
    inter = r_set & e_set
    jaccard = len(inter) / len(union) if union else float('nan')

    return {
        'benchmark': bench, 'n': n,
        'full_acc': full_acc,
        'best_fixed_action': best_fixed,
        'best_fixed_acc': best_fixed_acc,
        'cost_passes_per_sample': cost,
        'hard_subset': {
            'n_hard': n_hard,
            'recover_acc':  hard_r_acc,
            'escalate_acc': hard_e_acc,
            'recover_recovered': int((cr == 1)[hard].sum()),
            'escalate_recovered': int((ce == 1)[hard].sum()),
            'both_recovered':     int(hard_both),
            'recover_only':       int(hard_r_only),
            'escalate_only':      int(hard_e_only),
            'neither_recovered':  int(hard_neither),
            'jaccard_overlap':    jaccard,
        },
        'oracle_any_correct': oracle_acc,
        'oracle_headroom_over_best_fixed': oracle_acc - best_fixed_acc,
    }


def main():
    os.makedirs(DST, exist_ok=True)
    summary = {}
    for bench in BENCHES:
        rows = join_benchmark(bench)
        out = os.path.join(DST, f'{bench}_action_table.json')
        json.dump(rows, open(out, 'w'), indent=2)
        s = summarize(bench, rows)
        summary[bench] = s

    json.dump(summary, open(os.path.join(DST, 'summary.json'), 'w'), indent=2)

    # Pretty print
    print(f"\n{'='*96}")
    print(f"{'Bench':<17} {'n':<5} {'direct':<9} {'recover':<9} {'escalate':<9} {'best':<25} {'oracle':<9} {'head':<7}")
    print('-' * 96)
    for b, s in summary.items():
        fa = s['full_acc']
        print(f"{b:<17} {s['n']:<5} {fa['answer_direct']*100:>6.2f}%  {fa['recover_then_answer']*100:>6.2f}%  "
              f"{fa['escalate_to_stronger_model']*100:>6.2f}%  {s['best_fixed_action']:<25} "
              f"{s['oracle_any_correct']*100:>6.2f}%  {s['oracle_headroom_over_best_fixed']*100:>+5.2f}pp")
    print('-' * 96)

    print(f"\nHard subset (where answer_direct is wrong):")
    print(f"{'Bench':<17} {'n_hard':<7} {'recov acc':<10} {'esc acc':<10} "
          f"{'both':<6} {'r_only':<7} {'e_only':<7} {'neither':<8} {'jaccard':<8}")
    for b, s in summary.items():
        hs = s['hard_subset']
        ja = hs['jaccard_overlap']
        print(f"{b:<17} {hs['n_hard']:<7} {hs['recover_acc']*100:>7.2f}%  {hs['escalate_acc']*100:>7.2f}%  "
              f"{hs['both_recovered']:<6} {hs['recover_only']:<7} {hs['escalate_only']:<7} "
              f"{hs['neither_recovered']:<8} {ja:<8.3f}")
    print(f"\nSaved to {DST}")


if __name__ == '__main__':
    main()
