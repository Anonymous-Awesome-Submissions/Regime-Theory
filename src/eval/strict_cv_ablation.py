"""Strict-CV re-verification of the Hallu probe ablation.

Question: in v11 (leaky) we found that dropping the consistency / blind /
sv3 / sv7 probe blocks from the 41-d scalar feature block degraded KM_K04
on Hallu from -0.024 to -0.006 (multi-seed mean), suggesting probes were
load-bearing for cluster geometry even though their per-feature AUC for
"direct correct" was at chance.

Re-verify under v14 strict CV (no leakage at the family-fit step).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, './src/controllers')
import controller_arc_p_v11 as v11
import controller_arc_p_v14 as v14

DST = "./data/unified_v14"


def main():
    rows, Xstd, Xscalar, Xtxt, L, C, R, K, cats, vis, sets = v11.load_bench('HallusionBench')
    n = len(L)
    fair = float(L[:, 0].mean())
    print(f'n={n}  always_direct={fair:.4f}\n')

    # Index layout: 0-19 = v4_1, 20-27 = consistency, 28-32 = blind,
    #               33-36 = sv3, 37-40 = sv7
    feat_groups = {
        'all (41-d)':           np.arange(41),
        'no_consistency':       np.array([i for i in range(41) if not (20 <= i < 28)]),
        'no_blind':             np.array([i for i in range(41) if not (28 <= i < 33)]),
        'no_sv3':               np.array([i for i in range(41) if not (33 <= i < 37)]),
        'no_sv7':               np.array([i for i in range(41) if not (37 <= i < 41)]),
        'only_v4_1 (20-d)':     np.arange(20),
        'only_consistency':     np.arange(20, 28),
        'only_blind':           np.arange(28, 33),
        'only_v4_1+consist':    np.arange(28),
        'no_v4_1 (probes only)': np.arange(20, 41),
    }

    seeds = list(range(42, 42 + 5 * 31, 31))
    print('Strict-CV KM_K04 ablation on Hallu (5 seeds)')
    print(f'{"feature subset":<26} d   mean ± std         Δ')
    print('-' * 70)
    out = {}
    for name, idx in feat_groups.items():
        if len(idx) < 2: continue
        Xsub = Xscalar[:, idx]
        losses = []
        for s in seeds:
            fac = lambda: v14.GroupRoutingKMeans(K=4, seed=s)
            _, l = v14.cv_eval_family(fac, Xsub, L, C, n_splits=5, seed=s)
            losses.append(l)
        m = float(np.mean(losses)); sd = float(np.std(losses))
        out[name] = {'d': len(idx), 'mean': m, 'std': sd, 'delta': m - fair}
        print(f'{name:<26} {len(idx):3d}  {m:.4f} ± {sd:.4f}  Δ={m-fair:+.4f}')

    json.dump(out, open(f'{DST}/ablation_strict_hallu_KM_K04.json', 'w'), indent=2)
    print(f'\nSaved {DST}/ablation_strict_hallu_KM_K04.json')


if __name__ == '__main__':
    main()
