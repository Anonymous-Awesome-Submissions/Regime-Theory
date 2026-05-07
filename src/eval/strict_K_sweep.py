"""Strict-CV K-sensitivity sweep on Hallu (and others), using v14 Family API.

Re-verifies the v11 K-sensitivity finding (K=4 is the peak) under strict CV.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, './src/controllers')
import controller_arc_p_v11 as v11
import controller_arc_p_v14 as v14

DST = "./data/unified_v14"


def main():
    out = {}
    for bench in ['HallusionBench', 'POPE', 'A-OKVQA', 'FEVER']:
        rows, Xstd, Xscalar, Xtxt, L, C, R, K_arr, cats, vis, sets = v11.load_bench(bench)
        n = len(L)
        fair = float(L[:, 0].mean())
        print(f'\n=== {bench}  n={n}  always_direct={fair:.4f} ===')
        seeds = list(range(42, 42 + 5 * 31, 31))
        bench_out = {'n': n, 'always_direct': fair, 'K_results': {}}
        for K in [2, 3, 4, 5, 6, 7, 8, 10, 12]:
            losses = []
            for s in seeds:
                fac = lambda K=K, s=s: v14.GroupRoutingKMeans(K=K, seed=s)
                _, l = v14.cv_eval_family(fac, Xscalar, L, C, n_splits=5, seed=s)
                losses.append(l)
            m = float(np.mean(losses)); sd = float(np.std(losses))
            d = m - fair
            bench_out['K_results'][K] = {'mean': m, 'std': sd, 'delta': d,
                                          'losses_per_seed': losses}
            print(f'  K={K:2d}  {m:.4f} ± {sd:.4f}  Δ={d:+.4f}')
        out[bench] = bench_out

    json.dump(out, open(f'{DST}/strict_K_sweep.json', 'w'), indent=2)
    print(f'\nSaved {DST}/strict_K_sweep.json')


if __name__ == '__main__':
    main()
