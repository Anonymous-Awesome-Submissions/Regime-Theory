"""Compute (p_g, gamma_g) per cluster for HallusionBench KMeans-K, matching
the paper's Theorem partition diagnostic in §6.4."""
import json, os, sys
import numpy as np
sys.path.insert(0, './src/controllers')
import controller_arc_p_v11 as v11
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
SEED = 42

rows, Xstd, Xscalar, Xtxt, L, C, R, K_arr, cats, vis, sets = v11.load_bench('HallusionBench')
n = L.shape[0]
A = L.shape[1]
print(f'n={n}  fair_fixed={L[:,0].mean():.4f}')

global_best_a = int(np.argmin(L.mean(axis=0)))
print(f'global best action = {global_best_a}')

Xs = StandardScaler().fit_transform(np.nan_to_num(Xscalar))
labels = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(Xs)

print(f'\n=== K={K} cluster anatomy ===')
print(f'{"g":<3} {"p_g":<8} {"size":<6} {"cell_best":<14} {"gamma_g":<10}')
total_pg_gammag = 0.0
for g in range(K):
    mask = (labels == g)
    p_g = mask.mean()
    Lg = L[mask].mean(axis=0)
    cell_best = int(np.argmin(Lg))
    if cell_best == global_best_a:
        gamma_g = 0.0
    else:
        gamma_g = float(L[mask, global_best_a].mean() - L[mask, cell_best].mean())
    total_pg_gammag += p_g * gamma_g
    print(f'{g:<3} {p_g:.3f}    {mask.sum():<6} {cell_best:<14} {gamma_g:.4f}')

print(f'\nmax_g p_g * gamma_g = {max((labels==g).mean() * (max(0.0, float(L[labels==g, 0].mean() - L[labels==g, int(np.argmin(L[labels==g].mean(axis=0)))].mean())) if int(np.argmin(L[labels==g].mean(axis=0))) != global_best_a else 0.0) for g in range(K)):.4f}')
print(f'sum_g p_g * gamma_g  = {total_pg_gammag:.4f}')
