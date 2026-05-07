"""Non-KMeans Π_1 realization (shallow CART partition router).

Reviewer concern: "Π_1 is instantiated primarily with KMeans; decision trees,
rule lists, or other structured partitions are not explored. If the paper
wants to claim the partition-escape is a property of the benchmark rather than
an artifact of KMeans, it should show a second, structurally different
Π_1 realization lands in the same regime locations."

This script adds a shallow-tree partition router:
  - Fit sklearn DecisionTreeClassifier on StandardScaler(X) using as label
    the per-sample row argmin a*_i = argmin_a L[i,a] -- the tree only sees
    which action is best per sample, not the magnitudes -- so the tree's
    decision surface is a true data-adaptive partition of X, and the leaf
    action assignment is taken as the cell-wise loss-argmin on the training
    samples that fall into that leaf (min_support=3).

The runner evaluates this new Π_1 family on all four benchmarks under the
same strict 5-fold-by-5-seed CV protocol and saves results to the unified
v14 artifact directory. A companion table (Table 6 in main.tex) reports the
cross-check against KMeans.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, './src/controllers')

import controller_arc_p_v11 as v11
from controller_arc_p_v14 import (
    Family, AlwaysDirect, cv_eval_family,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

DST = './data/unified_v14'


class GroupRoutingTree(Family):
    """Shallow CART partition router (Π_1, non-KMeans realization).

    The tree is grown on the per-sample best-action label a*_i = argmin_a
    L[i,a] rather than on any correctness signal; this makes the tree a
    discrete partition whose leaves correspond to (data-adaptive) cells
    of X. At fit time we override each leaf's prediction with the loss
    argmin over the training samples that fall into that leaf, matching
    the GroupRoutingKMeans cell-assignment recipe.
    """
    policy_class = 'Pi_1'

    def __init__(self, max_depth=3, seed=42, min_support=3):
        self.max_depth = max_depth
        self.seed = seed
        self.min_support = min_support
        self.name = f'tree_md{max_depth}'

    def fit(self, X_tr, L_tr, C_tr):
        self.scaler = StandardScaler().fit(X_tr)
        Xn = self.scaler.transform(X_tr)
        y = np.argmin(L_tr, axis=1)
        self.tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=max(self.min_support, 5),
            random_state=self.seed,
        ).fit(Xn, y)
        leaves_tr = self.tree.apply(Xn)
        self.global_best = int(np.argmin(L_tr.mean(axis=0)))
        self.leaf_action = {}
        for lf in np.unique(leaves_tr):
            mask = leaves_tr == lf
            if mask.sum() < self.min_support:
                self.leaf_action[int(lf)] = self.global_best
            else:
                self.leaf_action[int(lf)] = int(np.argmin(L_tr[mask].mean(axis=0)))

    def predict(self, X_te):
        Xn = self.scaler.transform(X_te)
        leaves = self.tree.apply(Xn)
        return np.array(
            [self.leaf_action.get(int(lf), self.global_best) for lf in leaves],
            dtype=int,
        )


def run_bench(bench, n_seeds=5):
    rows, Xstd, Xscalar, Xtxt, L, C, R, K, cats, vis, sets = v11.load_bench(bench)
    n = len(L)
    fair_const = float(L[:, 0].mean())
    print(f"\n{'='*80}\n{bench}  n={n}  always_direct={fair_const:.4f}\n{'='*80}")

    families = [
        ('tree_md3', lambda: GroupRoutingTree(max_depth=3), 'Pi_1_tree'),
        ('tree_md4', lambda: GroupRoutingTree(max_depth=4), 'Pi_1_tree'),
    ]

    out = {
        'bench': bench, 'n': n, 'always_direct': fair_const,
        'tree_per_family': {},
    }
    seeds = list(range(42, 42 + n_seeds * 31, 31))

    print(f'{"family":<14} {"class":<12} {"mean ± std":<22} {"Δ":>10}')
    print('-' * 64)
    for fam_name, fac, cls in families:
        losses = []
        for s in seeds:
            _, loss = cv_eval_family(fac, Xscalar, L, C, n_splits=5, seed=s)
            losses.append(float(loss))
        m = float(np.mean(losses)); sd = float(np.std(losses))
        out['tree_per_family'][fam_name] = {
            'class': cls, 'mean': m, 'std': sd,
            'delta': m - fair_const,
            'losses_per_seed': losses,
        }
        print(f'{fam_name:<14} {cls:<12} {m:.4f} ± {sd:.4f}   Δ={m-fair_const:+.4f}')

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bench', required=True,
                   choices=['HallusionBench', 'POPE', 'A-OKVQA', 'FEVER', 'FOLIO'])
    p.add_argument('--seeds', type=int, default=5)
    args = p.parse_args()
    os.makedirs(DST, exist_ok=True)
    r = run_bench(args.bench, n_seeds=args.seeds)
    out_path = f'{DST}/tree_router_{args.bench}.json'
    json.dump(r, open(out_path, 'w'), indent=2)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
