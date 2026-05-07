"""L2D baseline runner for the ARC NeurIPS 2026 submission.

Adds two Family-API-compatible baselines from the learning-to-defer literature
and runs them under the same strict nested CV protocol as v14:

  (A) MozannarL2D  — consistent softmax surrogate of Mozannar & Sontag (ICML 2020),
      adapted to the defer/triage action set. The original surrogate is
          ℓ_surr(h, x, y, m) = −log(h_y(x)) − 1[y=m]·log(h_{K+1}(x))
      In our setting there is no separate classifier-vs-expert split: each
      action carries its own per-sample loss, and the consistent surrogate
      collapses to a cost-sensitive cross-entropy over |A|=4 actions with
      sample weights  w_{i,a} = L_max_i − L[i,a].  We instantiate it as a
      multinomial logistic regression over the row-replicated training set,
      which is the canonical linear-family version of Mozannar's surrogate.

  (B) NarasimhanPlugIn — post-hoc plug-in estimator from Narasimhan et al.
      (NeurIPS 2022). The recipe is: train |A| independent cost regressors
      ĥ_a(x) ≈ E[L(x,a) | x], then pick π(x) = argmin_a ĥ_a(x).  We use
      HistGradientBoostingRegressor for each action.

Both families live in Π_2 (instance-level learned controllers) and are
evaluated against:
  (i) v14's Π_2 best (HGBC / SelectiveCalibrated)
  (ii) v14's per-class winners (so we can answer "does an L2D baseline
       beat the regime-predicted Π_1 winner on HallusionBench?")

Results are written to unified_v14/l2d_baselines_{bench}.json.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, './src/controllers')
import controller_arc_p_v11 as v11
import controller_arc_p_v14 as v14
from controller_arc_p_v14 import (
    Family, AlwaysDirect, FairFixedCV, GroupRoutingKMeans,
    LearnedHGBC, SelectiveCalibrated, cv_eval_family, strict_nested_auto_pick,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

ACTIONS = v14.ACTIONS
A_IDX = v14.A_IDX
DST = "./data/unified_v14"
N_ACTIONS = 4


# ============================================================================
# New L2D baseline families (both Pi_2: instance-level learned)
# ============================================================================

class MozannarL2D(Family):
    """Mozannar & Sontag 2020 consistent softmax surrogate (adapted).

    Training: expand each row into |A|=4 virtual rows, one per action,
    with sample-weight proportional to (L_max_i − L[i,a]). The action
    with largest weight is the "correct" class, and multinomial LR
    minimizes the row-weighted cross-entropy, which recovers the
    Mozannar consistent surrogate for cost-sensitive deferral.
    """
    policy_class = 'Pi_2'

    def __init__(self, C_reg=1.0, max_iter=2000, seed=42):
        self.C_reg = C_reg
        self.max_iter = max_iter
        self.seed = seed
        self.name = f'mozannar_C{C_reg}'

    def fit(self, X_tr, L_tr, C_tr):
        n, d = X_tr.shape
        self.scaler = StandardScaler().fit(X_tr)
        Xn = self.scaler.transform(X_tr)

        Xe = np.repeat(Xn, N_ACTIONS, axis=0)
        ye = np.tile(np.arange(N_ACTIONS), n)

        max_loss = L_tr.max(axis=1)
        we = np.zeros(n * N_ACTIONS, dtype=float)
        for i in range(n):
            for a in range(N_ACTIONS):
                we[i * N_ACTIONS + a] = float(max_loss[i] - L_tr[i, a])
        we = we + 1e-3

        self.clf = LogisticRegression(
            C=self.C_reg, max_iter=self.max_iter,
            solver='lbfgs',
            random_state=self.seed,
        )
        self.clf.fit(Xe, ye, sample_weight=we)

    def predict(self, X_te):
        Xn = self.scaler.transform(X_te)
        proba = self.clf.predict_proba(Xn)
        return self.clf.classes_[proba.argmax(axis=1)].astype(int)


class NarasimhanPlugIn(Family):
    """Narasimhan, Menon, Jitkrittum, Kumar 2022 post-hoc plug-in.

    For each action a, train a regressor ĥ_a(x) ≈ E[L(x,a) | x]; at test
    time pick π(x) = argmin_a ĥ_a(x). Uses HGBR for per-action regression,
    which is the modern version of the plug-in recipe.
    """
    policy_class = 'Pi_2'

    def __init__(self, max_iter=200, max_depth=4, lr=0.05, seed=42):
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.lr = lr
        self.seed = seed
        self.name = f'narasimhan_md{max_depth}'

    def fit(self, X_tr, L_tr, C_tr):
        self.scaler = StandardScaler().fit(X_tr)
        Xn = self.scaler.transform(X_tr)
        self.regs = []
        for a in range(N_ACTIONS):
            reg = HistGradientBoostingRegressor(
                max_iter=self.max_iter, max_depth=self.max_depth,
                learning_rate=self.lr, random_state=self.seed,
            )
            reg.fit(Xn, L_tr[:, a])
            self.regs.append(reg)

    def predict(self, X_te):
        Xn = self.scaler.transform(X_te)
        n_te = Xn.shape[0]
        Lhat = np.zeros((n_te, N_ACTIONS))
        for a in range(N_ACTIONS):
            Lhat[:, a] = self.regs[a].predict(Xn)
        return Lhat.argmin(axis=1).astype(int)


# ============================================================================
# Benchmark runner
# ============================================================================

def run_bench(bench, n_seeds=5):
    rows, Xstd, Xscalar, Xtxt, L, C, R, K, cats, vis, sets = v11.load_bench(bench)
    n = len(L)
    fair_const = float(L[:, 0].mean())
    print(f"\n{'='*100}\n{bench}  n={n}  always_direct={fair_const:.4f}\n{'='*100}")

    # New L2D families only
    l2d_families = [
        ('mozannar_C1.0',    lambda: MozannarL2D(C_reg=1.0),       'Pi_2_L2D'),
        ('mozannar_C0.3',    lambda: MozannarL2D(C_reg=0.3),       'Pi_2_L2D'),
        ('narasimhan_md3',   lambda: NarasimhanPlugIn(max_depth=3),'Pi_2_L2D'),
        ('narasimhan_md4',   lambda: NarasimhanPlugIn(max_depth=4),'Pi_2_L2D'),
    ]

    # Comparison pool: v14 canonical families (same as v14.run_bench)
    arc_families = [
        ('always_direct',       AlwaysDirect,                            'Pi_0'),
        ('fair_fixed_train',    FairFixedCV,                             'Pi_0'),
        ('KM_K04',              lambda: GroupRoutingKMeans(K=4),         'Pi_1'),
        ('KM_K05',              lambda: GroupRoutingKMeans(K=5),         'Pi_1'),
        ('KM_K06',              lambda: GroupRoutingKMeans(K=6),         'Pi_1'),
        ('KM_K08',              lambda: GroupRoutingKMeans(K=8),         'Pi_1'),
        ('HGBC_md3',            lambda: LearnedHGBC(max_depth=3),        'Pi_2'),
        ('HGBC_md4',            lambda: LearnedHGBC(max_depth=4),        'Pi_2'),
        ('selective_C0.3',      lambda: SelectiveCalibrated(C_reg=0.3),  'Pi_2'),
    ]

    out = {
        'bench': bench, 'n': n, 'always_direct': fair_const,
        'l2d_per_family': {}, 'arc_per_family': {},
        'l2d_vs_arc_summary': {},
    }

    seeds = list(range(42, 42 + n_seeds * 31, 31))

    # --- New L2D baselines ---
    print('\n--- L2D baselines: strict 5-fold CV, 5 seeds ---')
    print(f'{"family":<22} {"class":<12} {"mean ± std":<22} {"Δ":>10}')
    print('-' * 74)
    for fam_name, fac, cls in l2d_families:
        losses = []
        for s in seeds:
            _, loss = cv_eval_family(fac, Xscalar, L, C, n_splits=5, seed=s)
            losses.append(loss)
        m = float(np.mean(losses)); sd = float(np.std(losses))
        out['l2d_per_family'][fam_name] = {
            'class': cls, 'mean': m, 'std': sd,
            'delta': m - fair_const, 'losses_per_seed': losses,
        }
        print(f'{fam_name:<22} {cls:<12} {m:.4f} ± {sd:.4f}   Δ={m-fair_const:+.4f}')

    # --- ARC canonical families (cross-check; same numbers as v14) ---
    print('\n--- ARC canonical families (cross-check to v14) ---')
    print(f'{"family":<22} {"class":<6} {"mean ± std":<22} {"Δ":>10}')
    print('-' * 72)
    for fam_name, fac, cls in arc_families:
        losses = []
        for s in seeds:
            _, loss = cv_eval_family(fac, Xscalar, L, C, n_splits=5, seed=s)
            losses.append(loss)
        m = float(np.mean(losses)); sd = float(np.std(losses))
        out['arc_per_family'][fam_name] = {
            'class': cls, 'mean': m, 'std': sd,
            'delta': m - fair_const, 'losses_per_seed': losses,
        }
        print(f'{fam_name:<22} {cls:<6} {m:.4f} ± {sd:.4f}   Δ={m-fair_const:+.4f}')

    # --- Summary: head-to-head between L2D best and ARC per-class best ---
    l2d_best_name = min(out['l2d_per_family'],
                        key=lambda k: out['l2d_per_family'][k]['mean'])
    l2d_best = out['l2d_per_family'][l2d_best_name]

    classes_in_arc = sorted({v['class'] for v in out['arc_per_family'].values()})
    arc_per_class_best = {}
    for cls in classes_in_arc:
        in_cls = {k: v for k, v in out['arc_per_family'].items() if v['class'] == cls}
        bn = min(in_cls, key=lambda k: in_cls[k]['mean'])
        arc_per_class_best[cls] = {'family': bn, **in_cls[bn]}

    print('\n--- Head-to-head summary ---')
    print(f'L2D best:      {l2d_best_name}  loss={l2d_best["mean"]:.4f} ± {l2d_best["std"]:.4f}  Δ={l2d_best["delta"]:+.4f}')
    for cls in classes_in_arc:
        r = arc_per_class_best[cls]
        print(f'ARC {cls:<4} best: {r["family"]:<22}  loss={r["mean"]:.4f} ± {r["std"]:.4f}  Δ={r["delta"]:+.4f}')

    # verdict: does the L2D best beat the ARC overall winner on this bench?
    arc_overall = min(arc_per_class_best.values(), key=lambda v: v['mean'])
    l2d_beats = l2d_best['mean'] < arc_overall['mean'] - max(l2d_best['std'], arc_overall['std'])
    l2d_ties = abs(l2d_best['mean'] - arc_overall['mean']) <= max(l2d_best['std'], arc_overall['std'])
    verdict = ('L2D beats ARC overall' if l2d_beats
               else ('tie' if l2d_ties else 'ARC overall beats L2D'))
    print(f'\nOverall verdict: {verdict}')
    print(f'  L2D best loss = {l2d_best["mean"]:.4f}')
    print(f'  ARC overall best = {arc_overall["family"]} ({arc_overall["class"]}) = {arc_overall["mean"]:.4f}')

    out['l2d_vs_arc_summary'] = {
        'l2d_best_family': l2d_best_name,
        'l2d_best_loss': l2d_best['mean'],
        'l2d_best_std': l2d_best['std'],
        'arc_per_class_best': arc_per_class_best,
        'arc_overall_best_family': arc_overall['family'],
        'arc_overall_best_class': arc_overall['class'],
        'arc_overall_best_loss': arc_overall['mean'],
        'verdict': verdict,
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bench', required=True)
    p.add_argument('--seeds', type=int, default=5)
    args = p.parse_args()
    os.makedirs(DST, exist_ok=True)
    r = run_bench(args.bench, n_seeds=args.seeds)
    out_path = f'{DST}/l2d_baselines_{args.bench}.json'
    json.dump(r, open(out_path, 'w'), indent=2)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
