"""ARC v14 — strict nested CV controller (claude, 2026-04-13, after codex audit).

Fixes the v12/v13 protocol bug: the chosen family in those scripts was
re-fit on the FULL dataset and only the outer-test slice was read out.
That meant each family's internal KFold could place outer-test samples
into inner-train folds, which is data leakage at the family level (the
hyperparameters and cluster centers were chosen with outer-test info).

v14 enforces strict nested CV:

  Outer 5-fold CV over the data:
    For each outer fold (outer_train, outer_test):
      1. Inner 5-fold CV inside outer_train ONLY:
         For each candidate family:
           - run the family in CV-fair mode on outer_train only
           - score = mean inner-CV loss on outer_train
         Pick the family with lowest inner CV loss.
      2. Re-fit the picked family ONCE on outer_train only.
      3. Predict on outer_test (no further CV).
      4. Append to chosen_outer[outer_test].
    Aggregate over outer folds.
    Repeat for n_seeds, report mean ± std of the resulting mean loss.

This is also organized by **policy class** Pi_0..Pi_2, per codex's
Section 3 / Step 2 in communication_AI.md:

  Pi_0 (fixed):     always-direct / always-escalate / fair-fixed-CV
  Pi_1 (group):     group routing on (cat, cat×vis, KMeans on scalar/text)
  Pi_2 (instance):  learned HGBC, calibrated selective LR

The family pool is split into these 3 classes; we both report each class's
best CV-fair number AND run an outer auto-pick across the three classes.

NOTE on family `fit` semantics:
  - Pi_0.fair_fixed_cv: pick best mean-loss action on outer_train
  - Pi_1.group_routing_kmeans: fit KMeans on outer_train, per cluster pick
    train-fold best action; on outer_test, predict cluster and assign action
  - Pi_2.learned_hgbc: fit HGBC cost-sensitive on outer_train, predict on
    outer_test
  - Pi_2.selective_calibrated: fit per-action LR on outer_train, plug-in
    expected loss on outer_test, argmin
"""
import argparse, json, os, time
import numpy as np
import sys
sys.path.insert(0, './src/controllers')
import controller_arc_p_v11 as v11
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

ACTIONS = v11.ACTIONS
A_IDX = v11.A_IDX
DST = "./data/unified_v14"


# ============================================================================
# Family fit/predict APIs (each family is a (fit, predict) pair)
# ============================================================================

class Family:
    """A family: fit(X_tr, L_tr, C_tr), predict(X_te) -> chosen action ints."""
    name = 'base'
    policy_class = 'Pi_0'
    def fit(self, X_tr, L_tr, C_tr): raise NotImplementedError
    def predict(self, X_te): raise NotImplementedError


class AlwaysDirect(Family):
    name = 'always_direct'
    policy_class = 'Pi_0'
    def fit(self, X_tr, L_tr, C_tr): self._a = 0
    def predict(self, X_te): return np.full(len(X_te), self._a, dtype=int)


class FairFixedCV(Family):
    name = 'fair_fixed_train'
    policy_class = 'Pi_0'
    def fit(self, X_tr, L_tr, C_tr):
        self._a = int(np.argmin(L_tr.mean(axis=0)))
    def predict(self, X_te): return np.full(len(X_te), self._a, dtype=int)


class GroupRoutingKMeans(Family):
    policy_class = 'Pi_1'
    def __init__(self, K, seed=42, min_support=3):
        self.K = K; self.seed = seed; self.min_support = min_support
        self.name = f'KM_K{K:02d}'
    def fit(self, X_tr, L_tr, C_tr):
        self.scaler = StandardScaler().fit(X_tr)
        Xn = self.scaler.transform(X_tr)
        self.km = KMeans(n_clusters=self.K, random_state=self.seed, n_init=10).fit(Xn)
        gtr = self.km.predict(Xn)
        self.global_best = int(np.argmin(L_tr.mean(axis=0)))
        self.cluster_action = {}
        for g in range(self.K):
            mask = gtr == g
            if mask.sum() < self.min_support:
                self.cluster_action[g] = self.global_best
            else:
                self.cluster_action[g] = int(np.argmin(L_tr[mask].mean(axis=0)))
    def predict(self, X_te):
        Xn = self.scaler.transform(X_te)
        gte = self.km.predict(Xn)
        return np.array([self.cluster_action[g] for g in gte], dtype=int)


class LearnedHGBC(Family):
    policy_class = 'Pi_2'
    def __init__(self, max_iter=200, max_depth=4, lr=0.05, seed=42):
        self.max_iter = max_iter; self.max_depth = max_depth
        self.lr = lr; self.seed = seed
        self.name = f'HGBC_md{max_depth}'
    def fit(self, X_tr, L_tr, C_tr):
        n, d = X_tr.shape
        n_classes = 4
        Xe = np.repeat(X_tr, n_classes, axis=0)
        ye = np.tile(np.arange(n_classes), n)
        max_loss = L_tr.max(axis=1)
        we = np.zeros(n * n_classes, dtype=float)
        for i in range(n):
            for a in range(n_classes):
                we[i * n_classes + a] = float(max_loss[i] - L_tr[i, a])
        we = we + 1e-3
        self.scaler = StandardScaler().fit(X_tr)
        Xn = self.scaler.transform(X_tr)
        Xe_s = np.repeat(Xn, n_classes, axis=0)
        self.clf = HistGradientBoostingClassifier(
            max_iter=self.max_iter, max_depth=self.max_depth,
            learning_rate=self.lr, random_state=self.seed)
        self.clf.fit(Xe_s, ye, sample_weight=we)
    def predict(self, X_te):
        Xn = self.scaler.transform(X_te)
        proba = self.clf.predict_proba(Xn)
        return self.clf.classes_[proba.argmax(axis=1)].astype(int)


class SelectiveCalibrated(Family):
    policy_class = 'Pi_2'
    def __init__(self, C_reg=0.3, seed=42):
        self.C_reg = C_reg; self.seed = seed
        self.name = f'sel_C{C_reg}'
    def fit(self, X_tr, L_tr, C_tr):
        self.scaler = StandardScaler().fit(X_tr)
        Xn = self.scaler.transform(X_tr)
        self.clfs = {}; self.l_w = {}; self.l_r = {}
        for a in range(3):
            y = C_tr[:, a]
            if y.std() < 1e-6:
                self.clfs[a] = float(y.mean())
            else:
                clf = LogisticRegression(max_iter=2000, C=self.C_reg)
                clf.fit(Xn, y)
                self.clfs[a] = clf
            mask_w = (C_tr[:, a] == 0); mask_r = (C_tr[:, a] == 1)
            self.l_w[a] = float(L_tr[mask_w, a].mean()) if mask_w.sum() else 0.0
            self.l_r[a] = float(L_tr[mask_r, a].mean()) if mask_r.sum() else 0.0
        self.l_abstain = float(L_tr[:, 3].mean())
    def predict(self, X_te):
        Xn = self.scaler.transform(X_te)
        n_te = len(Xn)
        Lhat = np.zeros((n_te, 4))
        for a in range(3):
            obj = self.clfs[a]
            if isinstance(obj, float):
                p = np.full(n_te, obj)
            else:
                p = obj.predict_proba(Xn)[:, 1]
            Lhat[:, a] = (1 - p) * self.l_w[a] + p * self.l_r[a]
        Lhat[:, 3] = self.l_abstain
        return Lhat.argmin(1).astype(int)


# ============================================================================
# Pi_3 — prior-augmented residual families (brute-force conjunction mining)
# ============================================================================

def _mine_conjunctions(X_tr, L_tr, n_thresholds=12, max_rules=8,
                        min_support=4, min_gain=0.01):
    """Source-B prior acquisition: brute-force search over all
    (f1, op1, t1) AND (f2, op2, t2) conjunctions on the scalar block.
    Returns a list of selected rules (greedy by total gain on
    incremental support) plus the train-fold default action."""
    n_tr, d = X_tr.shape
    fair_action = int(np.argmin(L_tr.mean(axis=0)))
    feat_thresholds = []
    for f in range(d):
        v = X_tr[:, f]
        if v.std() < 1e-6:
            feat_thresholds.append([])
        else:
            feat_thresholds.append(list(np.unique(np.percentile(
                v, np.linspace(10, 90, n_thresholds)))))
    candidates = []
    for f1 in range(d):
        if not feat_thresholds[f1]: continue
        v1 = X_tr[:, f1]
        for t1 in feat_thresholds[f1]:
            for op1 in (0, 1):
                mask1 = (v1 <= t1) if op1 == 0 else (v1 > t1)
                if mask1.sum() < min_support: continue
                for f2 in range(f1 + 1, d):
                    if not feat_thresholds[f2]: continue
                    v2 = X_tr[:, f2]
                    for t2 in feat_thresholds[f2]:
                        for op2 in (0, 1):
                            mask2 = (v2 <= t2) if op2 == 0 else (v2 > t2)
                            mask = mask1 & mask2
                            sup = int(mask.sum())
                            if sup < min_support: continue
                            sub_L = L_tr[mask]
                            best_a = int(np.argmin(sub_L.mean(axis=0)))
                            if best_a == fair_action: continue
                            gain = float(L_tr[mask, fair_action].mean()
                                          - sub_L[:, best_a].mean())
                            if gain <= min_gain: continue
                            candidates.append({
                                'f1': f1, 'op1': op1, 't1': float(t1),
                                'f2': f2, 'op2': op2, 't2': float(t2),
                                'action': best_a, 'gain': gain,
                                'total_gain': gain * sup,
                            })
    candidates.sort(key=lambda c: -c['total_gain'])
    selected = []
    covered = np.zeros(n_tr, dtype=bool)
    for c in candidates:
        if len(selected) >= max_rules: break
        v1 = X_tr[:, c['f1']]; v2 = X_tr[:, c['f2']]
        m1 = (v1 <= c['t1']) if c['op1'] == 0 else (v1 > c['t1'])
        m2 = (v2 <= c['t2']) if c['op2'] == 0 else (v2 > c['t2'])
        mask = m1 & m2 & ~covered
        if mask.sum() < min_support: continue
        sub_L = L_tr[mask]
        best_a = int(np.argmin(sub_L.mean(axis=0)))
        if best_a == fair_action: continue
        gain = float(L_tr[mask, fair_action].mean() - sub_L[:, best_a].mean())
        if gain <= min_gain: continue
        selected.append({**c, 'action': best_a})
        covered |= mask
    return selected, fair_action


def _apply_rules(X_te, rules, default_action):
    n_te = X_te.shape[0]
    chosen = np.full(n_te, default_action, dtype=int)
    assigned = np.zeros(n_te, dtype=bool)
    for r in rules:
        v1 = X_te[:, r['f1']]; v2 = X_te[:, r['f2']]
        m1 = (v1 <= r['t1']) if r['op1'] == 0 else (v1 > r['t1'])
        m2 = (v2 <= r['t2']) if r['op2'] == 0 else (v2 > r['t2'])
        mask = m1 & m2 & ~assigned
        chosen[mask] = r['action']
        assigned[mask] = True
    return chosen, assigned


class PriorOnlyConjunction(Family):
    """Pi_3 prior-only: brute-force conjunction rules + fair_fixed default."""
    name = 'prior_only_conj'
    policy_class = 'Pi_3'
    def fit(self, X_tr, L_tr, C_tr):
        # Note: rules are mined on the RAW scalar block, no scaling needed
        # because thresholds are computed from per-feature percentiles.
        self.rules, self.default_action = _mine_conjunctions(X_tr, L_tr)
    def predict(self, X_te):
        chosen, _ = _apply_rules(X_te, self.rules, self.default_action)
        return chosen


class HybridPriorLearned(Family):
    """Pi_3 hybrid: prior rules cover what they can; learned residual
    (selective LR on scalar) handles uncovered samples.
    Train-time: mine rules on X_tr, train LR on the FULL X_tr (residual
    indices not used here — LR sees everything but is only consulted on
    samples that the prior does not cover at test time)."""
    name = 'hybrid_prior_learned'
    policy_class = 'Pi_3'
    def __init__(self, C_reg=0.3, seed=42):
        self.C_reg = C_reg; self.seed = seed
    def fit(self, X_tr, L_tr, C_tr):
        self.rules, self.default_action = _mine_conjunctions(X_tr, L_tr)
        # Selective LR residual on outer-train ONLY
        self.scaler = StandardScaler().fit(X_tr)
        Xn = self.scaler.transform(X_tr)
        self.clfs = {}; self.l_w = {}; self.l_r = {}
        for a in range(3):
            y = C_tr[:, a]
            if y.std() < 1e-6:
                self.clfs[a] = float(y.mean())
            else:
                clf = LogisticRegression(max_iter=2000, C=self.C_reg)
                clf.fit(Xn, y)
                self.clfs[a] = clf
            mask_w = (C_tr[:, a] == 0); mask_r = (C_tr[:, a] == 1)
            self.l_w[a] = float(L_tr[mask_w, a].mean()) if mask_w.sum() else 0.0
            self.l_r[a] = float(L_tr[mask_r, a].mean()) if mask_r.sum() else 0.0
        self.l_abstain = float(L_tr[:, 3].mean())
    def predict(self, X_te):
        # Apply prior rules first
        chosen, assigned = _apply_rules(X_te, self.rules, self.default_action)
        # For uncovered samples, use selective LR
        unassigned_idx = np.where(~assigned)[0]
        if len(unassigned_idx) == 0:
            return chosen
        Xn = self.scaler.transform(X_te[unassigned_idx])
        n_un = len(unassigned_idx)
        Lhat = np.zeros((n_un, 4))
        for a in range(3):
            obj = self.clfs[a]
            if isinstance(obj, float):
                p = np.full(n_un, obj)
            else:
                p = obj.predict_proba(Xn)[:, 1]
            Lhat[:, a] = (1 - p) * self.l_w[a] + p * self.l_r[a]
        Lhat[:, 3] = self.l_abstain
        chosen[unassigned_idx] = Lhat.argmin(1).astype(int)
        return chosen


# ============================================================================
# Strict nested CV
# ============================================================================

def candidate_families():
    return [
        AlwaysDirect(),
        FairFixedCV(),
        GroupRoutingKMeans(K=4),
        GroupRoutingKMeans(K=5),
        GroupRoutingKMeans(K=6),
        GroupRoutingKMeans(K=8),
        LearnedHGBC(max_depth=3),
        LearnedHGBC(max_depth=4),
        SelectiveCalibrated(C_reg=0.3),
    ]


def cv_eval_family(family_factory, X, L, C, n_splits=5, seed=42):
    """Strict CV: for each fold, fit on train, predict on test. No leakage."""
    n = len(L); chosen = np.zeros(n, dtype=int)
    for tr, te in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(L):
        fam = family_factory()
        fam.fit(X[tr], L[tr], C[tr])
        chosen[te] = fam.predict(X[te])
    return chosen, float(L[np.arange(n), chosen].mean())


def strict_nested_auto_pick(X, L, C, family_factories, n_outer=5, n_inner=5, seed=42):
    """Strict nested CV with auto-pick.
    For each outer fold:
      1. Inner CV inside outer_train: each family is CV-evaluated only on
         outer_train slices.
      2. Pick family with lowest inner mean loss.
      3. Re-fit picked family on full outer_train.
      4. Predict on outer_test.
    Returns chosen_per_sample, mean_loss, picked_per_fold."""
    n = len(L); chosen_outer = np.zeros(n, dtype=int)
    picked_per_fold = []
    inner_scores_per_fold = []
    for tr, te in KFold(n_outer, shuffle=True, random_state=seed).split(L):
        # Inner CV over outer_train ONLY
        inner_scores = {}
        n_tr = len(tr)
        for fac in family_factories:
            inner_chosen = np.zeros(n_tr, dtype=int)
            for itr, ite in KFold(n_inner, shuffle=True, random_state=seed + 1).split(tr):
                # itr/ite index into tr, so the absolute indices are tr[itr] / tr[ite]
                fam = fac()
                fam.fit(X[tr[itr]], L[tr[itr]], C[tr[itr]])
                inner_chosen[ite] = fam.predict(X[tr[ite]])
            inner_scores[fac().name] = float(L[tr][np.arange(n_tr), inner_chosen].mean())
        best_name = min(inner_scores, key=lambda k: inner_scores[k])
        picked_per_fold.append(best_name)
        inner_scores_per_fold.append(inner_scores)
        # Refit picked on FULL outer_train, predict on outer_test
        best_fac = next(f for f in family_factories if f().name == best_name)
        fam = best_fac()
        fam.fit(X[tr], L[tr], C[tr])
        chosen_outer[te] = fam.predict(X[te])
    mean_loss = float(L[np.arange(n), chosen_outer].mean())
    return chosen_outer, mean_loss, picked_per_fold, inner_scores_per_fold


def run_bench(bench, n_seeds=5):
    rows, Xstd, Xscalar, Xtxt, L, C, R, K, cats, vis, sets = v11.load_bench(bench)
    n = len(L)
    fair_const = float(L[:, 0].mean())
    print(f"\n{'='*100}\n{bench}  n={n}  always_direct={fair_const:.4f}\n{'='*100}")

    families = [
        ('always_direct',       AlwaysDirect, 'Pi_0'),
        ('fair_fixed_train',    FairFixedCV,  'Pi_0'),
        ('KM_K04',              lambda: GroupRoutingKMeans(K=4), 'Pi_1'),
        ('KM_K05',              lambda: GroupRoutingKMeans(K=5), 'Pi_1'),
        ('KM_K06',              lambda: GroupRoutingKMeans(K=6), 'Pi_1'),
        ('KM_K08',              lambda: GroupRoutingKMeans(K=8), 'Pi_1'),
        ('HGBC_md3',            lambda: LearnedHGBC(max_depth=3), 'Pi_2'),
        ('HGBC_md4',            lambda: LearnedHGBC(max_depth=4), 'Pi_2'),
        ('selective_C0.3',      lambda: SelectiveCalibrated(C_reg=0.3), 'Pi_2'),
        ('prior_only_conj',     PriorOnlyConjunction, 'Pi_3'),
        ('hybrid_prior_learned', lambda: HybridPriorLearned(C_reg=0.3), 'Pi_3'),
    ]

    out = {'bench': bench, 'n': n, 'always_direct': fair_const,
           'per_family': {}, 'per_class_best': {}, 'auto_pick': {}}

    seeds = list(range(42, 42 + n_seeds * 31, 31))

    # Per-family multi-seed CV (strict)
    print('\n--- Per-family strict CV (5 seeds) ---')
    print(f'{"family":<22} {"class":<6} {"mean ± std":<22} {"Δ":>10}')
    print('-' * 72)
    for fam_name, fac, cls in families:
        losses = []
        for s in seeds:
            _, loss = cv_eval_family(fac, Xscalar, L, C, n_splits=5, seed=s)
            losses.append(loss)
        m = float(np.mean(losses)); sd = float(np.std(losses))
        out['per_family'][fam_name] = {
            'class': cls, 'mean': m, 'std': sd,
            'delta': m - fair_const, 'losses_per_seed': losses,
        }
        print(f'{fam_name:<22} {cls:<6} {m:.4f} ± {sd:.4f}   Δ={m-fair_const:+.4f}')

    # Per policy-class best (must enumerate ALL classes present in the family pool,
    # including Pi_3 — codex audit Section 8 Issue 1, fixed 2026-04-13)
    classes_in_pool = sorted({v['class'] for v in out['per_family'].values()})
    print(f'\n--- Best per policy class (classes_in_pool = {classes_in_pool}) ---')
    for cls in classes_in_pool:
        in_class = {k: v for k, v in out['per_family'].items() if v['class'] == cls}
        if not in_class: continue
        best_name = min(in_class, key=lambda k: in_class[k]['mean'])
        out['per_class_best'][cls] = {'family': best_name, **in_class[best_name]}
        r = in_class[best_name]
        print(f'{cls}: best = {best_name}  loss={r["mean"]:.4f} ± {r["std"]:.4f}  Δ={r["delta"]:+.4f}')

    # Auto-pick (strict nested CV) over all families
    print('\n--- Strict nested-CV auto-pick (5 seeds) ---')
    factories = [fac for _, fac, _ in families]
    auto_losses = []
    auto_picks = []
    for s in seeds:
        _, loss, picked, inner = strict_nested_auto_pick(Xscalar, L, C, factories, seed=s)
        auto_losses.append(loss)
        auto_picks.append(picked)
        print(f'  seed={s}  ARC-P_strict={loss:.4f}  Δ={loss-fair_const:+.4f}  picked={picked}')
    m = float(np.mean(auto_losses)); sd = float(np.std(auto_losses))
    out['auto_pick'] = {
        'mean': m, 'std': sd, 'delta': m - fair_const,
        'losses_per_seed': auto_losses, 'picked_per_fold_per_seed': auto_picks,
    }
    print(f'\nARC-P strict nested auto-pick: {m:.4f} ± {sd:.4f}  Δ={m-fair_const:+.4f}')

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bench', required=True)
    p.add_argument('--seeds', type=int, default=5)
    args = p.parse_args()
    os.makedirs(DST, exist_ok=True)
    r = run_bench(args.bench, n_seeds=args.seeds)
    json.dump(r, open(f'{DST}/controller_v14_{args.bench}.json', 'w'), indent=2)
    print(f'\nSaved {DST}/controller_v14_{args.bench}.json')


if __name__ == '__main__':
    main()
