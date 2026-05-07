"""ARC v11 controller (claude, 2026-04-13).

Built on top of v10 with one critical addition motivated by the diagnostic
analysis (claude_note.md, Iteration 3): on Hallu, instance-level correctness
prediction is bounded at AUC=0.671 and cannot beat fair_fixed at any
coverage. But the dataset has STRUCTURAL groups (category × visual_input
axis × question-text cluster) where the per-group best fixed action
*does* differ from the global best, and therefore per-group fixed routing
can beat fair_fixed without needing instance-level prediction.

This is the new "group_routing" family. It is theoretically well-founded
under the regime decomposition: rather than searching for a heterogeneous
instance-level policy (high sample complexity), we search a SMALL space
of group-level routing policies (very low sample complexity).

Families (all CV-fair):
  - fair_fixed                 (gate baseline)
  - oracle_fixed               (hindsight diagnostic)
  - learned_only_v11           (HGBC on FULL features)
  - learned_scalar_only        (HGBC on scalar block only)
  - selective_v11              (calibrated selective)
  - group_routing_cat          (per-category fixed)
  - group_routing_catvis       (per-(cat, vis_input) fixed) — the Hallu winner
  - group_routing_kmeans_K     (per-question-cluster fixed) for K in {4,8}
  - prior_only_v11             (brute-force conjunctions)
  - hybrid_v11                 (priors then group_routing residual)
"""
import argparse, json, os, time
import numpy as np
from collections import Counter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

UNIFIED_V4   = "./data/unified_v4"
UNIFIED_V10  = "./data/unified_v10"
DST          = "./data/unified_v11"
BENCHES = ['POPE', 'HallusionBench', 'A-OKVQA', 'FEVER', 'FOLIO']
ACTIONS = ['answer_direct', 'recover_then_answer', 'escalate_to_stronger_model', 'abstain']
A_IDX = {a: i for i, a in enumerate(ACTIONS)}


def load_bench(bench):
    d = json.load(open(os.path.join(UNIFIED_V4, f'{bench}_loss_matrix.json')))
    rows = d['rows']
    feat_npz = np.load(os.path.join(UNIFIED_V10, f'{bench}_features.npz'), allow_pickle=True)
    Xfull = feat_npz['X']
    sample_ids_feats = list(feat_npz['sample_ids'])
    feat_meta = json.load(open(os.path.join(UNIFIED_V10, f'{bench}_feature_names.json')))

    sid_to_loss_idx = {r['sample_id']: i for i, r in enumerate(rows)}
    keep_feat, keep_loss = [], []
    for fi, sid in enumerate(sample_ids_feats):
        if sid in sid_to_loss_idx:
            keep_feat.append(fi); keep_loss.append(sid_to_loss_idx[sid])
    rows = [rows[i] for i in keep_loss]
    Xfull = Xfull[keep_feat]
    n = len(rows)

    Xstd = np.nan_to_num(Xfull.astype(np.float64))
    mu = Xstd.mean(axis=0, keepdims=True)
    sd = Xstd.std(axis=0, keepdims=True).clip(min=1e-6)
    Xstd = (Xstd - mu) / sd

    n_text = feat_meta.get('n_text_features', Xfull.shape[1])
    n_scalar = feat_meta.get('n_scalar_features', n_text - 1920)
    Xscalar = np.nan_to_num(Xfull[:, :n_scalar].astype(np.float64))
    Xtxt = np.nan_to_num(Xfull[:, n_scalar:n_scalar + 1920].astype(np.float64))

    L = np.zeros((n, 4)); C = np.zeros((n, 4), dtype=int)
    R = np.zeros((n, 4)); K = np.zeros((n, 4))
    cats = []; vis_axis = []; set_axis = []
    for i, r in enumerate(rows):
        for a in ACTIONS:
            ai = A_IDX[a]
            L[i, ai] = r['actions'][a]['loss']
            C[i, ai] = r['actions'][a]['correct']
            R[i, ai] = r['actions'][a]['risk']
            K[i, ai] = r['actions'][a]['cost']
        cats.append(r.get('category', 'NA') or 'NA')
        sid = r['sample_id']
        parts = sid.split('_')
        # heuristic: hallu -> hallu_{set}_{fig}_{q}_{vis}; pope -> pope_{id}; etc.
        set_axis.append(parts[1] if len(parts) > 1 else 'NA')
        vis_axis.append(parts[-1] if len(parts) > 4 else 'NA')
    return rows, Xstd, Xscalar, Xtxt, L, C, R, K, np.array(cats), np.array(vis_axis), np.array(set_axis)


def evaluate(chosen, L, C, R, K):
    n = len(chosen)
    loss = L[np.arange(n), chosen]; crr = C[np.arange(n), chosen]
    abstain = (chosen == A_IDX['abstain']); non_ab = ~abstain
    sel_acc = float(crr[non_ab].mean()) if non_ab.sum() else 0.0
    return {
        'mean_loss': float(loss.mean()),
        'mean_acc':  float(crr.mean()),
        'selective_acc': sel_acc,
        'coverage':  float(non_ab.mean()),
        'action_distribution': {ACTIONS[a]: int((chosen == a).sum()) for a in range(4)},
    }


# ---- baselines

def policy_fair_best_fixed_by_loss(L, n_splits=5, seed=42):
    n = L.shape[0]; chosen = np.zeros(n, int); fold_choices = []
    for tr, te in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(L):
        best_tr = int(np.argmin(L[tr].mean(axis=0)))
        chosen[te] = best_tr
        fold_choices.append(ACTIONS[best_tr])
    return chosen.astype(int), fold_choices


def policy_oracle_best_fixed_by_loss(L):
    best = int(np.argmin(L.mean(axis=0)))
    return np.full(L.shape[0], best, dtype=int), ACTIONS[best]


# ---- HGBC and selective (same as v10) ----

def policy_hgbc(Xstd, L, n_splits=5, seed=42, max_depth=4, max_iter=200, lr=0.05):
    n, d = Xstd.shape
    n_classes = 4
    Xe = np.repeat(Xstd, n_classes, axis=0)
    ye = np.tile(np.arange(n_classes), n)
    max_loss = L.max(axis=1)
    we = np.zeros(n * n_classes, dtype=float)
    for i in range(n):
        for a in range(n_classes):
            we[i * n_classes + a] = float(max_loss[i] - L[i, a])
    we = we + 1e-3
    chosen = np.zeros(n, dtype=int)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(Xstd):
        tr_e = np.concatenate([np.arange(i * n_classes, (i + 1) * n_classes) for i in tr])
        clf = HistGradientBoostingClassifier(max_iter=max_iter, max_depth=max_depth,
                                             learning_rate=lr, random_state=seed)
        clf.fit(Xe[tr_e], ye[tr_e], sample_weight=we[tr_e])
        proba = clf.predict_proba(Xstd[te])
        chosen[te] = clf.classes_[proba.argmax(axis=1)]
    return chosen.astype(int)


def policy_selective_calibrated(Xstd, L, C, n_splits=5, seed=42, C_reg=0.3):
    n = Xstd.shape[0]; chosen = np.zeros(n, int)
    for tr, te in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(Xstd):
        Lhat = np.zeros((len(te), 4))
        for a in range(3):
            y = C[tr, a]
            if y.std() < 1e-6:
                p_corr = np.full(len(te), float(y.mean()))
            else:
                clf = LogisticRegression(max_iter=2000, C=C_reg)
                clf.fit(Xstd[tr], y)
                p_corr = clf.predict_proba(Xstd[te])[:, 1]
            mask_w = (C[tr, a] == 0); mask_r = (C[tr, a] == 1)
            l_w = float(L[tr, a][mask_w].mean()) if mask_w.sum() else 0.0
            l_r = float(L[tr, a][mask_r].mean()) if mask_r.sum() else 0.0
            Lhat[:, a] = (1.0 - p_corr) * l_w + p_corr * l_r
        Lhat[:, 3] = float(L[tr, 3].mean())
        chosen[te] = Lhat.argmin(1)
    return chosen.astype(int)


# ---- group routing (the new family) ----

def policy_group_routing(group_labels, L, n_splits=5, seed=42, min_support=3):
    n = L.shape[0]; chosen = np.zeros(n, int)
    for tr, te in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(L):
        global_best = int(np.argmin(L[tr].mean(axis=0)))
        gtr = group_labels[tr]; gte = group_labels[te]
        for g in np.unique(gte):
            mask_tr = gtr == g; mask_te = gte == g
            if not mask_te.any(): continue
            if mask_tr.sum() < min_support:
                chosen[te[mask_te]] = global_best
            else:
                best = int(np.argmin(L[tr][mask_tr].mean(axis=0)))
                chosen[te[mask_te]] = best
    return chosen.astype(int)


def policy_group_routing_kmeans(Xtext, L, K, n_splits=5, seed=42, min_support=3):
    n = L.shape[0]; chosen = np.zeros(n, int)
    for tr, te in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(L):
        scaler = StandardScaler().fit(Xtext[tr])
        Xtr = scaler.transform(Xtext[tr])
        Xte = scaler.transform(Xtext[te])
        global_best = int(np.argmin(L[tr].mean(axis=0)))
        km = KMeans(n_clusters=K, random_state=seed, n_init=5).fit(Xtr)
        gtr = km.predict(Xtr); gte = km.predict(Xte)
        for g in range(K):
            mask_tr = gtr == g; mask_te = gte == g
            if not mask_te.any(): continue
            if mask_tr.sum() < min_support:
                chosen[te[mask_te]] = global_best
            else:
                best = int(np.argmin(L[tr][mask_tr].mean(axis=0)))
                chosen[te[mask_te]] = best
    return chosen.astype(int)


# ---- brute-force priors (unchanged from v10) ----

def acquire_priors_brute_force(Xscalar_tr, L_tr, n_thresholds=12,
                                 max_rules=8, min_support=4, min_gain=0.01):
    n_tr, d = Xscalar_tr.shape
    fair_action = int(np.argmin(L_tr.mean(axis=0)))
    feat_thresholds = []
    for f in range(d):
        v = Xscalar_tr[:, f]
        if v.std() < 1e-6:
            feat_thresholds.append([])
        else:
            feat_thresholds.append(list(np.unique(np.percentile(v, np.linspace(10, 90, n_thresholds)))))
    candidates = []
    for f1 in range(d):
        if not feat_thresholds[f1]: continue
        v1 = Xscalar_tr[:, f1]
        for t1 in feat_thresholds[f1]:
            for op1 in (0, 1):
                mask1 = (v1 <= t1) if op1 == 0 else (v1 > t1)
                if mask1.sum() < min_support: continue
                for f2 in range(f1 + 1, d):
                    if not feat_thresholds[f2]: continue
                    v2 = Xscalar_tr[:, f2]
                    for t2 in feat_thresholds[f2]:
                        for op2 in (0, 1):
                            mask2 = (v2 <= t2) if op2 == 0 else (v2 > t2)
                            mask = mask1 & mask2
                            sup = int(mask.sum())
                            if sup < min_support: continue
                            sub_L = L_tr[mask]
                            best_a = int(np.argmin(sub_L.mean(axis=0)))
                            if best_a == fair_action: continue
                            gain = float(L_tr[mask, fair_action].mean() - sub_L[:, best_a].mean())
                            if gain <= min_gain: continue
                            candidates.append({
                                'f1': f1, 'op1': op1, 't1': float(t1),
                                'f2': f2, 'op2': op2, 't2': float(t2),
                                'action': best_a, 'support': sup,
                                'gain': gain, 'total_gain': gain * sup,
                            })
    candidates.sort(key=lambda c: -c['total_gain'])
    selected = []
    covered = np.zeros(n_tr, dtype=bool)
    for c in candidates:
        if len(selected) >= max_rules: break
        v1 = Xscalar_tr[:, c['f1']]; v2 = Xscalar_tr[:, c['f2']]
        m1 = (v1 <= c['t1']) if c['op1'] == 0 else (v1 > c['t1'])
        m2 = (v2 <= c['t2']) if c['op2'] == 0 else (v2 > c['t2'])
        mask = m1 & m2 & ~covered
        if mask.sum() < min_support: continue
        sub_L = L_tr[mask]
        best_a = int(np.argmin(sub_L.mean(axis=0)))
        if best_a == fair_action: continue
        gain = float(L_tr[mask, fair_action].mean() - sub_L[:, best_a].mean())
        if gain <= min_gain: continue
        selected.append({**c, 'action': best_a, 'incremental_support': int(mask.sum()),
                         'incremental_gain': gain})
        covered |= mask
    return selected, fair_action


def apply_priors(Xscalar, rules, default_action):
    n = Xscalar.shape[0]; chosen = np.full(n, default_action, dtype=int)
    assigned = np.zeros(n, dtype=bool)
    for r in rules:
        v1 = Xscalar[:, r['f1']]; v2 = Xscalar[:, r['f2']]
        m1 = (v1 <= r['t1']) if r['op1'] == 0 else (v1 > r['t1'])
        m2 = (v2 <= r['t2']) if r['op2'] == 0 else (v2 > r['t2'])
        mask = m1 & m2 & ~assigned
        chosen[mask] = r['action']; assigned[mask] = True
    return chosen, assigned


def policy_hybrid_priors_then_groups(Xscalar, Xtxt, group_labels, L, n_splits=5, seed=42):
    """Hybrid: brute-force priors take precedence; samples not covered by
    any prior fall back to group routing (using the supplied group labels)."""
    n = L.shape[0]; chosen = np.zeros(n, int)
    rules_per_fold = []
    for tr, te in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(L):
        rules, fair = acquire_priors_brute_force(Xscalar[tr], L[tr])
        rules_per_fold.append({'rules': rules, 'fair_action': ACTIONS[fair]})
        # group routing on test fold using train fold to set per-group action
        gtr = group_labels[tr]; gte = group_labels[te]
        group_action_te = np.full(len(te), fair, dtype=int)
        for g in np.unique(gte):
            mask_te = gte == g
            mask_tr = gtr == g
            if mask_tr.sum() < 3: continue
            best = int(np.argmin(L[tr][mask_tr].mean(axis=0)))
            group_action_te[mask_te] = best
        ch_te, assigned = apply_priors(Xscalar[te], rules, fair)
        ch_te[~assigned] = group_action_te[~assigned]
        chosen[te] = ch_te
    return chosen.astype(int), rules_per_fold


def fmt(m, ref):
    d = m.get('action_distribution', {})
    dist = ' '.join(f"{k[:3]}={v}" for k, v in d.items() if v > 0)
    return (f"loss={m['mean_loss']:.4f}  Δ={m['mean_loss']-ref:+.4f}  "
            f"acc={m['mean_acc']*100:5.2f}%  sel={m['selective_acc']*100:5.2f}%  "
            f"cov={m['coverage']*100:5.1f}%  [{dist}]")


def run_bench(bench):
    rows, Xstd, Xscalar, Xtxt, L, C, R, K, cats, vis, sets = load_bench(bench)
    n, d = Xstd.shape
    print(f"\n{'='*128}\n{bench}  n={n}  feat_dim_full={d}  n_scalar={Xscalar.shape[1]}\n{'='*128}")
    print('mean loss per fixed action:  ' + '  '.join(
        f"{a[:10]}={L[:, A_IDX[a]].mean():.4f}" for a in ACTIONS))
    oracle_loss = float(L[np.arange(n), L.argmin(1)].mean())
    print(f"oracle (per-sample loss-optimal): loss={oracle_loss:.4f}\n")

    out = {'bench': bench, 'n': n, 'feat_dim_full': d, 'n_scalar': int(Xscalar.shape[1]),
           'oracle_loss': oracle_loss,
           'mean_loss_per_fixed_action': {ACTIONS[a]: float(L[:, a].mean()) for a in range(4)}}

    cF, fold_names = policy_fair_best_fixed_by_loss(L)
    out['fair_best_fixed_by_loss'] = {'fold_choices': fold_names, **evaluate(cF, L, C, R, K)}
    cO, name_O = policy_oracle_best_fixed_by_loss(L)
    out['oracle_best_fixed_by_loss'] = {'name': name_O, **evaluate(cO, L, C, R, K)}

    for label, fn in [
        ('learned_only_v11_full', lambda: policy_hgbc(Xstd, L)),
        ('learned_scalar_only',   lambda: policy_hgbc(Xscalar, L)),
        ('selective_v11_scalar',  lambda: policy_selective_calibrated(Xscalar, L, C)),
    ]:
        t0 = time.time(); ch = fn()
        out[label] = evaluate(ch, L, C, R, K)
        print(f'  {label:<28} done in {time.time()-t0:.1f}s')

    # Group routing families — these are the new ones
    catvis = np.array([f'{a}_{b}' for a, b in zip(cats, vis)])
    catset = np.array([f'{a}_{b}' for a, b in zip(cats, sets)])
    for label, gl in [
        ('group_route_cat',     cats),
        ('group_route_vis',     vis),
        ('group_route_catvis',  catvis),
        ('group_route_catset',  catset),
    ]:
        t0 = time.time()
        ch = policy_group_routing(gl, L)
        out[label] = evaluate(ch, L, C, R, K)
        print(f'  {label:<28} done in {time.time()-t0:.1f}s  n_unique_groups={len(set(gl))}')

    for K_clust in [4, 8, 12]:
        label = f'group_route_kmeans_K{K_clust}'
        t0 = time.time()
        ch = policy_group_routing_kmeans(Xtxt, L, K_clust)
        out[label] = evaluate(ch, L, C, R, K)
        print(f'  {label:<28} done in {time.time()-t0:.1f}s')

    print('  hybrid_v11 (priors + group_route_catvis residual)')
    t0 = time.time()
    cH, rules_pf = policy_hybrid_priors_then_groups(Xscalar, Xtxt, catvis, L)
    out['hybrid_v11_catvis'] = {**evaluate(cH, L, C, R, K), 'rules_per_fold': rules_pf}
    print(f'    done in {time.time()-t0:.1f}s')

    fair_loss = out['fair_best_fixed_by_loss']['mean_loss']
    print(f"\n{'POLICY':<32} {'loss':>9}  result")
    print('-' * 130)
    keys = ['oracle_best_fixed_by_loss', 'fair_best_fixed_by_loss',
            'learned_only_v11_full', 'learned_scalar_only',
            'selective_v11_scalar',
            'group_route_cat', 'group_route_vis', 'group_route_catvis', 'group_route_catset',
            'group_route_kmeans_K4', 'group_route_kmeans_K8', 'group_route_kmeans_K12',
            'hybrid_v11_catvis']
    for key in keys:
        if key not in out: continue
        m = out[key]
        label = key
        if key == 'oracle_best_fixed_by_loss':
            label = f"oracle_fixed ({m['name']})"
        elif key == 'fair_best_fixed_by_loss':
            label = "fair_fixed (CV) [GATE]"
        print(f"{label:<32} {m['mean_loss']:>9.4f}  {fmt(m, fair_loss)}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bench', choices=BENCHES, default=None)
    args = p.parse_args()
    benches = [args.bench] if args.bench else BENCHES
    os.makedirs(DST, exist_ok=True)
    summary = {}
    for b in benches:
        try:
            r = run_bench(b)
        except Exception as e:
            print(f'[skip] {b}: {type(e).__name__}: {e}'); continue
        summary[b] = r
    out_name = 'controller_v11_results.json' if not args.bench else f'controller_v11_{args.bench}.json'
    json.dump(summary, open(os.path.join(DST, out_name), 'w'), indent=2)
    print(f"\nSaved to {DST}/{out_name}")


if __name__ == '__main__':
    main()
