"""Controlled synthetic validation of Π_3 (prior + residual).

Motivation
==========
The four real benchmarks used in the main paper (HallusionBench, POPE,
A-OKVQA, FEVER) all have their winning class at Pi_1 or Pi_2, and Pi_3
is empirically dominated on every one of them (Table 1). This is honest
but leaves the lattice's top rung empirically unvalidated: a skeptical
reviewer can ask whether Pi_3 has *any* regime in which it is the right
class, or whether it is a decorative extension.

This script is the Π_3 analog of `synth_regime_phase_v4.py` (which
validates Corollary 1 item 3 in a controlled setting): we construct a
data-generating process with an explicit, *orthogonal* prior information
source z that is not in the feature block X, and sweep its strength. The
prediction is that Pi_3 (prior-gated + Pi_2 residual) strictly wins when
the prior carries enough mutual information that its gated decisions are
more reliable than anything Pi_2 can extract from X alone.

DGP
---
* 4 latent clusters on R^6, p = (0.35, 0.25, 0.20, 0.20)
* Feature block X = cluster means + iid gaussian noise  (the thing Π_0/Π_1/Π_2 see)
* Smooth latent score s = X · w, with w fixed unit-norm
* Hidden prior scalar z ~ N(0, 1), drawn *independently* of X. This is
  the key twist: z is NOT a function of X, so Pi_2 cannot learn it from
  X alone no matter how large n is. z is only visible to the Pi_3 family
  as a separate "prior oracle" channel.
* Correctness logit:
      logit(c_direct = 1 | X, z) = bk * s(X) + z_strength * z + cluster_bump[g]
  bk is fixed at 1.6 (moderate Pi_2 signal) and cluster_bump is small
  (so Pi_1 has modest partition value), the sweep knob is z_strength.
* Loss matrix: L[direct]=0 if correct else 1, L[recover]=0.55,
  L[escalate]=0.32, L[abstain]=0.70  (same as v4).

Families
--------
Pi_0/Pi_1/Pi_2 see only X. Pi_3 is given X *and* the raw z value as a
prior oracle channel, and is defined by:

    Pi_3(x, z) =
        a_direct     if  z > +tau
        a_escalate   if  z < -tau
        Pi_2_residual(x)   otherwise

where Pi_2_residual is a regularized multinomial LR fit on the subset of
training samples for which |z_train| <= tau (the residual set). tau is
fixed at 1.0 (the prior fires on ~32% of the population under z ~ N(0,1),
matching typical high-confidence prior gating).

Expected result
---------------
At z_strength = 0 the prior channel is uncorrelated with correctness and
Pi_3 should reduce to (a slightly worse version of) Pi_2; at moderate
z_strength the prior gate is accurate and Pi_3 should start to beat Pi_2;
at large z_strength z dominates the logit and Pi_3 should win decisively
across all n. This is the *scope regime* for Pi_3: when the prior has
non-trivial mutual information with correctness on top of X, the top
rung of the lattice is strictly optimal.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, './src/controllers')
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


OUT_FIG = './figures'
OUT_ART = './data/unified_v14'

N_ACTIONS = 4
P_CLUSTERS = np.array([0.35, 0.25, 0.20, 0.20])
CLUSTER_BUMP = np.array([0.3, 0.3, 0.3, -0.3])  # weak partition so Pi_1 isn't the story
L_RECOVER = 0.55
L_ESCALATE = 0.32
L_ABSTAIN = 0.70
BK_FIXED = 1.6  # Pi_2 in high-signal regime
TAU = 1.0  # prior confidence threshold


def gen_data(n, z_strength, seed):
    rng = np.random.default_rng(seed)
    g = rng.choice(4, size=n, p=P_CLUSTERS)
    d = 6
    means = np.zeros((4, d))
    for k in range(4):
        means[k, k] = 2.0
    X = means[g] + rng.normal(size=(n, d))
    w = np.array([0.30, 0.30, 0.30, 0.30, 0.60, 0.70])
    w = w / np.linalg.norm(w)
    s = X @ w + rng.normal(scale=0.3, size=n)
    z = rng.normal(size=n)  # orthogonal prior channel
    logit = BK_FIXED * s + z_strength * z + CLUSTER_BUMP[g]
    p_correct = 1.0 / (1.0 + np.exp(-logit))
    c_d = (rng.uniform(size=n) < p_correct).astype(int)
    L = np.zeros((n, N_ACTIONS))
    L[:, 0] = np.where(c_d == 1, 0.0, 1.0)
    L[:, 1] = L_RECOVER
    L[:, 2] = L_ESCALATE
    L[:, 3] = L_ABSTAIN
    return X, z, L, c_d


# ========== families ==========
def fit_pi0(X_tr, L_tr):
    return int(np.argmin(L_tr.mean(axis=0)))


def fit_pi1(X_tr, L_tr, K=4, seed=0):
    sc = StandardScaler().fit(X_tr)
    km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(sc.transform(X_tr))
    gtr = km.predict(sc.transform(X_tr))
    global_best = int(np.argmin(L_tr.mean(axis=0)))
    act = {}
    for k in range(K):
        m = gtr == k
        act[k] = int(np.argmin(L_tr[m].mean(0))) if m.sum() >= 3 else global_best
    return sc, km, act


def pred_pi1(sc, km, act, X_te):
    gte = km.predict(sc.transform(X_te))
    return np.array([act[g] for g in gte], dtype=int)


def fit_pi2(X_tr, L_tr, seed=0):
    n = len(L_tr)
    sc = StandardScaler().fit(X_tr)
    Xn = sc.transform(X_tr)
    Xe = np.repeat(Xn, N_ACTIONS, axis=0)
    ye = np.tile(np.arange(N_ACTIONS), n)
    Lmax = L_tr.max(axis=1)
    we = np.zeros(n * N_ACTIONS)
    for i in range(n):
        for a in range(N_ACTIONS):
            we[i * N_ACTIONS + a] = float(Lmax[i] - L_tr[i, a])
    we += 1e-3
    clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', random_state=seed)
    clf.fit(Xe, ye, sample_weight=we)
    return sc, clf


def pred_pi2(sc, clf, X_te):
    return clf.classes_[clf.predict_proba(sc.transform(X_te)).argmax(axis=1)].astype(int)


def fit_pi3(X_tr, z_tr, L_tr, seed=0):
    """Pi_3: prior-gated rule fallback-ing to a Pi_2 residual trained on the
    non-gated subset of the training data."""
    gate = np.abs(z_tr) > TAU
    residual = ~gate
    if residual.sum() < 10:
        # tiny residual pool -> fall back to fitting Pi_2 on all training data
        sc_r, clf_r = fit_pi2(X_tr, L_tr, seed=seed)
    else:
        sc_r, clf_r = fit_pi2(X_tr[residual], L_tr[residual], seed=seed)
    return sc_r, clf_r


def pred_pi3(sc_r, clf_r, X_te, z_te):
    out = np.empty(len(z_te), dtype=int)
    pos = z_te > TAU
    neg = z_te < -TAU
    out[pos] = 0  # a_direct
    out[neg] = 2  # a_escalate
    rest = ~(pos | neg)
    if rest.any():
        out[rest] = pred_pi2(sc_r, clf_r, X_te[rest])
    return out


# ========== cv evaluators ==========
def cv_eval_pi0(X, z, L, seed=0):
    n = len(L); chosen = np.zeros(n, dtype=int)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(L):
        a = fit_pi0(X[tr], L[tr])
        chosen[te] = a
    return float(L[np.arange(n), chosen].mean())


def cv_eval_pi1(X, z, L, seed=0, K=4):
    n = len(L); chosen = np.zeros(n, dtype=int)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(L):
        sc, km, act = fit_pi1(X[tr], L[tr], K=K, seed=seed)
        chosen[te] = pred_pi1(sc, km, act, X[te])
    return float(L[np.arange(n), chosen].mean())


def cv_eval_pi2(X, z, L, seed=0):
    n = len(L); chosen = np.zeros(n, dtype=int)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(L):
        sc, clf = fit_pi2(X[tr], L[tr], seed=seed)
        chosen[te] = pred_pi2(sc, clf, X[te])
    return float(L[np.arange(n), chosen].mean())


def cv_eval_pi3(X, z, L, seed=0):
    n = len(L); chosen = np.zeros(n, dtype=int)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(L):
        sc_r, clf_r = fit_pi3(X[tr], z[tr], L[tr], seed=seed)
        chosen[te] = pred_pi3(sc_r, clf_r, X[te], z[te])
    return float(L[np.arange(n), chosen].mean())


def run_grid(n_vals, z_vals, n_seeds=3):
    results = {}
    for n in n_vals:
        for zs in z_vals:
            l0s, l1s, l2s, l3s = [], [], [], []
            for s in range(n_seeds):
                seed = 42 + 31 * s
                X, z, L, c_d = gen_data(n, zs, seed)
                l0s.append(cv_eval_pi0(X, z, L, seed=seed))
                l1s.append(cv_eval_pi1(X, z, L, seed=seed, K=4))
                l2s.append(cv_eval_pi2(X, z, L, seed=seed))
                l3s.append(cv_eval_pi3(X, z, L, seed=seed))
            r = {
                'pi0': float(np.mean(l0s)), 'pi0_std': float(np.std(l0s)),
                'pi1': float(np.mean(l1s)), 'pi1_std': float(np.std(l1s)),
                'pi2': float(np.mean(l2s)), 'pi2_std': float(np.std(l2s)),
                'pi3': float(np.mean(l3s)), 'pi3_std': float(np.std(l3s)),
            }
            vals = {'pi0': r['pi0'], 'pi1': r['pi1'],
                    'pi2': r['pi2'], 'pi3': r['pi3']}
            r['winner'] = min(vals, key=lambda k: vals[k])
            results[(n, zs)] = r
            print(f'n={n:5d}  z={zs:4.1f}  '
                  f'pi0={r["pi0"]:.4f}  pi1={r["pi1"]:.4f}  '
                  f'pi2={r["pi2"]:.4f}  pi3={r["pi3"]:.4f}  -> {r["winner"]}')
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=3)
    args = p.parse_args()

    n_vals = [300, 600, 1200, 2400, 4800]
    z_vals = [0.0, 0.5, 1.0, 1.5, 2.5, 4.0]

    print(f'\nSynthetic Pi_3 validation: {len(n_vals)} x {len(z_vals)} '
          f'cells, {args.seeds} seeds, bk={BK_FIXED}, tau={TAU}\n')

    results = run_grid(n_vals, z_vals, n_seeds=args.seeds)

    os.makedirs(OUT_ART, exist_ok=True)
    json.dump({f'{k[0]}_{k[1]}': v for k, v in results.items()},
              open(f'{OUT_ART}/synth_pi3_validation.json', 'w'), indent=2)

    from collections import Counter
    print('\nWinner distribution:',
          dict(Counter(r['winner'] for r in results.values())))
    print(f"\nSaved {OUT_ART}/synth_pi3_validation.json")


if __name__ == '__main__':
    main()
