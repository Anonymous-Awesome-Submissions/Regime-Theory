"""Synthetic phase diagram v3 for the ARC NeurIPS submission.

v2 failed because the cluster heterogeneity was too strong: cluster 3
had direct-correct probability 0.40 vs. 0.90 on the rest, which gave
Pi_1 a partition gap that dominated every cell regardless of n or
beta. v3 redesigns the data-generating process so that the *dominant*
signal for direct correctness is a smooth continuous feature (so
Pi_2 has something to learn), with only a small residual cluster
heterogeneity (so Pi_1 can still escape at low n-beta^2).

Data-generating process:
  * 4 clusters on R^6, well-separated Gaussian means, p=(0.35,0.25,0.20,0.20).
  * A global latent score s = x @ w with w in unit-norm.
  * Direct-correct probability depends on both the latent score and
    a small cluster bump:
        p_correct(x) = sigmoid(beta_knob * s(x) + bump(g(x)))
    where bump = (+0.3, +0.3, -0.1, -0.4) -> cluster 3 is the
    partition-escape cluster but with a modest action gap.
  * Loss matrix: L[direct]=0 if correct else 1.0,
                 L[recover]=0.55, L[escalate]=0.32, L[abstain]=0.70.
  * beta_knob controls how well the linear feature s separates c_d.

The point of this setup is to produce three regions:
  (a) small n, small beta_knob -> Pi_0 competitive (no useful score,
      partition gap tiny): baseline region.
  (b) small n, moderate beta_knob -> Pi_1 wins (Bernstein bound blocks
      Pi_2 despite positive beta; cluster bump is enough to escape).
  (c) large n, moderate-to-large beta_knob -> Pi_2 wins (above Bernstein
      threshold, smooth signal dominates partition gap).

Sweep:
  * n   in {150, 300, 600, 1200, 2400, 4800}
  * beta_knob in {0.0, 0.5, 1.0, 1.6, 2.4, 3.5}   (roughly alpha in 0.55..0.90)
  * 3 seeds per cell

The output 2D phase diagram is overlaid with the Bernstein threshold
curve from Theorem 2 for alpha=0.75, q=0.3, delta=0.05.
"""

import argparse, json, os, sys, math
import numpy as np

sys.path.insert(0, './src/controllers')
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

OUT_FIG = './figures'
OUT_ART = './data/unified_v14'

N_ACTIONS = 4
P_CLUSTERS = np.array([0.35, 0.25, 0.20, 0.20])
# v4: larger cluster bumps so clusters 0/1/2 prefer direct
# (p_correct ~ 0.82) while cluster 3 prefers escalate (p_correct ~ 0.27).
# v3 used (+0.3,+0.3,-0.1,-0.4) which left ALL clusters with
# p_correct < 0.68, so even Pi_0 picked escalate and Pi_1 had no
# escape. v4 fixes this by lifting clusters 0/1/2 well above the
# direct/escalate threshold and pushing cluster 3 well below it.
CLUSTER_BUMP = np.array([1.50, 1.50, 1.50, -1.00])
L_RECOVER = 0.55
L_ESCALATE = 0.32
L_ABSTAIN = 0.70


def gen_data(n, beta_knob, seed):
    rng = np.random.default_rng(seed)
    g = rng.choice(4, size=n, p=P_CLUSTERS)
    d = 6
    means = np.zeros((4, d))
    for k in range(4):
        means[k, k] = 2.0
    X = means[g] + rng.normal(size=(n, d))
    # global latent direction
    w = np.array([0.30, 0.30, 0.30, 0.30, 0.60, 0.70])
    w = w / np.linalg.norm(w)
    s = X @ w + rng.normal(scale=0.3, size=n)
    # correctness logit
    logit = beta_knob * s + CLUSTER_BUMP[g]
    p_correct = 1.0 / (1.0 + np.exp(-logit))
    c_d = (rng.uniform(size=n) < p_correct).astype(int)
    L = np.zeros((n, N_ACTIONS))
    L[:, 0] = np.where(c_d == 1, 0.0, 1.0)
    L[:, 1] = L_RECOVER
    L[:, 2] = L_ESCALATE
    L[:, 3] = L_ABSTAIN
    C = np.zeros((n, N_ACTIONS), dtype=int)
    C[:, 0] = c_d; C[:, 1] = 1; C[:, 2] = 1; C[:, 3] = 1
    return X, L, C, c_d


def alpha_min_compute(L, c_d):
    L_a = L_ESCALATE
    mask_r = c_d == 1
    mask_w = c_d == 0
    if mask_r.sum() == 0 or mask_w.sum() == 0:
        return 0.5
    L_r = float(L[mask_r, 0].mean())
    L_w = float(L[mask_w, 0].mean())
    if L_w - L_r < 1e-9:
        return 0.5
    return (L_a - L_r) / (L_w - L_r)


# --- Families ---
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
    clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs',
                             random_state=seed)
    clf.fit(Xe, ye, sample_weight=we)
    return sc, clf

def pred_pi2(sc, clf, X_te):
    return clf.classes_[clf.predict_proba(sc.transform(X_te)).argmax(axis=1)].astype(int)


def cv_eval_pi0(X, L, n_splits=5, seed=0):
    n = len(L); chosen = np.zeros(n, dtype=int)
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(L):
        a = fit_pi0(X[tr], L[tr])
        chosen[te] = a
    return float(L[np.arange(n), chosen].mean())

def cv_eval_pi1(X, L, n_splits=5, seed=0, K=4):
    n = len(L); chosen = np.zeros(n, dtype=int)
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(L):
        sc, km, act = fit_pi1(X[tr], L[tr], K=K, seed=seed)
        chosen[te] = pred_pi1(sc, km, act, X[te])
    return float(L[np.arange(n), chosen].mean())

def cv_eval_pi2(X, L, n_splits=5, seed=0):
    n = len(L); chosen = np.zeros(n, dtype=int)
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(L):
        sc, clf = fit_pi2(X[tr], L[tr], seed=seed)
        chosen[te] = pred_pi2(sc, clf, X[te])
    return float(L[np.arange(n), chosen].mean())


def run_grid(n_vals, beta_vals, n_seeds=3):
    results = {}
    for n in n_vals:
        for bk in beta_vals:
            l0s, l1s, l2s, bs, as_ = [], [], [], [], []
            for s in range(n_seeds):
                seed = 42 + 31 * s
                X, L, C, c_d = gen_data(n, bk, seed)
                l0s.append(cv_eval_pi0(X, L, seed=seed))
                l1s.append(cv_eval_pi1(X, L, seed=seed, K=4))
                l2s.append(cv_eval_pi2(X, L, seed=seed))
                sc = StandardScaler().fit(X)
                clf_auc = LogisticRegression(max_iter=2000).fit(sc.transform(X), c_d)
                try:
                    auc = roc_auc_score(c_d, clf_auc.predict_proba(sc.transform(X))[:, 1])
                except Exception:
                    auc = 0.5
                amin = alpha_min_compute(L, c_d)
                as_.append(auc); bs.append(max(auc - amin, 0.0))
            r = {
                'pi0': float(np.mean(l0s)), 'pi0_std': float(np.std(l0s)),
                'pi1': float(np.mean(l1s)), 'pi1_std': float(np.std(l1s)),
                'pi2': float(np.mean(l2s)), 'pi2_std': float(np.std(l2s)),
                'alpha_emp': float(np.mean(as_)),
                'beta': float(np.mean(bs)),
            }
            vals = {'pi0': r['pi0'], 'pi1': r['pi1'], 'pi2': r['pi2']}
            r['winner'] = min(vals, key=lambda k: vals[k])
            results[(n, bk)] = r
            print(f'n={n:5d}  bk={bk:4.1f}  alpha={r["alpha_emp"]:.3f}  '
                  f'beta={r["beta"]:.3f}  pi0={r["pi0"]:.4f}  '
                  f'pi1={r["pi1"]:.4f}  pi2={r["pi2"]:.4f}  -> {r["winner"]}')
    return results


def plot_phase(results, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'font.family': 'serif',
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})

    CMAP = {'pi0': '#555555', 'pi1': '#1f77b4', 'pi2': '#d62728'}
    LBL = {'pi0': r'$\Pi_0$ wins', 'pi1': r'$\Pi_1$ wins', 'pi2': r'$\Pi_2$ wins'}

    fig, ax = plt.subplots(figsize=(6.1, 3.9))

    for (n, bk), r in results.items():
        ax.scatter(n, r['beta'], s=170, c=CMAP[r['winner']], marker='s',
                   edgecolor='black', linewidth=0.8, zorder=3)

    alpha = 0.75; q = 0.3; delta = 0.05
    lt = math.log(2 / delta)
    betas = np.linspace(0.02, 0.40, 300)
    n_min = 2 * alpha * (1 - alpha) * lt / (q * betas**2)
    ax.plot(n_min, betas, color='black', lw=1.8, linestyle='--',
            label=r'Bernstein $T_2$ (Thm~2)', zorder=2)

    ax.set_xscale('log')
    ax.set_xlabel('sample size $n$ (log scale)')
    ax.set_ylabel(r'empirical margin $\beta = \alpha_{\mathrm{emp}} - \alpha_{\min}$')
    ax.set_title('Synthetic phase diagram: empirical winner vs.\\ theoretical threshold')
    ax.grid(True, which='both', alpha=0.25, linestyle=':')

    ns = sorted(set(n for n, _ in results.keys()))
    ax.set_xlim(ns[0] * 0.7, ns[-1] * 1.5)
    betas_obs = [r['beta'] for r in results.values()]
    ax.set_ylim(-0.005, max(betas_obs) * 1.25 + 0.02)

    from matplotlib.lines import Line2D
    leg = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=CMAP['pi0'],
               markeredgecolor='black', markersize=11, label=LBL['pi0']),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=CMAP['pi1'],
               markeredgecolor='black', markersize=11, label=LBL['pi1']),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=CMAP['pi2'],
               markeredgecolor='black', markersize=11, label=LBL['pi2']),
        Line2D([0], [0], color='black', linestyle='--', lw=1.8,
               label=r'Bernstein $T_2$ threshold'),
    ]
    ax.legend(handles=leg, loc='upper right', framealpha=0.93, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f'wrote {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=3)
    args = p.parse_args()

    n_vals = [150, 300, 600, 1200, 2400, 4800]
    beta_knob_vals = [0.0, 0.5, 1.0, 1.6, 2.4, 3.5]

    print(f'\nSynth phase-diagram v3: {len(n_vals)} x {len(beta_knob_vals)} '
          f'cells, {args.seeds} seeds\n')

    results = run_grid(n_vals, beta_knob_vals, n_seeds=args.seeds)

    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs(OUT_ART, exist_ok=True)
    json.dump({f'{k[0]}_{k[1]}': v for k, v in results.items()},
              open(f'{OUT_ART}/synth_regime_phase_v4.json', 'w'), indent=2)
    plot_phase(results, f'{OUT_FIG}/fig_phase_synth.pdf')

    from collections import Counter
    print('\nWinner distribution:',
          dict(Counter(r['winner'] for r in results.values())))


if __name__ == '__main__':
    main()
