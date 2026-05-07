"""Compute every number that needs to appear in the revised paper.tex.
Prints them in a structured form so the editor can copy-paste."""
import json, sys, os, math
import numpy as np
sys.path.insert(0, './src/controllers')
import controller_arc_p_v11 as v11
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

DATA = './data/unified_v14'

print("=" * 80)
print("MASTER NUMBER REGISTRY (new judge = InternVL2.5-8B)")
print("=" * 80)

# ---------- Per-bench Π0/Π1/Π2/Π3 best ----------
print("\n### Table 1: per-class winners ###")
results = {}
for b in ['HallusionBench', 'POPE', 'A-OKVQA', 'FEVER']:
    d = json.load(open(f'{DATA}/controller_v14_{b}.json'))
    pcb = d['per_class_best']
    pi0_best = pcb['Pi_0']['mean']
    always_direct = d['always_direct']
    results[b] = {'pi0_best': pi0_best, 'always_direct': always_direct}
    print(f'\n{b}  (n={d["n"]})  Π₀_best={pi0_best:.4f}  (always_direct={always_direct:.4f})')
    for cls in ['Pi_0', 'Pi_1', 'Pi_2', 'Pi_3']:
        v = pcb[cls]
        results[b][cls] = (v['mean'], v['std'], v['family'])
        d_vs_pi0 = v['mean'] - pi0_best
        print(f"  {cls}: {v['family']:<22} {v['mean']:.4f}±{v['std']:.4f}   Δ vs Π₀_best={d_vs_pi0:+.4f}")

# ---------- Theory quantities (Table 2) ----------
print("\n\n### Table 2: theory quantities (recompute from new loss matrices) ###")
THEORY = {}
for b in ['POPE', 'HallusionBench', 'A-OKVQA', 'FEVER']:
    rows, Xstd, Xscalar, Xtxt, L, C, R, K, cats, vis, sets = v11.load_bench(b)
    n = L.shape[0]
    direct_correct = C[:, 0]
    L_direct = L[:, 0]
    L_a = float(L[:, 3].mean())
    L_r = float(L_direct[direct_correct == 1].mean())
    L_w = float(L_direct[direct_correct == 0].mean())
    p_w = float((direct_correct == 0).mean())
    a_min = max(0.0, min(1.0, (L_a - L_r) / (L_w - L_r) if L_w != L_r else 1.0))

    # α_emp via best LR-AUC over scalar block
    Xs = np.nan_to_num(Xscalar)
    best_auc = 0.0
    for C_reg in [0.01, 0.05, 0.1, 0.3, 1.0]:
        preds = np.zeros(n)
        for tr, te in KFold(5, shuffle=True, random_state=42).split(Xs):
            sc = StandardScaler().fit(Xs[tr])
            clf = LogisticRegression(C=C_reg, max_iter=1000, random_state=42).fit(sc.transform(Xs[tr]), direct_correct[tr])
            preds[te] = clf.predict_proba(sc.transform(Xs[te]))[:, 1]
        auc = roc_auc_score(direct_correct, preds)
        if auc > best_auc: best_auc = auc
    a_emp = best_auc
    beta = a_emp - a_min
    nbeta2 = n * beta * beta
    n_min = math.ceil(2 * a_emp * (1 - a_emp) * math.log(2/0.05) / (0.3 * beta * beta))
    THEORY[b] = dict(n=n, alpha=a_emp, alpha_min=a_min, beta=beta, nbeta2=nbeta2, nmin=n_min,
                      L_r=L_r, L_w=L_w, L_a=L_a, p_w=p_w)
    print(f"{b}  n={n}  α_emp={a_emp:.4f}  α_min={a_min:.4f}  β={beta:.4f}  nβ²={nbeta2:.2f}  n_min={n_min}")

# Lemma-1 ceiling for POPE:
b='POPE'
ceiling = 0.3 * THEORY[b]['beta'] * (THEORY[b]['L_w'] - THEORY[b]['L_r'])
print(f"\nPOPE Lemma-1 asymptotic ceiling q*β*(L_w-L_r) at q=0.3 = {ceiling:.4f}")

# ---------- L2D Table 3 ----------
print("\n\n### Table 3: L2D best per bench ###")
for b in ['HallusionBench', 'POPE', 'A-OKVQA', 'FEVER']:
    d = json.load(open(f'{DATA}/l2d_baselines_{b}.json'))
    fair = d['always_direct']
    fams = d['l2d_per_family']
    moz = min((v for k,v in fams.items() if 'mozannar' in k.lower()), key=lambda x: x['mean'])
    nar = min((v for k,v in fams.items() if 'narasim' in k.lower()), key=lambda x: x['mean'])
    print(f"{b}  Π₀={fair:.4f}")
    print(f"  Mozannar best:  {moz['mean']:.4f}±{moz['std']:.4f}   Δ vs Π₀={moz['mean']-fair:+.4f}")
    print(f"  Narasimhan best:{nar['mean']:.4f}±{nar['std']:.4f}   Δ vs Π₀={nar['mean']-fair:+.4f}")

# ---------- Tree router Table 6 ----------
print("\n\n### Table 6: tree router best per bench ###")
for b in ['HallusionBench', 'POPE', 'A-OKVQA', 'FEVER']:
    d = json.load(open(f'{DATA}/tree_router_{b}.json'))
    fair = d['always_direct']
    fams = {k:v for k,v in d['tree_per_family'].items() if isinstance(v, dict) and 'mean' in v}
    best_t = min(fams.items(), key=lambda x: x[1]['mean'])
    print(f"{b}  Π₀={fair:.4f}  best CART={best_t[0]} {best_t[1]['mean']:.4f}±{best_t[1]['std']:.4f}  Δ={best_t[1]['mean']-fair:+.4f}")

# ---------- HallusionBench K-sweep ----------
print("\n\n### Fig 7 left: K-sweep ###")
d = json.load(open(f'{DATA}/strict_K_sweep.json'))
for b in ['HallusionBench']:
    h = d[b]
    print(f"{b}  always_direct={h['always_direct']:.4f}")
    for K, v in h['K_results'].items():
        print(f"  K={K}  loss={v['mean']:.4f}±{v['std']:.4f}  Δ={v['delta']:+.4f}")

# ---------- Cluster anatomy K=4 ----------
print("\n\n### Fig 7 right: cluster anatomy at K=4 ###")
rows, Xstd, Xscalar, Xtxt, L, C, R, K_arr, cats, vis, sets = v11.load_bench('HallusionBench')
n = L.shape[0]
global_best_a = int(np.argmin(L.mean(axis=0)))
print(f'global best fixed action = action[{global_best_a}]   loss={L[:, global_best_a].mean():.4f}')
Xs = StandardScaler().fit_transform(np.nan_to_num(Xscalar))
labels = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(Xs)
total = 0.0
mx = 0.0
print('  g    p_g     size   cell_best_a  γ_g       p_g·γ_g')
for g in range(4):
    mask = (labels == g)
    p_g = mask.mean()
    Lg = L[mask].mean(axis=0)
    cell_best = int(np.argmin(Lg))
    if cell_best == global_best_a:
        gamma_g = 0.0
    else:
        gamma_g = float(L[mask, global_best_a].mean() - L[mask, cell_best].mean())
    total += p_g * gamma_g
    mx = max(mx, p_g * gamma_g)
    print(f'  {g}    {p_g:.3f}   {mask.sum():<6} {cell_best}            {gamma_g:.4f}    {p_g*gamma_g:.4f}')
print(f'  max p_g·γ_g = {mx:.4f}    sum p_g·γ_g = {total:.4f}')

# Empirical Π₁ KM_K04 vs always_direct
ad = float(L[:, 0].mean())
print(f'\nalways_direct = {ad:.4f}')
print(f'KM_K04 strict-CV gain (from strict_K_sweep):')
ksw = json.load(open(f'{DATA}/strict_K_sweep.json'))['HallusionBench']['K_results']['4']
print(f'  K=4 CV loss={ksw["mean"]:.4f}±{ksw["std"]:.4f}  Δ vs always_direct = {ksw["delta"]:+.4f}')

# ---------- nsweep A-OKVQA ----------
print("\n\n### §6.5 A-OKVQA n-sweep ###")
d = json.load(open(f'{DATA}/subsample_nsweep_A-OKVQA.json'))
print(f'theory n_min = {d["theory"]["n_min"]}')
for s in d['sweep']:
    n = s['target_n']
    p0 = s['families']['pi0_direct']['mean']
    p1 = s['families']['pi1_km']
    p2 = s['families']['pi2_hgbc']
    print(f'  n={n}  Π₀={p0:.4f}  Π₂-Π₀={p2["mean"]-p0:+.4f}±{p2["std"]:.3f}   Π₁-Π₀={p1["mean"]-p0:+.4f}')

# ---------- nsweep FEVER ----------
print("\n\n### §6.5 FEVER n-sweep ###")
d = json.load(open(f'{DATA}/subsample_nsweep_FEVER.json'))
print(f'theory n_min = {d["theory"]["n_min"]}')
for s in d['sweep']:
    n = s['target_n']
    p0 = s['families']['pi0_direct']['mean']
    p1 = s['families']['pi1_km']
    p2 = s['families']['pi2_hgbc']
    print(f'  n={n}  Π₀={p0:.4f}  Π₂-Π₀={p2["mean"]-p0:+.4f}±{p2["std"]:.3f}   Π₁-Π₀={p1["mean"]-p0:+.4f}')

# ---------- Strict-CV ablation Table 4 ----------
print("\n\n### Table 4: strict_cv_ablation HallusionBench ###")
d = json.load(open(f'{DATA}/ablation_strict_hallu_KM_K04.json'))
print(json.dumps(d, indent=1)[:1500])

# ---------- Augment Table 5 ----------
print("\n\n### Table 5: augment ablation ###")
# Canonical from controller_v14
d = json.load(open(f'{DATA}/controller_v14_HallusionBench.json'))
pcb = d['per_class_best']
print(f"Canonical Π₂ best: {pcb['Pi_2']['mean']:.4f}±{pcb['Pi_2']['std']:.4f}")
print(f"Canonical Π₁ best: {pcb['Pi_1']['mean']:.4f}±{pcb['Pi_1']['std']:.4f}")
# Augmented from strict_K_sweep maybe? Actually the augmented Π₂ was via controller_v15_runner.
# For now we keep the canonical numbers.

# ---------- Π₃ rationale Table 7 ----------
print("\n\n### Table 7: A-OKVQA Π₃ rationale ###")
d = json.load(open(f'{DATA}/pi3_rationale_A-OKVQA.json'))
print(json.dumps(d, indent=1)[:2000])

# ---------- Π₂ rich text ----------
print("\n\n### Table 7 mid: Π₂ rich text ###")
d = json.load(open(f'{DATA}/pi2_rich_text_A-OKVQA.json'))
print(json.dumps(d, indent=1)[:2500])

# ---------- §6.4 always_direct decomposition (line 546) ----------
print("\n\n### §6.4 always_direct loss decomposition: 1-mean(c) + mean(risk) + 0.05 ###")
for b in ['HallusionBench','POPE','A-OKVQA','FEVER']:
    rows, Xstd, Xscalar, Xtxt, L, C, R, K, cats, vis, sets = v11.load_bench(b)
    cm = float(C[:,0].mean()); rm = float(R[:,0].mean())
    ad = 1-cm + rm + 0.05
    print(f'{b}: ({1-cm:.4f}) + ({rm:.4f}) + 0.05 = {ad:.4f}    [must equal always_direct={float(L[:,0].mean()):.4f}]')

# ---------- §6.10 auto-pick ----------
print("\n\n### §6.10 nested auto-pick deltas ###")
for b in ['HallusionBench','POPE','A-OKVQA','FEVER']:
    d = json.load(open(f'{DATA}/controller_v14_{b}.json'))
    ap = d.get('auto_pick', {})
    print(f'{b} auto_pick keys: {list(ap.keys()) if isinstance(ap,dict) else type(ap).__name__}')
    if isinstance(ap, dict):
        m = ap.get('mean_loss') or ap.get('mean')
        s = ap.get('std_loss') or ap.get('std', 0)
        if m is not None:
            print(f"  loss={m:.4f}±{s:.4f}   Δ vs always_direct={m-d['always_direct']:+.4f}")
