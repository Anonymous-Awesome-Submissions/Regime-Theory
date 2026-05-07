"""Compute α_min, the critical AUC threshold for selective gain to exist.

Following the Chow-1970 / Franc-2023 / Goren-2024 reject-option theory
adapted to the bounded-AUC selective controller problem.

For each bench:
  - L_r := E[L(direct) | direct correct]
  - L_w := E[L(direct) | direct wrong]
  - L_a := E[L(abstain)] = constant
  - p_w := P(direct wrong)
  - α_emp := empirical CV-AUC of best instance-level predictor

Critical AUC threshold for selective gain (asymptotic, q → 0+):
    α_min = (L_a - L_r) / (L_w - L_r)

Interpretation: any rejector built on a score with AUC < α_min cannot
beat always_direct at any coverage. (This is necessary; sufficiency needs
the score to be properly calibrated.)

Compute α_min and α_emp for all 4 benches, plus the "feasibility margin"
α_emp - α_min. A positive margin means selective gain is possible; a
negative margin means it is provably impossible.
"""
import json, sys, os
import numpy as np
sys.path.insert(0, './src/controllers')
import controller_arc_p_v11 as v11
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


def compute_for_bench(bench):
    rows, Xstd, Xscalar, Xtxt, L, C, R, K, cats, vis, sets = v11.load_bench(bench)
    n = len(L)
    direct_correct = C[:, 0]
    L_direct = L[:, 0]
    L_a = L[:, 3].mean()
    L_abstain = L_a
    L_r = L_direct[direct_correct == 1].mean()
    L_w = L_direct[direct_correct == 0].mean()
    p_w = (direct_correct == 0).mean()
    fair = float(L_direct.mean())

    # α_min from reject-option theory
    if L_w == L_r:
        alpha_min = 1.0
    else:
        alpha_min = (L_abstain - L_r) / (L_w - L_r)
    alpha_min = max(0.0, min(1.0, alpha_min))

    # Empirical AUC ceiling: best LR / HGBC over the scalar block
    Xs = np.nan_to_num(Xscalar)
    aucs = {}
    # LR with several C
    for C_reg in [0.01, 0.05, 0.1, 0.3, 1.0]:
        preds = np.zeros(n)
        for tr, te in KFold(5, shuffle=True, random_state=42).split(Xs):
            sc = StandardScaler().fit(Xs[tr])
            clf = LogisticRegression(max_iter=2000, C=C_reg)
            clf.fit(sc.transform(Xs[tr]), direct_correct[tr])
            preds[te] = clf.predict_proba(sc.transform(Xs[te]))[:, 1]
        aucs[f'lr_C{C_reg}'] = roc_auc_score(direct_correct, preds)
    alpha_emp = max(aucs.values())
    margin = alpha_emp - alpha_min

    print(f'\n=== {bench} (n={n}) ===')
    print(f'  L_r (direct correct)  = {L_r:.4f}')
    print(f'  L_w (direct wrong)    = {L_w:.4f}')
    print(f'  L_a (abstain)         = {L_abstain:.4f}')
    print(f'  p_w                   = {p_w:.4f}')
    print(f'  fair_fixed (= L_dir)  = {fair:.4f}')
    print(f'  α_min (critical AUC for gain) = {alpha_min:.4f}')
    print(f'  α_emp (best CV-LR AUC scalar) = {alpha_emp:.4f}')
    print(f'  feasibility margin (α_emp − α_min) = {margin:+.4f}')
    print(f'  per-C AUCs: {aucs}')

    # Maximum achievable selective gain bound under perfect ranking:
    # ΔL_max ≈ ∫_{q*}^0 (α_emp · (L_w - L_r) - (L_a - L_r)) dq
    # The integrand is constant in q for the asymptotic bound
    instant_gain_per_unit_coverage = alpha_emp * (L_w - L_r) - (L_a - L_r)
    print(f'  Asymptotic gain per unit q if α=α_emp: {instant_gain_per_unit_coverage:+.4f}')
    return {
        'bench': bench, 'n': n, 'L_r': L_r, 'L_w': L_w, 'L_a': L_abstain,
        'p_w': p_w, 'fair': fair,
        'alpha_min': alpha_min, 'alpha_emp': alpha_emp,
        'margin': margin,
        'gain_per_unit_coverage_at_alpha_emp': instant_gain_per_unit_coverage,
        'aucs': aucs,
    }


def main():
    out = {}
    for bench in ['POPE', 'HallusionBench', 'A-OKVQA', 'FEVER']:
        out[bench] = compute_for_bench(bench)
    print('\n\n=== SUMMARY ===')
    print(f'{"bench":<18}{"L_r":>9}{"L_w":>9}{"L_a":>9}{"α_min":>9}{"α_emp":>9}{"margin":>10}{"gain":>10}')
    for b, r in out.items():
        print(f'{b:<18}{r["L_r"]:>9.4f}{r["L_w"]:>9.4f}{r["L_a"]:>9.4f}'
              f'{r["alpha_min"]:>9.4f}{r["alpha_emp"]:>9.4f}'
              f'{r["margin"]:>+10.4f}{r["gain_per_unit_coverage_at_alpha_emp"]:>+10.4f}')
    os.makedirs('./data/unified_v15', exist_ok=True)
    json.dump(out, open('./data/unified_v15/alpha_min.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
