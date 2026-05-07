"""Loss-weight sensitivity for the four core ARC benchmarks.

This script reuses the released per-action components in
data/unified_v4/*_loss_matrix.json:

    correct, risk, cost

and recomputes the loss matrix under local perturbations of the canonical
paper loss

    L = 1.0 * (1 - correct) + 1.0 * risk + 0.05 * cost.

For each perturbation, it evaluates the deployable Pi_0/Pi_1/Pi_2 family
pool under strict 5-fold CV and reports the best class. It intentionally
excludes Pi_3/oracle channels because this is a sensitivity check for the
four core benchmark table.
"""

import json
import os
import sys
import argparse
from collections import Counter

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(ROOT, "scripts", "arc") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "scripts", "arc"))

from controller_arc_p_v14 import (  # noqa: E402
    ACTIONS,
    A_IDX,
    AlwaysDirect,
    FairFixedCV,
    GroupRoutingKMeans,
    LearnedHGBC,
    SelectiveCalibrated,
    cv_eval_family,
)


BENCHES = ["POPE", "HallusionBench", "A-OKVQA", "FEVER"]
DATA_V4 = os.path.join(ROOT, "data", "unified_v4")
DATA_V10 = os.path.join(ROOT, "data", "unified_v10")
OUT_DIR = os.path.join(ROOT, "data", "unified_v14")

WEIGHT_GRID = [
    ("canonical", (1.00, 1.00, 0.050)),
    ("correct_low", (0.75, 1.00, 0.050)),
    ("correct_high", (1.25, 1.00, 0.050)),
    ("risk_low", (1.00, 0.75, 0.050)),
    ("risk_high", (1.00, 1.25, 0.050)),
    ("cost_low", (1.00, 1.00, 0.025)),
    ("cost_high", (1.00, 1.00, 0.100)),
]


def load_components_and_features(bench):
    loss_path = os.path.join(DATA_V4, f"{bench}_loss_matrix.json")
    feat_path = os.path.join(DATA_V10, f"{bench}_features.npz")
    meta_path = os.path.join(DATA_V10, f"{bench}_feature_names.json")

    rows_all = json.load(open(loss_path))["rows"]
    feat_npz = np.load(feat_path, allow_pickle=True)
    Xfull_all = feat_npz["X"]
    sample_ids_feats = list(feat_npz["sample_ids"])
    feat_meta = json.load(open(meta_path))

    sid_to_loss_idx = {r["sample_id"]: i for i, r in enumerate(rows_all)}
    keep_feat, keep_loss = [], []
    for fi, sid in enumerate(sample_ids_feats):
        if sid in sid_to_loss_idx:
            keep_feat.append(fi)
            keep_loss.append(sid_to_loss_idx[sid])

    rows = [rows_all[i] for i in keep_loss]
    Xfull = np.nan_to_num(Xfull_all[keep_feat].astype(np.float64))
    n_text = feat_meta.get("n_text_features", Xfull.shape[1])
    n_scalar = feat_meta.get("n_scalar_features", n_text - 1920)
    Xscalar = np.nan_to_num(Xfull[:, :n_scalar].astype(np.float64))

    n = len(rows)
    C = np.zeros((n, len(ACTIONS)), dtype=int)
    R = np.zeros((n, len(ACTIONS)), dtype=float)
    K = np.zeros((n, len(ACTIONS)), dtype=float)
    stored_L = np.zeros((n, len(ACTIONS)), dtype=float)
    for i, r in enumerate(rows):
        for a in ACTIONS:
            ai = A_IDX[a]
            entry = r["actions"][a]
            C[i, ai] = int(entry["correct"])
            R[i, ai] = float(entry.get("risk", 0.0))
            K[i, ai] = float(entry.get("cost", 0.0))
            stored_L[i, ai] = float(entry["loss"])

    return Xscalar, C, R, K, stored_L


def loss_from_weights(C, R, K, weights):
    w_c, w_h, w_k = weights
    return w_c * (1.0 - C.astype(float)) + w_h * R + w_k * K


def family_pool():
    return [
        ("always_direct", "Pi_0", AlwaysDirect),
        ("fair_fixed_train", "Pi_0", FairFixedCV),
        ("KM_K04", "Pi_1", lambda: GroupRoutingKMeans(K=4)),
        ("KM_K05", "Pi_1", lambda: GroupRoutingKMeans(K=5)),
        ("KM_K06", "Pi_1", lambda: GroupRoutingKMeans(K=6)),
        ("KM_K08", "Pi_1", lambda: GroupRoutingKMeans(K=8)),
        ("HGBC_md3", "Pi_2", lambda: LearnedHGBC(max_depth=3)),
        ("HGBC_md4", "Pi_2", lambda: LearnedHGBC(max_depth=4)),
        ("selective_C0.3", "Pi_2", lambda: SelectiveCalibrated(C_reg=0.3)),
    ]


def evaluate_bench_variant(bench, X, C, R, K, weights, seeds):
    L = loss_from_weights(C, R, K, weights)
    out = {
        "bench": bench,
        "weights": {"w_c": weights[0], "w_h": weights[1], "w_k": weights[2]},
        "always_direct": float(L[:, A_IDX["answer_direct"]].mean()),
        "per_family": {},
        "per_class_best": {},
    }

    for fam_name, cls, fac in family_pool():
        losses = []
        for seed in seeds:
            _, loss = cv_eval_family(fac, X, L, C, n_splits=5, seed=seed)
            losses.append(float(loss))
        out["per_family"][fam_name] = {
            "class": cls,
            "mean": float(np.mean(losses)),
            "std": float(np.std(losses)),
            "losses_per_seed": losses,
        }

    for cls in ["Pi_0", "Pi_1", "Pi_2"]:
        subset = {
            k: v for k, v in out["per_family"].items()
            if v["class"] == cls
        }
        best_name = min(subset, key=lambda k: subset[k]["mean"])
        out["per_class_best"][cls] = {
            "family": best_name,
            **subset[best_name],
        }

    best_cls = min(out["per_class_best"], key=lambda k: out["per_class_best"][k]["mean"])
    out["winner_class"] = best_cls
    out["winner_family"] = out["per_class_best"][best_cls]["family"]
    out["winner_loss"] = out["per_class_best"][best_cls]["mean"]
    out["runner_up_class"] = sorted(
        out["per_class_best"],
        key=lambda k: out["per_class_best"][k]["mean"],
    )[1]
    out["margin_to_runner_up"] = (
        out["per_class_best"][out["runner_up_class"]]["mean"]
        - out["winner_loss"]
    )
    return out


def verify_canonical_matches_stored(bench, C, R, K, stored_L):
    canonical_L = loss_from_weights(C, R, K, WEIGHT_GRID[0][1])
    return float(np.max(np.abs(canonical_L - stored_L)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--benches", nargs="*", default=BENCHES)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    seeds = [42 + 31 * i for i in range(args.seeds)]
    results = {
        "description": "Local one-factor perturbation of canonical core loss weights.",
        "canonical_loss": "1.0*(1-correct) + 1.0*risk + 0.05*cost",
        "seeds": seeds,
        "weight_grid": [
            {"name": name, "w_c": w[0], "w_h": w[1], "w_k": w[2]}
            for name, w in WEIGHT_GRID
        ],
        "benchmarks": {},
    }

    print(f"Loss-weight sensitivity: strict 5-fold CV, {len(seeds)} seeds", flush=True)
    print("canonical loss = 1*(1-correct) + 1*risk + 0.05*cost\n")

    for bench in args.benches:
        X, C, R, K, stored_L = load_components_and_features(bench)
        max_abs = verify_canonical_matches_stored(bench, C, R, K, stored_L)
        if max_abs > 1e-10:
            raise RuntimeError(f"{bench}: canonical recomputation mismatch {max_abs}")

        bench_results = {}
        print(f"=== {bench} ===", flush=True)
        print(f"{'variant':<14} {'weights':<24} {'winner':<5} {'family':<16} {'loss':>8} {'margin':>8}", flush=True)
        for variant_name, weights in WEIGHT_GRID:
            r = evaluate_bench_variant(bench, X, C, R, K, weights, seeds)
            bench_results[variant_name] = r
            print(
                f"{variant_name:<14} "
                f"({weights[0]:.2f},{weights[1]:.2f},{weights[2]:.3f})   "
                f"{r['winner_class']:<5} {r['winner_family']:<16} "
                f"{r['winner_loss']:>8.4f} {r['margin_to_runner_up']:>8.4f}"
                ,
                flush=True,
            )

        counts = Counter(r["winner_class"] for r in bench_results.values())
        canonical_winner = bench_results["canonical"]["winner_class"]
        stable = sum(
            1 for r in bench_results.values()
            if r["winner_class"] == canonical_winner
        )
        print(
            f"winner counts: {dict(counts)}; canonical winner "
            f"{canonical_winner} stable in {stable}/{len(WEIGHT_GRID)} variants\n"
            ,
            flush=True,
        )
        results["benchmarks"][bench] = {
            "max_abs_canonical_recompute_error": max_abs,
            "canonical_winner": canonical_winner,
            "stable_count": stable,
            "num_variants": len(WEIGHT_GRID),
            "winner_counts": dict(counts),
            "variants": bench_results,
        }

    out_path = os.path.join(OUT_DIR, "loss_weight_sensitivity.json")
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
