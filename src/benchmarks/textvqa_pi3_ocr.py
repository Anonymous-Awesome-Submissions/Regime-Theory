"""Deployable Pi_3 OCR-channel stress test on TextVQA.

This experiment is intentionally separate from the four canonical ARC
benchmarks.  It asks whether the top rung of the lattice can win on a real
benchmark when the external prior channel is prediction-time available and not
answer-derived.  TextVQA is a natural fit: the released Rosetta OCR tokens are
computed from the image and are available without using labels.

Protocol
--------
1. Use the official TextVQA train split only to freeze two upstream actions:
   * direct: a question/image-class-only answer predictor.
   * ocr:    an OCR-copy prior trained to select among OCR n-grams.
2. On the official validation split, compute per-sample losses for the two
   actions using TextVQA/VQA-style soft accuracy.
3. Run strict 5-fold x 5-seed CV over policy classes:
   * Pi_0: fixed actions.
   * Pi_1: low-depth CART partition routers over scalar non-OCR features.
   * Pi_2: logistic action router over scalar non-OCR features.
   * Pi_3: OCR-prior confidence gate + a Pi_2 residual on the non-gated pool.

The lower rungs never see OCR token text or OCR-prior confidence.  Pi_3 is the
only class allowed to consume the external OCR prior channel at prediction time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import FeatureHasher, DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data", "TextVQA")
OUT = os.path.join(ROOT, "data", "unified_v14", "textvqa_pi3_ocr.json")

TRAIN_QA = os.path.join(DATA, "TextVQA_0.5.1_train.json")
VAL_QA = os.path.join(DATA, "TextVQA_0.5.1_val.json")
TRAIN_OCR = os.path.join(DATA, "TextVQA_Rosetta_OCR_v0.2_train.json")
VAL_OCR = os.path.join(DATA, "TextVQA_Rosetta_OCR_v0.2_val.json")

ARTICLES = {"a", "an", "the"}
PUNCT = re.compile(r"[^0-9a-z]+")
WORD = re.compile(r"[a-z0-9]+")


def norm_text(s: str) -> str:
    s = s.lower().strip()
    s = PUNCT.sub(" ", s)
    toks = [t for t in s.split() if t and t not in ARTICLES]
    return " ".join(toks)


def tokenize(s: str) -> List[str]:
    return WORD.findall(s.lower())


def majority_answer(answers: Sequence[str]) -> str:
    counts = Counter(norm_text(a) for a in answers if norm_text(a))
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def vqa_score(pred: str, answers: Sequence[str]) -> float:
    p = norm_text(pred)
    if not p:
        return 0.0
    cnt = sum(1 for a in answers if norm_text(a) == p)
    return min(cnt / 3.0, 1.0)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)["data"]


def load_ocr(path: str) -> Dict[str, dict]:
    rows = load_json(path)
    return {r["image_id"]: r for r in rows}


def attach_ocr(qa_rows: List[dict], ocr_by_image: Dict[str, dict]) -> List[dict]:
    out = []
    for r in qa_rows:
        o = ocr_by_image.get(r["image_id"], {})
        rr = dict(r)
        rr["ocr_tokens"] = o.get("ocr_tokens", []) or []
        rr["ocr_info"] = o.get("ocr_info", []) or []
        out.append(rr)
    return out


def make_direct_text(row: dict) -> str:
    classes = " ".join(row.get("image_classes") or [])
    return f"{row['question']} [classes] {classes}"


def fit_direct_predictor(train_rows: List[dict], max_classes: int = 2000):
    y_all = [majority_answer(r["answers"]) for r in train_rows]
    top = {a for a, _ in Counter(y_all).most_common(max_classes) if a}
    keep = [i for i, y in enumerate(y_all) if y in top]
    X_text = [make_direct_text(train_rows[i]) for i in keep]
    y = [y_all[i] for i in keep]
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000),
        ComplementNB(alpha=0.15),
    )
    pipe.fit(X_text, y)
    return pipe


def predict_direct(pipe, rows: List[dict]) -> Tuple[List[str], np.ndarray]:
    X_text = [make_direct_text(r) for r in rows]
    pred = pipe.predict(X_text).tolist()
    if hasattr(pipe[-1], "predict_proba"):
        prob = pipe.predict_proba(X_text)
        conf = prob.max(axis=1)
    else:
        conf = np.ones(len(rows)) * 0.5
    return pred, np.asarray(conf, dtype=float)


@dataclass
class Cand:
    text: str
    start: int
    ngram: int
    freq: int
    top: float = 0.5
    left: float = 0.5
    area: float = 0.0


def ocr_candidates(row: dict, max_ngram: int = 4, max_cands: int = 60) -> List[Cand]:
    toks_raw = row.get("ocr_tokens") or []
    toks = [norm_text(t) for t in toks_raw]
    toks = [t for t in toks if t]
    if not toks:
        return []

    infos = row.get("ocr_info") or []
    seen: Dict[str, Cand] = {}
    counts = Counter(toks)
    for i in range(len(toks)):
        for n in range(1, max_ngram + 1):
            if i + n > len(toks):
                break
            txt = " ".join(toks[i : i + n])
            if not txt or len(txt) > 60:
                continue
            if txt in seen:
                seen[txt].freq += 1
                continue
            top = left = 0.5
            area = 0.0
            if i < len(infos):
                bb = infos[i].get("bounding_box", {}) if isinstance(infos[i], dict) else {}
                top = float(bb.get("top_left_y", 0.5) or 0.5)
                left = float(bb.get("top_left_x", 0.5) or 0.5)
                area = float(bb.get("width", 0.0) or 0.0) * float(bb.get("height", 0.0) or 0.0)
            seen[txt] = Cand(txt, i, n, counts[toks[i]], top, left, area)

    cands = list(seen.values())
    cands.sort(key=lambda c: (c.freq, -c.start, c.ngram, len(c.text)), reverse=True)
    return cands[:max_cands]


TEXT_KEYS = {
    "brand", "name", "word", "text", "say", "says", "spell", "spells", "written",
    "title", "logo", "sign", "number", "date", "year", "time", "price", "website",
    "phone", "address", "label", "company", "author", "team",
}


def question_keys(row: dict) -> List[str]:
    q_toks = tokenize(row["question"])
    keys = sorted(set(q_toks) & TEXT_KEYS)
    if q_toks:
        keys.append(f"q0:{q_toks[0]}")
    return keys


def cand_features(row: dict, cand: Cand, train_answer_freq: Dict[str, int] | None = None,
                  key_answer_freq: Dict[Tuple[str, str], int] | None = None) -> dict:
    q_toks = tokenize(row["question"])
    c_toks = tokenize(cand.text)
    q_set = set(q_toks)
    c_set = set(c_toks)
    train_answer_freq = train_answer_freq or {}
    key_answer_freq = key_answer_freq or {}
    cand_norm = norm_text(cand.text)
    key_freq = max([key_answer_freq.get((k, cand_norm), 0) for k in question_keys(row)] or [0])
    feats = {
        "bias": 1.0,
        f"q0={q_toks[0] if q_toks else '<none>'}": 1.0,
        f"cand_ngram={cand.ngram}": 1.0,
        "cand_len_chars": len(cand.text) / 40.0,
        "cand_len_words": len(c_toks) / 4.0,
        "cand_freq": math.log1p(cand.freq),
        "cand_pos": cand.start / max(1, len(row.get("ocr_tokens") or [])),
        "cand_top": cand.top,
        "cand_left": cand.left,
        "cand_area": cand.area * 100.0,
        "cand_train_ans_freq": math.log1p(train_answer_freq.get(cand_norm, 0)),
        "cand_key_ans_freq": math.log1p(key_freq),
        "q_len": len(q_toks) / 12.0,
        "q_textish": float(any(t in TEXT_KEYS for t in q_toks)),
        "q_cand_overlap": len(q_set & c_set) / max(1, len(c_set)),
        "cand_is_number": float(bool(re.search(r"\d", cand.text))),
        "cand_is_year": float(bool(re.fullmatch(r"(19|20)\d{2}", cand.text))),
        "cand_has_dot": float("." in cand.text),
    }
    for t in q_toks[:12]:
        feats[f"q={t}"] = 1.0
    for t in c_toks[:5]:
        feats[f"c={t}"] = 1.0
    for t in q_set & TEXT_KEYS:
        feats[f"key={t}"] = 1.0
        for ct in c_toks[:3]:
            feats[f"key={t}|c={ct}"] = 1.0
    return feats


def train_answer_frequency_maps(train_rows: List[dict]):
    answer_freq = Counter()
    key_freq = Counter()
    for r in train_rows:
        ans = majority_answer(r["answers"])
        if not ans:
            continue
        answer_freq[ans] += 1
        for k in question_keys(r):
            key_freq[(k, ans)] += 1
    return answer_freq, key_freq


def fit_ocr_prior(train_rows: List[dict], seed: int = 7):
    rng = np.random.default_rng(seed)
    answer_freq, key_freq = train_answer_frequency_maps(train_rows)
    feat_dicts = []
    labels = []
    weights = []
    for r in train_rows:
        cands = ocr_candidates(r)
        if not cands:
            continue
        scored = [(c, vqa_score(c.text, r["answers"])) for c in cands]
        positives = [(c, s) for c, s in scored if s > 0]
        negatives = [(c, s) for c, s in scored if s == 0]
        if len(negatives) > 25:
            idx = rng.choice(len(negatives), size=25, replace=False)
            negatives = [negatives[i] for i in idx]
        for c, s in positives + negatives:
            feat_dicts.append(cand_features(r, c, answer_freq, key_freq))
            labels.append(1 if s > 0 else 0)
            weights.append(1.0 + 2.0 * s if s > 0 else 0.35)

    hasher = FeatureHasher(n_features=2 ** 18, input_type="dict", alternate_sign=False)
    X = hasher.transform(feat_dicts)
    y = np.asarray(labels, dtype=int)
    w = np.asarray(weights, dtype=float)
    clf = SGDClassifier(
        loss="log_loss",
        alpha=2e-6,
        penalty="l2",
        max_iter=12,
        tol=1e-3,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(X, y, sample_weight=w)
    return hasher, clf, answer_freq, key_freq


def predict_ocr_prior(hasher, clf, rows: List[dict], answer_freq: Dict[str, int],
                      key_freq: Dict[Tuple[str, str], int]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    preds, confs, margins = [], [], []
    for r in rows:
        cands = ocr_candidates(r)
        if not cands:
            preds.append("")
            confs.append(0.0)
            margins.append(0.0)
            continue
        X = hasher.transform([cand_features(r, c, answer_freq, key_freq) for c in cands])
        if hasattr(clf, "predict_proba"):
            p = clf.predict_proba(X)[:, 1]
        else:
            d = clf.decision_function(X)
            p = 1.0 / (1.0 + np.exp(-d))
        order = np.argsort(-p)
        best = order[0]
        second = order[1] if len(order) > 1 else order[0]
        preds.append(cands[best].text)
        confs.append(float(p[best]))
        margins.append(float(p[best] - p[second]))
    return preds, np.asarray(confs), np.asarray(margins)


def scalar_feature_dict(row: dict, direct_conf: float) -> dict:
    q_toks = tokenize(row["question"])
    feats = {
        "direct_conf": float(direct_conf),
        "q_len": len(q_toks),
        "n_image_classes": len(row.get("image_classes") or []),
        f"q0={q_toks[0] if q_toks else '<none>'}": 1.0,
    }
    for t in q_toks[:12]:
        feats[f"q={t}"] = 1.0
    for t in set(q_toks) & TEXT_KEYS:
        feats[f"key={t}"] = 1.0
    for c in (row.get("image_classes") or [])[:8]:
        feats[f"cls={norm_text(c)}"] = 1.0
    return feats


def make_scalar_features(rows: List[dict], direct_conf: np.ndarray):
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform([scalar_feature_dict(r, c) for r, c in zip(rows, direct_conf)])
    return X, vec


def prior_feature_dict(row: dict, ocr_pred: str, ocr_conf: float, ocr_margin: float,
                       train_answer_freq: Dict[str, int]) -> dict:
    q_toks = tokenize(row["question"])
    p_toks = tokenize(ocr_pred)
    q_set, p_set = set(q_toks), set(p_toks)
    n_ocr = len(row.get("ocr_tokens") or [])
    feats = {
        "ocr_conf": float(ocr_conf),
        "ocr_margin": float(ocr_margin),
        "n_ocr": min(n_ocr, 40) / 40.0,
        "pred_len_words": len(p_toks) / 4.0,
        "pred_len_chars": len(ocr_pred) / 40.0,
        "pred_has_digit": float(bool(re.search(r"\d", ocr_pred))),
        "pred_is_year": float(bool(re.fullmatch(r"(19|20)\d{2}", ocr_pred))),
        "pred_q_overlap": len(q_set & p_set) / max(1, len(p_set)),
        "pred_train_freq": math.log1p(train_answer_freq.get(norm_text(ocr_pred), 0)),
        "q_textish": float(any(t in TEXT_KEYS for t in q_toks)),
    }
    for t in p_toks[:4]:
        feats[f"pred={t}"] = 1.0
    for t in q_set & TEXT_KEYS:
        feats[f"key={t}"] = 1.0
        for pt in p_toks[:3]:
            feats[f"key={t}|pred={pt}"] = 1.0
    return feats


def make_prior_features(rows: List[dict], ocr_pred: List[str], ocr_conf: np.ndarray,
                        ocr_margin: np.ndarray, train_answer_freq: Dict[str, int]):
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform([
        prior_feature_dict(r, p, c, m, train_answer_freq)
        for r, p, c, m in zip(rows, ocr_pred, ocr_conf, ocr_margin)
    ])
    return X, vec


class AlwaysAction:
    policy_class = "Pi_0"
    def __init__(self, action: int, name: str):
        self.action = action
        self.name = name
    def fit(self, X, L):
        return self
    def predict(self, X):
        return np.full(X.shape[0], self.action, dtype=int)


class FixedBest:
    policy_class = "Pi_0"
    name = "fixed_best_train"
    def fit(self, X, L):
        self.action = int(np.argmin(L.mean(axis=0)))
        return self
    def predict(self, X):
        return np.full(X.shape[0], self.action, dtype=int)


class CARTPartition:
    policy_class = "Pi_1"
    def __init__(self, depth: int = 3, seed: int = 0):
        self.depth = depth
        self.seed = seed
        self.name = f"cart_d{depth}"
    def fit(self, X, L):
        y = np.argmin(L, axis=1)
        self.tree = DecisionTreeClassifier(
            max_depth=self.depth,
            min_samples_leaf=20,
            random_state=self.seed,
        )
        self.tree.fit(X, y)
        leaves = self.tree.apply(X)
        self.leaf_action = {}
        for leaf in np.unique(leaves):
            idx = leaves == leaf
            self.leaf_action[int(leaf)] = int(np.argmin(L[idx].mean(axis=0)))
        self.fallback = int(np.argmin(L.mean(axis=0)))
        return self
    def predict(self, X):
        leaves = self.tree.apply(X)
        return np.array([self.leaf_action.get(int(l), self.fallback) for l in leaves], dtype=int)


class LogisticRouter:
    policy_class = "Pi_2"
    def __init__(self, C: float = 1.0, seed: int = 0):
        self.C = C
        self.seed = seed
        self.name = f"logit_C{C:g}"
    def fit(self, X, L):
        y = np.argmin(L, axis=1)
        if len(np.unique(y)) < 2:
            self.constant = int(y[0])
            self.model = None
        else:
            self.constant = None
            self.model = LogisticRegression(
                C=self.C,
                max_iter=1000,
                class_weight="balanced",
                random_state=self.seed,
            )
            self.model.fit(X, y)
        return self
    def predict(self, X):
        if self.model is None:
            return np.full(X.shape[0], self.constant, dtype=int)
        return self.model.predict(X).astype(int)


class HGBCRouter:
    policy_class = "Pi_2"
    def __init__(self, depth: int = 3, seed: int = 0):
        self.depth = depth
        self.seed = seed
        self.name = f"hgbc_d{depth}"
    def fit(self, X, L):
        y = np.argmin(L, axis=1)
        self.scaler = StandardScaler(with_mean=False)
        Xs = self.scaler.fit_transform(X).toarray()
        if len(np.unique(y)) < 2:
            self.constant = int(y[0]); self.model = None
        else:
            self.constant = None
            self.model = HistGradientBoostingClassifier(
                max_depth=self.depth,
                learning_rate=0.05,
                max_iter=150,
                random_state=self.seed,
            )
            self.model.fit(Xs, y)
        return self
    def predict(self, X):
        if self.model is None:
            return np.full(X.shape[0], self.constant, dtype=int)
        return self.model.predict(self.scaler.transform(X).toarray()).astype(int)


class Pi3OCRGate:
    policy_class = "Pi_3"
    name = "ocr_conf_gate_residual_auto"

    def __init__(self, ocr_conf: np.ndarray, seed: int = 0):
        self.ocr_conf_all = ocr_conf
        self.seed = seed
        self.quantiles = [0.50, 0.60, 0.70, 0.80, 0.88, 0.94]
        self.residual_grid = [
            ("fixed_best", lambda: FixedBest()),
            ("cart_d3", lambda: CARTPartition(depth=3, seed=self.seed)),
            ("cart_d4", lambda: CARTPartition(depth=4, seed=self.seed)),
            ("logit_C0.3", lambda: LogisticRouter(C=0.3, seed=self.seed)),
            ("logit_C1", lambda: LogisticRouter(C=1.0, seed=self.seed)),
        ]

    def _fit_residual(self, X, L, conf, q, residual_factory):
        th = float(np.quantile(conf, q))
        mid = conf < th
        if mid.sum() < 30:
            mid = np.ones(len(conf), dtype=bool)
        residual = residual_factory().fit(X[mid], L[mid])
        return th, residual

    def fit(self, X, L, idx=None):
        if idx is None:
            raise ValueError("Pi3OCRGate.fit requires global row indices via idx")
        conf = self.ocr_conf_all[idx]
        # Inner CV chooses the confidence quantile.  The prior action is action 1.
        best_q, best_residual_name, best_residual_factory, best_loss = None, None, None, float("inf")
        kf = KFold(n_splits=3, shuffle=True, random_state=self.seed)
        for q in self.quantiles:
            for residual_name, residual_factory in self.residual_grid:
                losses = []
                for tr, te in kf.split(np.arange(X.shape[0])):
                    th, residual = self._fit_residual(X[tr], L[tr], conf[tr], q, residual_factory)
                    out = np.empty(len(te), dtype=int)
                    gate = conf[te] >= th
                    out[gate] = 1
                    if (~gate).any():
                        out[~gate] = residual.predict(X[te][~gate])
                    losses.append(float(L[te, out].mean()))
                m = float(np.mean(losses))
                if m < best_loss:
                    best_loss, best_q = m, q
                    best_residual_name, best_residual_factory = residual_name, residual_factory
        self.best_q = best_q
        self.best_residual_name = best_residual_name
        self.threshold, self.residual = self._fit_residual(
            X, L, conf, best_q, best_residual_factory
        )
        return self

    def predict(self, X, idx=None):
        if idx is None:
            raise ValueError("Pi3OCRGate.predict requires global row indices via idx")
        conf = self.ocr_conf_all[idx]
        out = np.empty(X.shape[0], dtype=int)
        gate = conf >= self.threshold
        out[gate] = 1
        if (~gate).any():
            out[~gate] = self.residual.predict(X[~gate])
        return out


class PriorAugmentedLogit(LogisticRouter):
    policy_class = "Pi_3"
    def __init__(self, C: float = 1.0, seed: int = 0):
        super().__init__(C=C, seed=seed)
        self.name = f"prior_aug_logit_C{C:g}"


def cv_eval(factory, X, L, n_splits=5, seed=0, needs_idx=False):
    n = L.shape[0]
    chosen = np.zeros(n, dtype=int)
    for tr, te in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.arange(n)):
        fam = factory()
        if needs_idx:
            fam.fit(X[tr], L[tr], idx=tr)
            chosen[te] = fam.predict(X[te], idx=te)
        else:
            fam.fit(X[tr], L[tr])
            chosen[te] = fam.predict(X[te])
    return chosen, float(L[np.arange(n), chosen].mean())


def summarize_family(losses: List[float], fair: float, policy_class: str):
    return {
        "class": policy_class,
        "mean": float(np.mean(losses)),
        "std": float(np.std(losses)),
        "delta": float(np.mean(losses) - fair),
        "losses_per_seed": [float(x) for x in losses],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    args = p.parse_args()

    print("Loading TextVQA...")
    train = attach_ocr(load_json(TRAIN_QA), load_ocr(TRAIN_OCR))
    val = attach_ocr(load_json(VAL_QA), load_ocr(VAL_OCR))
    print(f"  train={len(train)} val={len(val)}")

    print("Fitting direct question-only predictor...")
    direct_model = fit_direct_predictor(train)
    direct_pred, direct_conf = predict_direct(direct_model, val)
    direct_acc = np.array([vqa_score(p, r["answers"]) for p, r in zip(direct_pred, val)])
    print(f"  direct soft acc={direct_acc.mean():.4f}")

    print("Fitting OCR-copy prior...")
    hasher, ocr_model, answer_freq, key_freq = fit_ocr_prior(train)
    ocr_pred, ocr_conf, ocr_margin = predict_ocr_prior(hasher, ocr_model, val, answer_freq, key_freq)
    ocr_acc = np.array([vqa_score(p, r["answers"]) for p, r in zip(ocr_pred, val)])
    print(f"  OCR soft acc={ocr_acc.mean():.4f}  conf_mean={ocr_conf.mean():.4f}")

    # Two answer actions: direct and OCR-prior.  Loss is 1 - VQA soft accuracy.
    L = np.column_stack([1.0 - direct_acc, 1.0 - ocr_acc])
    X, _ = make_scalar_features(val, direct_conf)
    X_prior, _ = make_prior_features(val, ocr_pred, ocr_conf, ocr_margin, answer_freq)
    X_aug = sparse.hstack([X, X_prior], format="csr")
    fair_const = float(L[:, 0].mean())
    print(f"  always_direct loss={fair_const:.4f}; always_ocr loss={L[:,1].mean():.4f}")

    # Diagnostic: how informative is OCR confidence on the validation split?
    ocr_correct = (ocr_acc > 0).astype(int)
    ocr_conf_auc = float(roc_auc_score(ocr_correct, ocr_conf)) if len(np.unique(ocr_correct)) == 2 else float("nan")

    lower_families = [
        ("always_direct", lambda: AlwaysAction(0, "always_direct"), "Pi_0", False),
        ("always_ocr", lambda: AlwaysAction(1, "always_ocr"), "Pi_0", False),
        ("fixed_best_train", lambda: FixedBest(), "Pi_0", False),
        ("cart_d3", lambda: CARTPartition(depth=3), "Pi_1", False),
        ("cart_d4", lambda: CARTPartition(depth=4), "Pi_1", False),
        ("logit_C0.3", lambda: LogisticRouter(C=0.3), "Pi_2", False),
        ("logit_C1", lambda: LogisticRouter(C=1.0), "Pi_2", False),
    ]
    pi3_families = [
        ("ocr_gate_residual", lambda: Pi3OCRGate(ocr_conf), "Pi_3", True, X),
        ("prior_aug_logit_C0.3", lambda: PriorAugmentedLogit(C=0.3), "Pi_3", False, X_aug),
        ("prior_aug_logit_C1", lambda: PriorAugmentedLogit(C=1.0), "Pi_3", False, X_aug),
        ("prior_aug_logit_C3", lambda: PriorAugmentedLogit(C=3.0), "Pi_3", False, X_aug),
    ]
    seeds = list(range(42, 42 + args.seeds * 31, 31))
    per_family = {}
    print("\nStrict CV over policy classes")
    for name, fac, cls, needs_idx in lower_families:
        losses = []
        for s in seeds:
            _, loss = cv_eval(fac, X, L, n_splits=5, seed=s, needs_idx=needs_idx)
            losses.append(loss)
        per_family[name] = summarize_family(losses, fair_const, cls)
        print(f"  {name:<20} {cls:<4} {np.mean(losses):.4f} ± {np.std(losses):.4f}  Δ={np.mean(losses)-fair_const:+.4f}")
    for name, fac, cls, needs_idx, Xfam in pi3_families:
        losses = []
        for s in seeds:
            _, loss = cv_eval(fac, Xfam, L, n_splits=5, seed=s, needs_idx=needs_idx)
            losses.append(loss)
        per_family[name] = summarize_family(losses, fair_const, cls)
        print(f"  {name:<20} {cls:<4} {np.mean(losses):.4f} ± {np.std(losses):.4f}  Δ={np.mean(losses)-fair_const:+.4f}")

    per_class_best = {}
    for cls in ["Pi_0", "Pi_1", "Pi_2", "Pi_3"]:
        cands = [(n, r) for n, r in per_family.items() if r["class"] == cls]
        best_name, best = min(cands, key=lambda kv: kv[1]["mean"])
        per_class_best[cls] = {"family": best_name, **best}

    best_deployable_lower = min(
        [(c, r) for c, r in per_class_best.items() if c in {"Pi_0", "Pi_1", "Pi_2"}],
        key=lambda kv: kv[1]["mean"],
    )
    pi3 = per_class_best["Pi_3"]
    summary = {
        "best_lower_class": best_deployable_lower[0],
        "best_lower_family": best_deployable_lower[1]["family"],
        "best_lower_mean": best_deployable_lower[1]["mean"],
        "best_pi3_family": pi3["family"],
        "best_pi3_mean": pi3["mean"],
        "pi3_minus_best_lower": pi3["mean"] - best_deployable_lower[1]["mean"],
        "verdict": "Pi_3_wins" if pi3["mean"] < best_deployable_lower[1]["mean"] else "lower_wins",
    }
    print("\nSummary")
    print(json.dumps(summary, indent=2))

    out = {
        "bench": "TextVQA-OCR",
        "n": len(val),
        "protocol": "official train freezes direct/OCR actions; strict val 5-fold x seed class routing",
        "action_soft_accuracy": {
            "direct": float(direct_acc.mean()),
            "ocr_prior": float(ocr_acc.mean()),
            "ocr_conf_auc_for_ocr_correct": ocr_conf_auc,
        },
        "always_direct": fair_const,
        "per_family": per_family,
        "per_class_best": per_class_best,
        "summary": summary,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
