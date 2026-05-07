"""
Standalone SMS-Spam experiment — empirical witness for Theorem 1 (residual bound).

Setup chosen via diagnostic in find_pi0_v2.py:
  - direct:   TF-IDF (1,2)-gram + LinearSVC (calibrated)   ~99.10%
  - retrieve: TF-IDF (1,1)-gram + KNN(k=10, cosine)        ~98.11%
  - defer:    TF-IDF (1,1)-gram + LogisticRegression       ~97.58%
  - abstain:  fixed-loss action

This is the saturated/strong-direct regime where defer is NOT better than direct
(simulating a real deployment with a strong task-specific direct model + a
generic fallback). Diagnostics: P(R)=0.009, Oracle Π₂ gain=0.0029, Π₁ gain=0.000
— all below the Theorem-1 thresholds, so Π₀ is predicted to win.

CPU-only, runs in <2 min on a login node.
"""
import argparse, json, os, sys
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, './src/controllers')
import controller_arc_p_v14 as v14

OUT = "./data/unified_v14"
ACTIONS = v14.ACTIONS
A_IDX = v14.A_IDX

COSTS = {'answer_direct': 1.0, 'recover_then_answer': 2.0,
         'escalate_to_stronger_model': 2.3, 'abstain': 0.0}


def margin_top2(probs):
    sp = np.sort(probs, axis=1)
    m = sp[:, -1] - sp[:, -2]
    return m, np.log(sp[:, -1] + 1e-12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--out_dir', default=OUT)
    p.add_argument('--split_ratio', type=float, default=0.2)
    args = p.parse_args()

    print('[init] CPU-only run — sms_spam Π₀ witness', flush=True)

    print('[data] loading sms_spam', flush=True)
    ds = load_dataset('sms_spam')
    all_data = ds['train']
    rng = np.random.default_rng(0)
    n_total = len(all_data)
    idx = rng.permutation(n_total)
    n_test = int(n_total * args.split_ratio)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    train_text = [all_data[int(i)]['sms'] for i in train_idx]
    train_labels = np.array([all_data[int(i)]['label'] for i in train_idx])
    test_text = [all_data[int(i)]['sms'] for i in test_idx]
    test_labels = np.array([all_data[int(i)]['label'] for i in test_idx])
    n = len(test_text)
    n_classes = int(max(train_labels.max(), test_labels.max())) + 1
    print(f'[data] train={len(train_text)} test={n} classes={n_classes}', flush=True)

    print('[tfidf] vectorizing (uni + bi)', flush=True)
    v_uni = TfidfVectorizer(max_features=20000, ngram_range=(1, 1), min_df=2)
    v_bi  = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)
    Xtr_u = v_uni.fit_transform(train_text); Xte_u = v_uni.transform(test_text)
    Xtr_b = v_bi.fit_transform(train_text);  Xte_b = v_bi.transform(test_text)

    # Direct: SVC bigram (calibrated for probs)
    print('[direct] LinearSVC(bigram, calibrated)', flush=True)
    svc = LinearSVC(C=1.0, max_iter=2000)
    direct_clf = CalibratedClassifierCV(svc, cv=3)
    direct_clf.fit(Xtr_b, train_labels)
    direct_probs = direct_clf.predict_proba(Xte_b)
    direct_pred = direct_probs.argmax(axis=1)
    direct_margin, direct_top_lp = margin_top2(direct_probs)
    direct_acc = float((direct_pred == test_labels).mean())
    print(f'[direct] acc = {direct_acc:.4f}', flush=True)

    # Retrieve: KNN cosine
    print('[retrieve] KNN(k=10, cosine)', flush=True)
    knn = KNeighborsClassifier(n_neighbors=10, n_jobs=-1, metric='cosine')
    knn.fit(Xtr_u, train_labels)
    retrieve_probs = knn.predict_proba(Xte_u)
    retrieve_pred = retrieve_probs.argmax(axis=1)
    retrieve_margin, retrieve_top_lp = margin_top2(retrieve_probs)
    retrieve_acc = float((retrieve_pred == test_labels).mean())
    print(f'[retrieve] acc = {retrieve_acc:.4f}', flush=True)

    # Defer: LogReg unigram (deliberately weaker than direct)
    print('[defer] LogisticRegression(unigram)', flush=True)
    defer_clf = LogisticRegression(max_iter=1000, n_jobs=-1, C=1.0)
    defer_clf.fit(Xtr_u, train_labels)
    defer_probs = defer_clf.predict_proba(Xte_u)
    defer_pred = defer_probs.argmax(axis=1)
    defer_margin, defer_top_lp = margin_top2(defer_probs)
    defer_acc = float((defer_pred == test_labels).mean())
    print(f'[defer] acc = {defer_acc:.4f}', flush=True)

    # C and L
    C = np.zeros((n, 4), dtype=int)
    C[:, A_IDX['answer_direct']] = (direct_pred == test_labels).astype(int)
    C[:, A_IDX['recover_then_answer']] = (retrieve_pred == test_labels).astype(int)
    C[:, A_IDX['escalate_to_stronger_model']] = (defer_pred == test_labels).astype(int)
    C[:, A_IDX['abstain']] = 0

    L = np.zeros((n, 4))
    for a, ai in A_IDX.items():
        L[:, ai] = (1 - C[:, ai]) + 0.05 * COSTS[a]

    # Features (12-d)
    svd = TruncatedSVD(n_components=3, random_state=0)
    svd.fit(Xtr_b)
    pca = svd.transform(Xte_b)
    X = np.column_stack([
        direct_margin, retrieve_margin, defer_margin,
        (direct_pred == retrieve_pred).astype(float),
        (direct_pred == defer_pred).astype(float),
        (retrieve_pred == defer_pred).astype(float),
        direct_top_lp, retrieve_top_lp, defer_top_lp,
        pca[:, 0], pca[:, 1], pca[:, 2],
    ]).astype(np.float64)
    X = np.nan_to_num(X)
    print(f'[features] X.shape = {X.shape}', flush=True)

    fair_const = float(L[:, 0].mean())
    print(f'\nSMS-Spam  n={n}  always_direct loss = {fair_const:.4f}\n', flush=True)

    families = [
        ('always_direct',       v14.AlwaysDirect, 'Pi_0'),
        ('fair_fixed_train',    v14.FairFixedCV,  'Pi_0'),
        ('KM_K04',              lambda: v14.GroupRoutingKMeans(K=4), 'Pi_1'),
        ('KM_K05',              lambda: v14.GroupRoutingKMeans(K=5), 'Pi_1'),
        ('KM_K06',              lambda: v14.GroupRoutingKMeans(K=6), 'Pi_1'),
        ('KM_K08',              lambda: v14.GroupRoutingKMeans(K=8), 'Pi_1'),
        ('HGBC_md3',            lambda: v14.LearnedHGBC(max_depth=3), 'Pi_2'),
        ('HGBC_md4',            lambda: v14.LearnedHGBC(max_depth=4), 'Pi_2'),
        ('selective_C0.3',      lambda: v14.SelectiveCalibrated(C_reg=0.3), 'Pi_2'),
    ]

    seeds = list(range(42, 42 + args.seeds * 31, 31))
    out = {'bench': 'SMS-Spam', 'n': n,
           'always_direct': fair_const,
           'direct_acc': direct_acc, 'retrieve_acc': retrieve_acc,
           'defer_acc': defer_acc,
           'per_family': {}, 'per_class_best': {}}

    print(f'--- Per-family strict CV ({args.seeds} seeds) ---')
    print(f'{"family":<20} {"class":<6} {"mean ± std":<22} {"Δ":>10}')
    print('-' * 66)
    for fam_name, fac, cls in families:
        losses = []
        for s in seeds:
            _, loss = v14.cv_eval_family(fac, X, L, C, n_splits=5, seed=s)
            losses.append(loss)
        m, sd = float(np.mean(losses)), float(np.std(losses))
        out['per_family'][fam_name] = {
            'class': cls, 'mean': m, 'std': sd,
            'delta': m - fair_const, 'losses_per_seed': losses,
        }
        print(f'{fam_name:<20} {cls:<6} {m:.4f} ± {sd:.4f}   Δ={m-fair_const:+.4f}')

    print('\n--- Best per policy class ---')
    for cls in ['Pi_0', 'Pi_1', 'Pi_2']:
        in_class = {k: v for k, v in out['per_family'].items() if v['class'] == cls}
        if not in_class:
            continue
        best = min(in_class, key=lambda k: in_class[k]['mean'])
        out['per_class_best'][cls] = {'family': best, **in_class[best]}
        r = in_class[best]
        print(f'{cls}: best = {best}  loss={r["mean"]:.4f} ± {r["std"]:.4f}  Δ={r["delta"]:+.4f}')

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = f'{args.out_dir}/controller_v14_SMS-Spam.json'
    json.dump(out, open(json_path, 'w'), indent=2)
    np.savez(f'{args.out_dir}/SMS-Spam_loss_features.npz',
             L=L, C=C, X=X, test_labels=test_labels)
    print(f'\nSaved {json_path}')


if __name__ == '__main__':
    main()
