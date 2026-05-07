"""Rich-text Pi_2 ablation for the A-OKVQA Pi_3 oracle story.

The reviewer-facing question is whether Pi_2 can close the gold-rationale
oracle gap if it receives increasingly expressive rationale representations:
a binary exact-match feature, hand-engineered text-overlap statistics, and a
TF-IDF representation of the rationale text.

This script runs Π_2 (and Π_1 for reference) under three increasingly
expressive rationale representations:

  FEATURE SET A — scalar block (39-d) + binary exact_match (1-d) = 40-d
                  (the Round 19 setup; included here for continuity)

  FEATURE SET B — scalar block (39-d) + six hand-engineered text-overlap
                  scalars between direct-answer and rationale
                  (overlap count, overlap ratio, TF-IDF cosine,
                  length ratio, intersect size, plus exact_match) = 45-d

  FEATURE SET C — scalar block (39-d) + TF-IDF of concatenated rationale
                  text (top-200 tokens) = 239-d. This is the "rich text
                  representation" the reviewer asked for: HGBC can learn
                  arbitrary nonlinear combinations of rationale token
                  presence, which strictly subsumes the binary gate.

For each feature set we run KMeans-K=4 (Pi_1 reference) + HGBC-md3 /
HGBC-md4 / SelectiveCalibrated_C0.3 (three Pi_2 learners) under the
identical strict 5-fold × 5-seed CV. The key comparison is:

  best_{feature_set_C} Pi_2  vs.  Pi_3 rationale gate + KM residual

If Π_3 still strictly beats the richest Π_2 representation, the
architectural claim is strengthened. If it doesn't, Section 6.9 has to
honestly reframe the win as "channel + representation" rather than
"channel + architecture".
"""
import argparse, json, os, re, sys
import numpy as np

sys.path.insert(0, './src/controllers')
import controller_arc_p_v11 as v11
from controller_arc_p_v14 import (
    Family, AlwaysDirect, GroupRoutingKMeans, LearnedHGBC,
    SelectiveCalibrated, cv_eval_family,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DST = './data/unified_v14'
TOKEN_RE = re.compile(r'\w+')


def load_pi3_reference():
    """Return the current Pi_3 oracle mean/std produced by pi3_rationale_A-OKVQA."""
    p = f'{DST}/pi3_rationale_A-OKVQA.json'
    d = json.load(open(p))
    r = d['pi3']['pi3_rationale_km']
    return float(r['mean']), float(r['std'])


def tokenize(s):
    return TOKEN_RE.findall((s or '').lower())


def load_aokvqa_texts(rows):
    """Return (direct_answer_text, rationale_text_concat) for each row."""
    samp = json.load(open(
        './data/AOKVQA/samples_val.json'))
    by_sid = {s['sample_id']: s for s in samp}
    at = json.load(open(
        './data/action_tables/A-OKVQA__direct__3B.json'))
    direct_by_sid = {r['sample_id']: (r.get('answer') or '').strip().lower()
                     for r in at['results']}

    direct_texts, rationale_texts = [], []
    for r in rows:
        sid = r['sample_id']
        direct_texts.append(direct_by_sid.get(sid, ''))
        rats = by_sid.get(sid, {}).get('rationales', []) or []
        rationale_texts.append(' '.join(rats).lower())
    return direct_texts, rationale_texts


def build_feature_sets(X_scalar, direct_texts, rationale_texts):
    n = X_scalar.shape[0]
    # ---------- FEATURE SET A: scalar + binary exact_match ----------
    exact_match = np.zeros(n, dtype=float)
    overlap_count = np.zeros(n, dtype=float)
    overlap_ratio = np.zeros(n, dtype=float)
    len_ratio = np.zeros(n, dtype=float)
    intersect_size = np.zeros(n, dtype=float)

    for i in range(n):
        ans_toks = [t for t in tokenize(direct_texts[i]) if len(t) > 2]
        if not ans_toks:
            ans_toks = tokenize(direct_texts[i])
        rat_toks = tokenize(rationale_texts[i])
        rat_set = set(rat_toks)
        ans_set = set(ans_toks)
        inter = ans_set & rat_set
        exact_match[i] = 1.0 if len(inter) > 0 else 0.0
        overlap_count[i] = float(len(inter))
        overlap_ratio[i] = float(len(inter)) / max(len(ans_set), 1)
        len_ratio[i] = len(rat_toks) / max(len(ans_toks), 1)
        intersect_size[i] = float(len(inter))

    # ---------- TF-IDF cosine similarity between direct answer and rationale ----------
    tfidf = TfidfVectorizer(ngram_range=(1, 1), min_df=2, max_df=0.95,
                             stop_words='english')
    tfidf_rat = tfidf.fit_transform(rationale_texts)
    # encode direct answers in the same vocab
    tfidf_ans = tfidf.transform(direct_texts)
    # cosine similarity per row
    cos_sim = np.zeros(n, dtype=float)
    for i in range(n):
        a = tfidf_ans[i]
        r = tfidf_rat[i]
        if a.nnz == 0 or r.nnz == 0:
            cos_sim[i] = 0.0
        else:
            cos_sim[i] = float((a @ r.T).toarray()[0, 0] /
                               (np.sqrt((a.multiply(a)).sum()) *
                                np.sqrt((r.multiply(r)).sum()) + 1e-12))

    set_A = np.concatenate([X_scalar, exact_match.reshape(-1, 1)], axis=1)

    set_B_extras = np.stack([exact_match, overlap_count, overlap_ratio,
                              cos_sim, len_ratio, intersect_size], axis=1)
    set_B = np.concatenate([X_scalar, set_B_extras], axis=1)

    # ---------- FEATURE SET C: scalar + top-200 TF-IDF of rationale ----------
    tfidf_200 = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9,
                                 stop_words='english', max_features=200)
    rat_vec = tfidf_200.fit_transform(rationale_texts).toarray()
    print(f'  rationale TF-IDF vocab size: {rat_vec.shape[1]}')
    set_C = np.concatenate([X_scalar, rat_vec], axis=1)

    return set_A, set_B, set_C


def run_ablation(n_seeds=5):
    bench = 'A-OKVQA'
    rows, Xstd, Xscalar, Xtxt, L, C, R, K, cats, vis, sets = v11.load_bench(bench)
    direct_texts, rationale_texts = load_aokvqa_texts(rows)
    fair_const = float(L[:, 0].mean())
    n = len(rows)

    print(f"\n{'='*80}\n{bench}  n={n}  direct={fair_const:.4f}")
    print(f"canonical scalar block: {Xscalar.shape[1]} dims")
    set_A, set_B, set_C = build_feature_sets(Xscalar, direct_texts, rationale_texts)
    print(f"set A (scalar + exact_match): {set_A.shape[1]} dims")
    print(f"set B (scalar + 6 text stats): {set_B.shape[1]} dims")
    print(f"set C (scalar + TF-IDF-200):   {set_C.shape[1]} dims")
    print('=' * 80)

    families = [
        ('km_K4',          lambda: GroupRoutingKMeans(K=4),        'Pi_1'),
        ('hgbc_md3',       lambda: LearnedHGBC(max_depth=3),       'Pi_2'),
        ('hgbc_md4',       lambda: LearnedHGBC(max_depth=4),       'Pi_2'),
        ('selective_C0.3', lambda: SelectiveCalibrated(C_reg=0.3), 'Pi_2'),
    ]

    seeds = list(range(42, 42 + n_seeds * 31, 31))
    out = {'bench': bench, 'n': n, 'direct': fair_const, 'sets': {}}

    def eval_block(tag, X):
        print(f'\n--- {tag} (dims={X.shape[1]}) ---')
        print(f'{"family":<16} {"class":<6} {"mean ± std":<22}')
        results = {}
        for fam_name, fac, cls in families:
            losses = []
            for s in seeds:
                _, loss = cv_eval_family(fac, X, L, C, n_splits=5, seed=s)
                losses.append(float(loss))
            m = float(np.mean(losses)); sd = float(np.std(losses))
            results[fam_name] = {'class': cls, 'mean': m, 'std': sd,
                                  'losses_per_seed': losses,
                                  'delta': m - fair_const}
            print(f'  {fam_name:<14} {cls:<6} {m:.4f} ± {sd:.4f}  Δ={m-fair_const:+.4f}')
        return results

    out['sets']['canonical'] = eval_block('Canonical (39-d scalar only)', Xscalar)
    out['sets']['A_exact_match'] = eval_block('A: scalar + exact_match (40-d)', set_A)
    out['sets']['B_text_stats']  = eval_block('B: scalar + 6 text overlap stats (45-d)', set_B)
    out['sets']['C_tfidf_rich']  = eval_block('C: scalar + rationale TF-IDF top-200 (~239-d)', set_C)

    # Head-to-head summary
    print(f'\n--- Head-to-head on A-OKVQA ---')
    pi3_ref, pi3_std = load_pi3_reference()
    def best_pi2(results):
        pi2_members = {k: v for k, v in results.items() if v['class'] == 'Pi_2'}
        return min(pi2_members.values(), key=lambda v: v['mean'])
    def best_pi1(results):
        return results['km_K4']

    print(f'{"config":<42} {"Pi_2 best":<22} {"Pi_1 KM_K4":<22}')
    for tag in ['canonical', 'A_exact_match', 'B_text_stats', 'C_tfidf_rich']:
        p2 = best_pi2(out['sets'][tag])
        p1 = best_pi1(out['sets'][tag])
        print(f'{tag:<42} {p2["mean"]:.4f} ± {p2["std"]:.4f} ({p2["class"]:<4})  {p1["mean"]:.4f} ± {p1["std"]:.4f}')
    print(f'{"Pi_3 rationale gate + KM residual":<42} {pi3_ref:.4f} ± {pi3_std:.4f}')

    # Sigma separation Π_3 vs best Π_2 under each set
    for tag in ['A_exact_match', 'B_text_stats', 'C_tfidf_rich']:
        p2 = best_pi2(out['sets'][tag])
        gap = p2['mean'] - pi3_ref
        pool = np.sqrt(p2['std']**2 + pi3_std**2)
        sigma = gap / pool if pool > 0 else 0
        print(f'Π_3 vs best Pi_2 under {tag}: Δ={gap:+.4f}, pooled-σ={sigma:+.2f}σ')

    out['summary'] = {
        'pi3_reference': pi3_ref,
        'pi3_std': pi3_std,
        'best_pi2_per_set': {
            tag: {
                'family': min((k for k in out['sets'][tag] if out['sets'][tag][k]['class']=='Pi_2'),
                              key=lambda k: out['sets'][tag][k]['mean']),
                'mean': best_pi2(out['sets'][tag])['mean'],
                'std':  best_pi2(out['sets'][tag])['std'],
                'pi3_gap': best_pi2(out['sets'][tag])['mean'] - pi3_ref,
            }
            for tag in ['canonical', 'A_exact_match', 'B_text_stats', 'C_tfidf_rich']
        },
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=5)
    args = p.parse_args()
    r = run_ablation(n_seeds=args.seeds)
    os.makedirs(DST, exist_ok=True)
    out_path = f'{DST}/pi2_rich_text_A-OKVQA.json'
    json.dump(r, open(out_path, 'w'), indent=2)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
