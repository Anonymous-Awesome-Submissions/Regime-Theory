"""Custom builder for unified_v10 features.npz, used for the rerun pipeline.

Replaces build_features_v10.py (whose v4_1 input dir is empty) with a
fresh path that goes action_table + v9 probes -> 41-d scalar block.

Per-bench differences:
  HallusionBench: full 41-d (cat_VD/VS + cos_q_context with image context)
  FEVER:          39-d (no category; cos_q_context = cos_q_recover_ctx text-only)
                  (blind probe absent for FEVER text-only)

Output:
  unified_v10/{bench}_features.npz   (X, sample_ids)
  unified_v10/{bench}_feature_names.json (n_scalar_features, names)
"""
import argparse, json, os, sys
import numpy as np
from sentence_transformers import SentenceTransformer

UN  = "./data"
DST = f"{UN}/unified_v10"
ACT = f"{UN}/unified"
CONSIST_DIR = f"{UN}/consistency_v9"

EMB_MODEL = 'all-MiniLM-L6-v2'

CONSIST_NAMES = [
    'consistency_top_count_ratio', 'sample_disagreement', 'semantic_entropy',
    'pred_entropy_t1', 'mean_sample_margin', 'sample_margin_std',
    'mean_sample_seq_logprob', 'sample_seq_logprob_std',
]
BLIND_NAMES = ['blind_disagrees', 'blind_margin', 'blind_seq_lp', 'blind_yes_minus_no', 'blind_p1']
SV3_NAMES = ['sv3_p_yes', 'sv3_p_no', 'sv3_yes_minus_no', 'sv3_text_is_yes']
SV7_NAMES = ['sv7_p_yes', 'sv7_p_no', 'sv7_yes_minus_no', 'sv7_text_is_yes']


def cos(a, b):
    if not a.shape or not b.shape:
        return 0.0
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def build(bench):
    rows = json.load(open(f'{ACT}/{bench}_action_table.json'))
    n = len(rows)
    print(f'[{bench}] action_table n={n}')

    # encode question + answers + context
    print(f'[{bench}] encoding sentences with {EMB_MODEL}...')
    enc = SentenceTransformer(EMB_MODEL)
    qs = [r.get('question') or '' for r in rows]
    if not qs[0]:
        # action_table doesn't store question; load from candidates
        cands = json.load(open(f'{UN}/GRACE/{bench}/candidates.json'))
        cmap = {c['sample_id']: c.get('question', '') for c in cands}
        qs = [cmap.get(r['sample_id'], '') for r in rows]
    a_d = [(r['answer_direct'].get('answer') or '') for r in rows]
    a_r = [(r['recover_then_answer'].get('answer') or '') for r in rows]
    a_e = [(r['escalate_to_stronger_model'].get('answer') or '') for r in rows]
    ctx = [(r['recover_then_answer'].get('recover_context') or '') for r in rows]
    Eq  = enc.encode(qs, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    Ed  = enc.encode(a_d, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    Er  = enc.encode(a_r, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    Ee  = enc.encode(a_e, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    Ec  = enc.encode(ctx, batch_size=64, show_progress_bar=False, convert_to_numpy=True)

    # ----- base 20-d scalar block (Hallu) / 18-d (FEVER no cat one-hot) -----
    base_names = []
    base_cols  = []
    def push(name, col):
        base_names.append(name); base_cols.append(col)

    md = np.array([r['answer_direct']['confidence_margin'] for r in rows]); push('direct_margin', md)
    push('direct_seq_lp', np.array([r['answer_direct']['seq_logprob'] for r in rows]))
    push('recover_margin', np.array([r['recover_then_answer']['confidence_margin'] for r in rows]))
    push('recover_seq_lp', np.array([r['recover_then_answer']['seq_logprob'] for r in rows]))
    push('recover_ctx_margin', np.array([r['recover_then_answer'].get('ctx_margin', 0.0) for r in rows]))
    push('escalate_margin', np.array([r['escalate_to_stronger_model']['confidence_margin'] for r in rows]))
    push('escalate_seq_lp', np.array([r['escalate_to_stronger_model']['seq_logprob'] for r in rows]))
    push('agree_dr', np.array([1.0 if (r['answer_direct'].get('answer') or '').strip().lower()
                               == (r['recover_then_answer'].get('answer') or '').strip().lower() else 0.0
                               for r in rows]))
    push('agree_de', np.array([1.0 if (r['answer_direct'].get('answer') or '').strip().lower()
                               == (r['escalate_to_stronger_model'].get('answer') or '').strip().lower() else 0.0
                               for r in rows]))
    push('agree_re', np.array([1.0 if (r['recover_then_answer'].get('answer') or '').strip().lower()
                               == (r['escalate_to_stronger_model'].get('answer') or '').strip().lower() else 0.0
                               for r in rows]))
    push('question_nwords', np.array([float(len((q or '').split())) for q in qs]))
    if bench == 'HallusionBench':
        cats = [r.get('category') or '' for r in rows]
        push('cat_VD', np.array([1.0 if c == 'VD' else 0.0 for c in cats]))
        push('cat_VS', np.array([1.0 if c == 'VS' else 0.0 for c in cats]))
    elif bench == 'FEVER':
        # 3 label-class indicators serve as 'category' equivalent: 0=S, 1=R, 2=NEI. Use gt label.
        gts = [int(r.get('gt') or 0) if str(r.get('gt') or '0').isdigit() else 0 for r in rows]
        push('cls_supports', np.array([1.0 if g == 0 else 0.0 for g in gts]))
        push('cls_refutes',  np.array([1.0 if g == 1 else 0.0 for g in gts]))
    push('cos_q_direct',   np.array([cos(Eq[i], Ed[i]) for i in range(n)]))
    push('cos_q_recover',  np.array([cos(Eq[i], Er[i]) for i in range(n)]))
    push('cos_q_escalate', np.array([cos(Eq[i], Ee[i]) for i in range(n)]))
    push('cos_direct_recover',  np.array([cos(Ed[i], Er[i]) for i in range(n)]))
    push('cos_direct_escalate', np.array([cos(Ed[i], Ee[i]) for i in range(n)]))
    push('cos_recover_escalate', np.array([cos(Er[i], Ee[i]) for i in range(n)]))
    push('cos_q_context',  np.array([cos(Eq[i], Ec[i]) for i in range(n)]))

    base = np.stack(base_cols, axis=1)
    print(f'[{bench}] base block shape={base.shape}')

    # ----- v9 probes -----
    sids = [r['sample_id'] for r in rows]
    extras = []
    extra_names = []
    for fname, names in [
        (f'{CONSIST_DIR}/{bench}__3B__K10.json', CONSIST_NAMES),
        (f'{CONSIST_DIR}/{bench}__blind__3B.json', BLIND_NAMES),
        (f'{CONSIST_DIR}/{bench}__selfverify__3B.json', SV3_NAMES),
        (f'{CONSIST_DIR}/{bench}__selfverify__7B.json', SV7_NAMES),
    ]:
        p = _load(fname)
        if p is None:
            print(f'[{bench}] {os.path.basename(fname)}: missing -> filling zeros')
            extras.append(np.zeros((n, len(names))))
            extra_names += names
            continue
        E = np.zeros((n, len(names)))
        m = 0
        for i, sid in enumerate(sids):
            v = p.get(sid)
            if v is None:
                continue
            m += 1
            for j, name in enumerate(names):
                # remap consistency names
                key_map = {
                    'consistency_top_count_ratio': 'self_consistency',
                    'mean_sample_margin': 'mean_margin',
                    'sample_margin_std': 'margin_std',
                    'mean_sample_seq_logprob': 'mean_seq_logprob',
                    'sample_seq_logprob_std': 'seq_logprob_std',
                }
                src_key = key_map.get(name, name)
                if name == 'sv3_text_is_yes' or name == 'sv7_text_is_yes':
                    E[i, j] = 1.0 if 'yes' in (v.get('verify_text', '') or '').lower() else 0.0
                elif name in ('sv3_p_yes', 'sv7_p_yes'):
                    E[i, j] = float(v.get('verify_p_yes', 0.0))
                elif name in ('sv3_p_no', 'sv7_p_no'):
                    E[i, j] = float(v.get('verify_p_no', 0.0))
                elif name in ('sv3_yes_minus_no', 'sv7_yes_minus_no'):
                    E[i, j] = float(v.get('verify_margin_yes_minus_no', 0.0))
                else:
                    E[i, j] = float(v.get(src_key, v.get(name, 0.0)))
        print(f'[{bench}] {os.path.basename(fname)}: matched {m}/{n}')
        extras.append(E)
        extra_names += names

    extra_block = np.concatenate(extras, axis=1)
    X = np.concatenate([base, extra_block], axis=1).astype(np.float32)
    feat_names = base_names + extra_names
    print(f'[{bench}] final X shape={X.shape}, scalar dim={X.shape[1]}')

    os.makedirs(DST, exist_ok=True)
    np.savez(f'{DST}/{bench}_features.npz', X=X, sample_ids=np.array(sids, dtype=object))
    meta = {
        'feature_names': feat_names,
        'n_features': X.shape[1],
        'n_samples': n,
        'n_text_features': X.shape[1],   # no separate text/image embeddings
        'n_image_features': 0,
        'n_scalar_features': X.shape[1],
        'n_consistency_features': extra_block.shape[1],
        'probe_blocks_used': extra_names,
    }
    json.dump(meta, open(f'{DST}/{bench}_feature_names.json', 'w'), indent=2)
    print(f'[{bench}] wrote {DST}/{bench}_features.npz')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bench', required=True, choices=['HallusionBench', 'FEVER'])
    args = p.parse_args()
    build(args.bench)


if __name__ == '__main__':
    main()
