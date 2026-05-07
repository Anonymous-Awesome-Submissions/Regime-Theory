"""ARC v10 Step 2: build augmented feature matrix with ALL probes.

Consumes:
  - v4_1 base features (20 scalar + 1920 text + 768 image)
  - v9 consistency features (8 stats from K=10 stochastic samples)
  - v10 blind probe (5 modality-bypass features)
  - v10 selfverify probe (4 self-verification features)
  - optional 7B selfverify (4 cross-model features)

Output: unified_v10/{bench}_features.npz
"""
import argparse, json, os
import numpy as np

UNIFIED_V4_1 = "./data/unified_v4_1"
CONSIST_DIR  = "./data/consistency_v9"
DST          = "./data/unified_v10"

CONSIST_NAMES = [
    'consistency_top_count_ratio', 'sample_disagreement', 'semantic_entropy',
    'pred_entropy_t1', 'mean_sample_margin', 'sample_margin_std',
    'mean_sample_seq_logprob', 'sample_seq_logprob_std',
]
BLIND_NAMES = [
    'blind_disagrees', 'blind_margin', 'blind_seq_lp',
    'blind_yes_minus_no', 'blind_p1',
]
SV3_NAMES = [
    'sv3_p_yes', 'sv3_p_no', 'sv3_yes_minus_no', 'sv3_text_is_yes',
]
SV7_NAMES = [
    'sv7_p_yes', 'sv7_p_no', 'sv7_yes_minus_no', 'sv7_text_is_yes',
]


def _load(path):
    if not os.path.exists(path): return None
    return json.load(open(path))


def build_one(bench, K=10):
    feat_npz = np.load(os.path.join(UNIFIED_V4_1, f'{bench}_features.npz'), allow_pickle=True)
    Xfull = feat_npz['X']
    sample_ids = list(feat_npz['sample_ids'])
    feat_meta = json.load(open(os.path.join(UNIFIED_V4_1, f'{bench}_feature_names.json')))
    n = len(sample_ids)

    consist = _load(os.path.join(CONSIST_DIR, f'{bench}__3B__K{K}.json'))
    blind   = _load(os.path.join(CONSIST_DIR, f'{bench}__blind__3B.json'))
    sv3     = _load(os.path.join(CONSIST_DIR, f'{bench}__selfverify__3B.json'))
    sv7     = _load(os.path.join(CONSIST_DIR, f'{bench}__selfverify__7B.json'))

    extra_blocks = []
    extra_names_total = []

    if consist:
        E = np.zeros((n, len(CONSIST_NAMES)))
        m = 0
        for i, sid in enumerate(sample_ids):
            c = consist.get(sid)
            if c is None: continue
            m += 1
            E[i, 0] = c.get('self_consistency', 0.0)
            E[i, 1] = c.get('sample_disagreement', 0.0)
            E[i, 2] = c.get('semantic_entropy', 0.0)
            E[i, 3] = c.get('pred_entropy_t1', 0.0)
            E[i, 4] = c.get('mean_margin', 0.0)
            E[i, 5] = c.get('margin_std', 0.0)
            E[i, 6] = c.get('mean_seq_logprob', 0.0)
            E[i, 7] = c.get('seq_logprob_std', 0.0)
        print(f'  consistency matched {m}/{n}')
        extra_blocks.append(E); extra_names_total += CONSIST_NAMES
    else:
        print('  consistency: missing')

    if blind:
        E = np.zeros((n, len(BLIND_NAMES)))
        m = 0
        for i, sid in enumerate(sample_ids):
            b = blind.get(sid)
            if b is None: continue
            m += 1
            E[i, 0] = b.get('blind_disagrees', 0.0)
            E[i, 1] = b.get('blind_margin', 0.0)
            E[i, 2] = b.get('blind_seq_lp', 0.0)
            E[i, 3] = b.get('blind_yes_minus_no', 0.0)
            E[i, 4] = b.get('blind_p1', 0.0)
        print(f'  blind matched {m}/{n}')
        extra_blocks.append(E); extra_names_total += BLIND_NAMES
    else:
        print('  blind: missing')

    if sv3:
        E = np.zeros((n, len(SV3_NAMES)))
        m = 0
        for i, sid in enumerate(sample_ids):
            s = sv3.get(sid)
            if s is None: continue
            m += 1
            E[i, 0] = s.get('verify_p_yes', 0.0)
            E[i, 1] = s.get('verify_p_no', 0.0)
            E[i, 2] = s.get('verify_margin_yes_minus_no', 0.0)
            E[i, 3] = 1.0 if 'yes' in (s.get('verify_text', '') or '').lower() else 0.0
        print(f'  sv3 matched {m}/{n}')
        extra_blocks.append(E); extra_names_total += SV3_NAMES
    else:
        print('  sv3: missing')

    if sv7:
        E = np.zeros((n, len(SV7_NAMES)))
        m = 0
        for i, sid in enumerate(sample_ids):
            s = sv7.get(sid)
            if s is None: continue
            m += 1
            E[i, 0] = s.get('verify_p_yes', 0.0)
            E[i, 1] = s.get('verify_p_no', 0.0)
            E[i, 2] = s.get('verify_margin_yes_minus_no', 0.0)
            E[i, 3] = 1.0 if 'yes' in (s.get('verify_text', '') or '').lower() else 0.0
        print(f'  sv7 matched {m}/{n}')
        extra_blocks.append(E); extra_names_total += SV7_NAMES
    else:
        print('  sv7: missing')

    if not extra_blocks:
        print('  [skip] no probes available'); return None

    n_text = feat_meta.get('n_text_features', Xfull.shape[1])
    scalar_dim = n_text - 1920
    pre = Xfull[:, :scalar_dim]
    mid = Xfull[:, scalar_dim:n_text]
    img = Xfull[:, n_text:]

    extra = np.concatenate(extra_blocks, axis=1)
    Xnew = np.concatenate([pre, extra, mid, img], axis=1)
    feat_names_new = (feat_meta['feature_names'][:scalar_dim] + extra_names_total
                      + feat_meta['feature_names'][scalar_dim:])
    new_meta = dict(feat_meta)
    new_meta['feature_names'] = feat_names_new
    new_meta['n_features'] = int(Xnew.shape[1])
    new_meta['n_text_features'] = int(n_text + extra.shape[1])
    new_meta['n_scalar_features'] = int(scalar_dim + extra.shape[1])
    new_meta['n_consistency_features'] = extra.shape[1]
    new_meta['probe_blocks_used'] = extra_names_total

    os.makedirs(DST, exist_ok=True)
    np.savez(os.path.join(DST, f'{bench}_features.npz'),
             X=Xnew, sample_ids=np.array(sample_ids, dtype=object))
    json.dump(new_meta, open(os.path.join(DST, f'{bench}_feature_names.json'), 'w'),
              indent=2)
    print(f'  wrote {DST}/{bench}_features.npz  shape={Xnew.shape}  scalar_dim={new_meta["n_scalar_features"]}')
    return Xnew.shape


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bench', default=None)
    args = p.parse_args()
    benches = [args.bench] if args.bench else ['POPE', 'HallusionBench', 'A-OKVQA', 'FEVER']
    for b in benches:
        print(f'==> {b}')
        build_one(b)


if __name__ == '__main__':
    main()
