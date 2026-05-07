"""End-to-end FOLIO pilot: action_table -> loss_matrix -> features -> controller.

Risk model: rule-based, h=0.5 if wrong, h=0 if correct (skip InternVL judge for pilot).
Feature block: per-action confidence margin + log-prob, agreements, label one-hot,
question length. ~12-d simple block.

Outputs go to repro_pkg_2026/data/{unified,unified_v4,unified_v10,unified_v14}/FOLIO_*.
Then runs controller_arc_p_v14 --bench FOLIO and reports Pi_0/Pi_1/Pi_2 winners +
theory ceilings (C_Pi1, C_Pi2).
"""
import json, os, sys, math
import numpy as np
sys.path.insert(0, './src/controllers')

DATA = './data'
ACT = f'{DATA}/action_tables'
UNI = f'{DATA}/unified'
UV4 = f'{DATA}/unified_v4'
UV10 = f'{DATA}/unified_v10'
UV14 = f'{DATA}/unified_v14'
for d in [UNI, UV4, UV10, UV14]: os.makedirs(d, exist_ok=True)

W_C, W_H, W_K = 1.0, 1.0, 0.05
COSTS = {'answer_direct': 1.0, 'recover_then_answer': 2.0,
         'escalate_to_stronger_model': 2.3, 'abstain': 0.0}


def loss_with_rule_risk(correct, action):
    """canonical core loss; rule-based h=0.5 if wrong else 0; abstain has h=0, c=0"""
    if action == 'abstain':
        return W_C * 1 + W_H * 0 + W_K * 0  # = 1.0
    h = 0.0 if correct else 0.5
    cost = COSTS[action]
    return W_C * (1 - correct) + W_H * h + W_K * cost


def main():
    # 1) Load 3 action outputs
    direct3 = {r['sample_id']: r for r in json.load(open(f'{ACT}/FOLIO__direct__3B.json'))['results']}
    recov3  = {r['sample_id']: r for r in json.load(open(f'{ACT}/FOLIO__recover__3B.json'))['results']}
    direct7 = {r['sample_id']: r for r in json.load(open(f'{ACT}/FOLIO__escalate__7B.json'))['results']}

    sids = sorted(set(direct3) & set(recov3) & set(direct7))
    print(f'[FOLIO] joined n = {len(sids)}')

    # 2) Build action_table
    rows = []
    for sid in sids:
        d3, r3, d7 = direct3[sid], recov3[sid], direct7[sid]
        rows.append({
            'sample_id': sid, 'benchmark': 'FOLIO',
            'category': d3.get('category'), 'gt': d3.get('gt'),
            'answer_direct': {
                'answer': d3['answer'], 'correct': int(d3['correct']),
                'confidence_margin': d3['confidence_margin'],
                'seq_logprob': d3['seq_logprob'],
                'cost_passes': 1, 'cost_tokens': d3.get('cost_tokens', 0),
            },
            'recover_then_answer': {
                'answer': r3['answer'], 'correct': int(r3['correct']),
                'confidence_margin': r3['confidence_margin'],
                'seq_logprob': r3['seq_logprob'],
                'cost_passes': 2, 'cost_tokens': r3.get('cost_tokens', 0),
                'ctx_margin': r3.get('ctx_margin', 0.0),
                'recover_context': (r3.get('recover_context') or '')[:200],
            },
            'escalate_to_stronger_model': {
                'answer': d7['answer'], 'correct': int(d7['correct']),
                'confidence_margin': d7['confidence_margin'],
                'seq_logprob': d7['seq_logprob'],
                'cost_passes': 1, 'cost_tokens': d7.get('cost_tokens', 0),
            },
            'abstain': {
                'answer': None, 'correct': 0, 'confidence_margin': 0.0,
                'seq_logprob': 0.0, 'cost_passes': 0, 'cost_tokens': 0,
            },
        })
    json.dump(rows, open(f'{UNI}/FOLIO_action_table.json', 'w'), indent=2)
    print(f'[FOLIO] wrote {UNI}/FOLIO_action_table.json')

    # 3) Build loss matrix (rule-based risk)
    lm_rows = []
    for r in rows:
        actions = {}
        for act in ['answer_direct', 'recover_then_answer', 'escalate_to_stronger_model', 'abstain']:
            sub = r[act]
            correct = int(sub.get('correct', 0))
            ll = loss_with_rule_risk(correct, act)
            actions[act] = {
                'answer': sub.get('answer'),
                'correct': correct,
                'risk': 0.0 if (act == 'abstain' or correct) else 0.5,
                'risk_label': 'grounded' if (act == 'abstain' or correct) else 'unverifiable',
                'prop_type': 'unknown',
                'cost': COSTS[act],
                'loss': ll,
                'reward': -ll,
                'confidence_margin': sub.get('confidence_margin', 0.0),
                'seq_logprob': sub.get('seq_logprob', 0.0),
            }
        lm_rows.append({
            'sample_id': r['sample_id'], 'benchmark': 'FOLIO',
            'category': r.get('category'), 'gt': r.get('gt'),
            'actions': actions,
        })
    LM = {'bench': 'FOLIO', 'n': len(lm_rows), 'alpha': 1.0, 'beta': 1.0, 'lambda': W_K, 'rows': lm_rows}
    json.dump(LM, open(f'{UV4}/FOLIO_loss_matrix.json', 'w'), indent=2)
    print(f'[FOLIO] wrote {UV4}/FOLIO_loss_matrix.json')

    # 4) Build simple 12-d feature block
    feat_names = []
    cols = []
    def push(name, col):
        feat_names.append(name); cols.append(np.asarray(col, dtype=float))
    push('direct_margin',        [r['answer_direct']['confidence_margin'] for r in rows])
    push('direct_seq_lp',        [r['answer_direct']['seq_logprob']       for r in rows])
    push('recover_margin',       [r['recover_then_answer']['confidence_margin'] for r in rows])
    push('recover_seq_lp',       [r['recover_then_answer']['seq_logprob']       for r in rows])
    push('recover_ctx_margin',   [r['recover_then_answer'].get('ctx_margin', 0.0) for r in rows])
    push('escalate_margin',      [r['escalate_to_stronger_model']['confidence_margin'] for r in rows])
    push('escalate_seq_lp',      [r['escalate_to_stronger_model']['seq_logprob']       for r in rows])
    def agree(a, b):
        return [1.0 if (r[a].get('answer') or '').strip().lower() ==
                       (r[b].get('answer') or '').strip().lower() else 0.0 for r in rows]
    push('agree_dr', agree('answer_direct', 'recover_then_answer'))
    push('agree_de', agree('answer_direct', 'escalate_to_stronger_model'))
    push('agree_re', agree('recover_then_answer', 'escalate_to_stronger_model'))
    # FOLIO 3-class label one-hots (using DIRECT prediction, not gt — features must be deployable)
    pred_d = [(r['answer_direct'].get('answer') or '').strip().lower() for r in rows]
    push('pred_true',      [1.0 if a == 'true' else 0.0 for a in pred_d])
    push('pred_uncertain', [1.0 if a == 'uncertain' else 0.0 for a in pred_d])

    X = np.stack(cols, axis=1).astype(np.float32)
    sids_arr = np.array([r['sample_id'] for r in rows], dtype=object)
    np.savez(f'{UV10}/FOLIO_features.npz', X=X, sample_ids=sids_arr)
    json.dump({
        'feature_names': feat_names,
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'n_text_features': X.shape[1],
        'n_image_features': 0,
        'n_scalar_features': X.shape[1],
        'n_consistency_features': 0,
        'probe_blocks_used': [],
    }, open(f'{UV10}/FOLIO_feature_names.json', 'w'), indent=2)
    print(f'[FOLIO] wrote {UV10}/FOLIO_features.npz, X.shape={X.shape}')

    # 5) Theoretical ceilings (C_Pi1 = max_K sum p_g gamma_g, C_Pi2 = q* beta (L_w-L_r))
    L = np.zeros((len(rows), 4))
    ACTS = ['answer_direct', 'recover_then_answer', 'escalate_to_stronger_model', 'abstain']
    for i, r in enumerate(lm_rows):
        for j, a in enumerate(ACTS):
            L[i, j] = r['actions'][a]['loss']
    # alpha, beta from direct correctness
    direct_correct = np.array([r['answer_direct']['correct'] for r in rows])
    L_d = L[:, 0]
    L_r = float(L_d[direct_correct == 1].mean())
    L_w = float(L_d[direct_correct == 0].mean())
    L_a = float(L[:, 3].mean())
    alpha_min = max(0.0, min(1.0, (L_a - L_r)/(L_w - L_r))) if L_w != L_r else 1.0
    # alpha_emp via 5-fold LR on standardized features
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    from sklearn.metrics import roc_auc_score
    Xs = np.nan_to_num(X)
    aucs = []
    for C_reg in [0.01, 0.05, 0.1, 0.3, 1.0]:
        preds = np.zeros(len(L))
        for tr, te in KFold(5, shuffle=True, random_state=42).split(Xs):
            sc = StandardScaler().fit(Xs[tr])
            clf = LogisticRegression(C=C_reg, max_iter=1000, random_state=42).fit(
                sc.transform(Xs[tr]), direct_correct[tr])
            preds[te] = clf.predict_proba(sc.transform(Xs[te]))[:, 1]
        aucs.append(roc_auc_score(direct_correct, preds))
    alpha_emp = max(aucs)
    beta = alpha_emp - alpha_min
    C_Pi2 = max(0.0, 0.3 * beta * (L_w - L_r))
    n_min_Bern = math.ceil(2 * alpha_emp * (1 - alpha_emp) * math.log(40) / (0.3 * beta * beta)) if beta > 0 else float('inf')

    # C_Pi1 via KMeans K=2..8, seed=42
    from sklearn.cluster import KMeans
    Xstd = StandardScaler().fit_transform(Xs)
    gba = int(np.argmin(L.mean(axis=0)))
    best_pi1 = 0.0; best_K = 0; best_anatomy = None
    for K_choice in [2, 3, 4, 5, 6, 8]:
        if K_choice >= len(L): continue
        labels = KMeans(n_clusters=K_choice, n_init=10, random_state=42).fit_predict(Xstd)
        s = 0.0
        anatomy = []
        for g in range(K_choice):
            m = labels == g
            if m.sum() == 0: continue
            mu = L[m].mean(axis=0); cb = int(np.argmin(mu))
            gam = 0.0 if cb == gba else float(mu[gba] - mu[cb])
            p_g = float(m.mean()); pg = p_g * gam
            anatomy.append(dict(g=g, n=int(m.sum()), p_g=p_g, gamma=gam, p_g_gamma=pg, cell_best=ACTS[cb]))
            s += pg
        if s > best_pi1:
            best_pi1 = s; best_K = K_choice; best_anatomy = anatomy
    C_Pi1 = best_pi1
    predicted = 'Pi_2' if C_Pi2 >= C_Pi1 else 'Pi_1'
    print()
    print('=' * 60)
    print('FOLIO theory ceilings (n =', len(L), ')')
    print('=' * 60)
    print(f'  L_r (direct correct) = {L_r:.4f}')
    print(f'  L_w (direct wrong)   = {L_w:.4f}')
    print(f'  L_a (abstain)        = {L_a:.4f}')
    print(f'  alpha_min            = {alpha_min:.4f}')
    print(f'  alpha_emp (best CV-LR AUC) = {alpha_emp:.4f}')
    print(f'  beta = alpha - alpha_min  = {beta:.4f}')
    print(f'  n_min^Bern (q=0.3, delta=0.05) = {n_min_Bern}')
    print(f'  C_Pi2 = q* beta (L_w-L_r) = 0.3 x {beta:.3f} x {L_w-L_r:.3f} = {C_Pi2:.4f}')
    print(f'  C_Pi1 = max sum p_g gamma_g = {C_Pi1:.4f}  (best K = {best_K})')
    if best_anatomy:
        print(f'  Best Pi_1 anatomy (K={best_K}):')
        for a in best_anatomy:
            print(f"    cluster {a['g']}: n={a['n']}, p_g={a['p_g']:.3f}, "
                  f"gamma={a['gamma']:.3f}, p*gamma={a['p_g_gamma']:.3f}, "
                  f"cell_best={a['cell_best']}")
    print()
    print(f'  *** PREDICTED WINNER: {predicted} ***')
    print(f'  margin C_Pi2 - C_Pi1 = {C_Pi2 - C_Pi1:+.4f}')

    # save theory summary
    json.dump(dict(n=len(L), alpha=alpha_emp, alpha_min=alpha_min, beta=beta,
                   L_r=L_r, L_w=L_w, L_a=L_a, n_min_Bern=n_min_Bern,
                   C_Pi1=C_Pi1, C_Pi1_best_K=best_K, C_Pi2=C_Pi2,
                   predicted_winner=predicted, anatomy=best_anatomy),
              open(f'{UV14}/folio_theory_ceiling.json', 'w'), indent=2)
    print(f'  saved theory ceilings to {UV14}/folio_theory_ceiling.json')


if __name__ == '__main__':
    main()
