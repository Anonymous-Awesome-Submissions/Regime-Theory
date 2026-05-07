"""Plot synth v4 phase diagram WITHOUT Bernstein curve overlay.

The Bernstein curve from Theorem 2 is satisfied by every cell in the v4
sweep (all beta are in [0.37, 0.62], giving n_min in roughly [12, 34]
which is far below the minimum plotted n of 150). Plotting the curve
on top of the sweep cells is misleading. This script reads the v4 JSON
artifact and produces a clean 2D winner map in (n, beta) space with
no Bernstein overlay; the caption and body text in the paper are
updated accordingly to frame this as Corollary item (iii) validation
rather than Theorem 2 validation.

Style is aligned with the other publication figures: STIX serif, muted
PALETTE blue/red (#0F4D92 / #B64342), strong black cell edges, no
external legend (cells are labeled in-place).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14, 'font.family': 'serif',
                     'font.serif': ['STIXGeneral', 'Liberation Serif',
                                    'DejaVu Serif'],
                     'mathtext.fontset': 'stix',
                     'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})

# Cross-figure color semantic: Π_1 = muted blue (#0F4D92, partition router),
# Π_2 = muted brick red (#B64342, instance-level). Aligned with Table 1 /
# fig_per_class / fig_phase / fig_cluster.
CMAP = {'pi1': '#0F4D92', 'pi2': '#B64342'}
LBL = {'pi1': r'$\Pi_1$', 'pi2': r'$\Pi_2$'}

OUT = './figures'
ART = './data/unified_v14'


def main():
    from matplotlib.colors import ListedColormap

    with open(f'{ART}/synth_regime_phase_v4.json') as f:
        raw = json.load(f)
    results = {}
    for k, v in raw.items():
        n, bk = k.split('_')
        results[(int(n), float(bk))] = v

    ns_sorted  = sorted({n  for (n, _)  in results.keys()})
    bks_sorted = sorted({bk for (_, bk) in results.keys()})

    # 2D winner grid: rows = bk (low at bottom), cols = n (small at left).
    # 0 -> pi1 (blue), 1 -> pi2 (red).
    grid = np.full((len(bks_sorted), len(ns_sorted)), np.nan)
    for (n, bk), r in results.items():
        i = bks_sorted.index(bk)
        j = ns_sorted.index(n)
        grid[i, j] = 1.0 if r['winner'] == 'pi2' else 0.0

    cmap = ListedColormap([CMAP['pi1'], CMAP['pi2']])

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    mesh = ax.pcolormesh(
        np.arange(len(ns_sorted)  + 1),
        np.arange(len(bks_sorted) + 1),
        grid, cmap=cmap, vmin=0, vmax=1,
        edgecolors='black', linewidth=1.5, shading='flat',
    )

    # In-cell winner label (replaces external legend) — Π_1 / Π_2 in white
    for i, bk in enumerate(bks_sorted):
        for j, n in enumerate(ns_sorted):
            winner = 'pi2' if grid[i, j] == 1.0 else 'pi1'
            ax.text(j + 0.5, i + 0.5, LBL[winner],
                    ha='center', va='center',
                    fontsize=14, color='white',
                    weight='bold', zorder=5)

    ax.set_xticks(np.arange(len(ns_sorted))  + 0.5)
    ax.set_xticklabels([str(n) for n in ns_sorted])
    ax.set_yticks(np.arange(len(bks_sorted)) + 0.5)
    ax.set_yticklabels([f'{bk:g}' for bk in bks_sorted])

    ax.set_xlabel(r'sample size $n$', fontsize=15)
    ax.set_ylabel(r'smoothness knob $\mathrm{bk}$', fontsize=15)
    ax.set_title(
        r'Synthetic phase transition: $\Pi_1 \leftrightarrow \Pi_2$ '
        r'in the Bernstein-viable regime',
        fontsize=14, pad=10, loc='left')
    ax.tick_params(axis='both', length=0)
    for s in ('top', 'right', 'bottom', 'left'):
        ax.spines[s].set_visible(False)

    plt.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    out_path = f'{OUT}/fig_phase_synth.pdf'
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
