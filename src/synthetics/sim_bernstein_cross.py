"""Theorem 2 cross-threshold synthetic validation.

At fixed (alpha, q, delta), sweeps (n, beta) and reports the empirical
sign-correctness frequency of the bottom-q precision estimator over many
seeds. The Bernstein sufficient condition

    n >= n_min(alpha, beta, q, delta) = 2 alpha (1-alpha) log(2/delta) / (q beta^2)

predicts sign correctness >= 1 - delta whenever n >= n_min. Below that
threshold, the bound does not certify sign correctness. The simulation is
the cleanest available empirical check on Theorem 2's threshold itself
(the synthetic in app:synth-pi12 stays above threshold by construction
and validates Cor 1 (iii) instead).

Outputs:
    figures/synth_bernstein_cross.json   raw rates per (n, beta)
    figures/fig_bernstein_cross.pdf      single-panel plot
"""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
ALPHA = 0.75
Q = 0.3
DELTA = 0.05
LOG_TERM = math.log(2 / DELTA)


def simulate(beta, n_grid, n_seeds=4000, seed_base=0):
    """For each n, draw n_seeds replications of m=floor(nq) Bernoulli(alpha)
    samples and report the fraction with empirical mean above alpha_min."""
    alpha_min = ALPHA - beta
    rng = np.random.default_rng(seed_base)
    rates = []
    for n in n_grid:
        m = max(int(np.floor(n * Q)), 1)
        sums = rng.binomial(m, ALPHA, size=n_seeds)
        mu_hat = sums / m
        rates.append(float((mu_hat > alpha_min).mean()))
    return rates


def main():
    beta_grid = [0.05, 0.10, 0.20]
    n_grid = [20, 30, 50, 80, 130, 200, 320, 500, 800, 1300, 2000, 3200, 5000, 8000]

    results = {}
    for beta in beta_grid:
        n_min = 2 * ALPHA * (1 - ALPHA) * LOG_TERM / (Q * beta ** 2)
        rates = simulate(beta, n_grid)
        results[beta] = {'n_grid': n_grid, 'rates': rates, 'n_min': n_min}

    # Save raw rates for reproducibility.
    with open(f'{OUT}/synth_bernstein_cross.json', 'w') as f:
        json.dump(
            {f'{b}': {'n_grid': r['n_grid'], 'rates': r['rates'],
                      'n_min': r['n_min']}
             for b, r in results.items()},
            f, indent=2,
        )

    plt.rcParams.update({
        'font.size': 9, 'font.family': 'serif',
        'font.serif': ['STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': 9, 'axes.titlesize': 10,
        'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False,
    })

    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    colors = ['#0F4D92', '#9A4D8E', '#B64342']
    markers = ['o', 's', 'D']

    for beta, color, marker in zip(beta_grid, colors, markers):
        r = results[beta]
        ax.plot(r['n_grid'], r['rates'],
                marker=marker, color=color, lw=1.6, markersize=5,
                label=fr'$\beta={beta}$  ($n_{{\min}}{{=}}{r["n_min"]:.0f}$)')
        ax.axvline(r['n_min'], color=color, lw=1, alpha=0.5, ls='--')

    ax.axhline(1 - DELTA, color='black', lw=0.8, ls=':', alpha=0.7,
               label=fr'target $1-\delta={1-DELTA}$')

    ax.set_xscale('log')
    ax.set_xlabel(r'sample size $n$')
    ax.set_ylabel(r'$\Pr[\,\mathrm{sign}(\widehat{\Delta}(q))=\mathrm{sign}(\Delta(q))\,]$')
    ax.set_ylim(0.45, 1.02)
    ax.grid(True, which='both', alpha=0.25, linestyle=':')
    ax.legend(loc='lower right', framealpha=0.92)
    ax.set_title(
        rf'Theorem~2 cross-threshold: $\alpha{{=}}{ALPHA}$, $q{{=}}{Q}$, $\delta{{=}}{DELTA}$',
        pad=8)

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig_bernstein_cross.pdf', bbox_inches='tight')
    plt.close()
    print('wrote synth_bernstein_cross.json and fig_bernstein_cross.pdf')


if __name__ == '__main__':
    main()
