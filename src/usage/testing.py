"""
Testing the beta prior model on upcoming movies.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))  # puts src/ on the path

from beta_prior_model import BetaPriorModel  # noqa: E402


def _plot_beta(model, alpha, beta, title, thresholds):
    s = model.summary(alpha, beta)
    print(s)

    x = np.linspace(0, 1, 500)
    y = beta_dist.pdf(x, alpha, beta)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x * 100, y / 100, color='steelblue', linewidth=2)
    ax.fill_between(x * 100, y / 100, alpha=0.15, color='steelblue')

    lo, hi = s['ci90_low'], s['ci90_high']
    mask = (x * 100 >= lo) & (x * 100 <= hi)
    ax.fill_between(x[mask] * 100, y[mask] / 100, alpha=0.30,
                    color='steelblue', label=f'90% CI  [{lo}–{hi}%]')

    colors = ['tomato', 'darkorange']
    for threshold, color in zip(thresholds, colors):
        p = model.p_above_threshold(alpha, beta, threshold)
        ax.axvline(threshold, color=color, linestyle='--', linewidth=1.4,
                   label=f'P(>{threshold}%) = {p:.1%}')

    ax.axvline(s['mean_pct'], color='steelblue', linestyle='-', linewidth=1.2,
               label=f'Mean = {s["mean_pct"]}%')

    ax.set_xlabel('Tomatometer Score (%)')
    ax.set_ylabel('Density')
    ax.set_title(f'{title}\nκ = {s["kappa"]}  (equivalent to ~{s["kappa"]:.0f} reviews of evidence)')
    ax.set_xlim(0, 100)
    ax.legend(framealpha=0.85)
    plt.tight_layout()
    plt.show()


def plot_prior(model, director, writer, title, thresholds=(60, 80)):
    alpha, beta = model.get_prior(director=director, writer=writer)
    _plot_beta(model, alpha, beta, f'Prior — {title}', thresholds)


def plot_posterior(model, director, writer, k_fresh, n_reviews, title, thresholds=(60, 80)):
    """Plot after k_fresh fresh reviews out of n_reviews total have come in."""
    alpha, beta = model.get_prior(director=director, writer=writer)
    alpha, beta = model.update(alpha, beta, k_fresh=k_fresh, n_reviews=n_reviews)
    _plot_beta(model, alpha, beta,
               f'Posterior — {title}\n({k_fresh} fresh / {n_reviews} reviews in)',
               thresholds)


model = BetaPriorModel.load(str(_HERE / 'beta_prior_model.pkl'))

## Test for "The Odyssey" ------------------------------------------
plot_prior(model,
           director="Christopher Nolan",
           writer="Christopher Nolan",
           title="The Odyssey  (dir. Christopher Nolan)",
           thresholds=(60, 85))

## ------------------------------------------------------------------

## Test for "Power Ballad" ------------------------------------------
plot_posterior(model,
           director="John Carney",
           writer="John Carney",
           k_fresh = 50, n_reviews = 56,
           title="Power Ballad  (dir. John Carney)",
           thresholds=(85, 90))



## ------------------------------------------------------------------