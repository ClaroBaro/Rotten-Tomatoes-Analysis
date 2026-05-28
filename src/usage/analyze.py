"""
Usage
-----
Pre-release prior (no reviews yet):
    python analyze.py "Christopher Nolan" "Christopher Nolan"

After reviews drop (cumulative totals):
    python analyze.py "John Carney" "John Carney" --fresh 50 --total 56

Custom Kalshi thresholds:
    python analyze.py "Christopher Nolan" "Christopher Nolan" --thresholds 75 80 85

With market prices for edge comparison:
    python analyze.py "Christopher Nolan" "Christopher Nolan" --market 75:91 80:80 85:64 90:34
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))
from beta_prior_model import BetaPriorModel  # noqa: E402


def plot(model, alpha, beta, title, thresholds):
    s = model.summary(alpha, beta)

    x = np.linspace(0, 1, 500)
    y = beta_dist.pdf(x, alpha, beta)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x * 100, y / 100, color='steelblue', linewidth=2)
    ax.fill_between(x * 100, y / 100, alpha=0.15, color='steelblue')

    lo, hi = s['ci90_low'], s['ci90_high']
    mask = (x * 100 >= lo) & (x * 100 <= hi)
    ax.fill_between(x[mask] * 100, y[mask] / 100, alpha=0.30,
                    color='steelblue', label=f'90% CI  [{lo}–{hi}%]')

    colors = ['tomato', 'darkorange', 'mediumseagreen', 'mediumpurple']
    for threshold, color in zip(thresholds, colors):
        p = model.p_above_threshold(alpha, beta, threshold)
        ax.axvline(threshold, color=color, linestyle='--', linewidth=1.4,
                   label=f'P(>{threshold}%) = {p:.1%}')

    ax.axvline(s['mean_pct'], color='steelblue', linestyle='-', linewidth=1.2,
               label=f'Mean = {s["mean_pct"]}%')

    ax.set_xlabel('Tomatometer Score (%)')
    ax.set_ylabel('Density')
    ax.set_title(f'{title}\nκ = {s["kappa"]}')
    ax.set_xlim(0, 100)
    ax.legend(framealpha=0.85)
    plt.tight_layout()
    plt.show()


def print_edge_table(model, alpha, beta, thresholds, market_prices):
    print(f"\n{'Threshold':<12} {'Model P':>9} {'Market P':>10} {'Gap':>8}  {'No cost':>8}  {'EV(No)':>8}  Signal")
    print('-' * 72)
    for t in thresholds:
        p_model = model.p_above_threshold(alpha, beta, t) * 100
        mkt = market_prices.get(t)
        if mkt is None:
            print(f'>{t:<11} {p_model:>9.1f}%   {"-":>9}   {"-":>7}   {"-":>7}   {"-":>7}')
            continue
        gap = p_model - mkt
        no_cost = 100 - mkt
        p_no = (100 - p_model) / 100
        ev_no = p_no * (100 - no_cost) / 100 - (1 - p_no) * no_cost / 100
        signal = 'BUY No' if ev_no > 0.05 else ('BUY Yes' if gap > 10 else 'pass')
        print(f'>{t:<11} {p_model:>9.1f}%  {mkt:>9}c  {gap:>+8.1f}  {no_cost:>7}c  {ev_no:>+7.2f}  {signal}')
    print()


def main():
    parser = argparse.ArgumentParser(description='Beta prior model for RT score prediction')
    parser.add_argument('director', help='Director name(s), comma-separated')
    parser.add_argument('writer',   help='Writer name(s), comma-separated')
    parser.add_argument('--fresh',      type=int, default=None, help='Fresh reviews so far (cumulative)')
    parser.add_argument('--total',      type=int, default=None, help='Total reviews so far (cumulative)')
    parser.add_argument('--thresholds', type=int, nargs='+', default=[60, 75, 80, 85],
                        help='Kalshi thresholds to evaluate (default: 60 75 80 85)')
    parser.add_argument('--market',     nargs='+', default=[],
                        help='Market Yes prices as threshold:price pairs, e.g. 80:79 85:64 90:34')
    args = parser.parse_args()

    model = BetaPriorModel.load(str(_HERE / 'beta_prior_model.pkl'))
    alpha, beta = model.get_prior(director=args.director, writer=args.writer)

    if args.fresh is not None and args.total is not None:
        alpha, beta = model.update(alpha, beta, k_fresh=args.fresh, n_reviews=args.total)
        review_tag = f'{args.fresh} fresh / {args.total} reviews in'
        stage = 'Posterior'
    else:
        review_tag = 'pre-release'
        stage = 'Prior'

    title = f'{stage} - {args.director}  ({review_tag})'
    print(f'\n{title}')
    print(model.summary(alpha, beta))

    market_prices = {}
    for item in args.market:
        t, p = item.split(':')
        market_prices[int(t)] = int(p)

    if market_prices:
        print_edge_table(model, alpha, beta, args.thresholds, market_prices)

    plot(model, alpha, beta, title, args.thresholds)


if __name__ == '__main__':
    main()
