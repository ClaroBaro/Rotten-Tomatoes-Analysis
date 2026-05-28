"""
Goal: using a movie's metadata to calculate a prior probability distribution for the
post-release rotten tomato score.
Features that are included:
    - director
    - writer
    - [side implementation] anticipation/forum sentiment

"""

import pickle

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class BetaPriorModel:
    """
    Pre-release prior for a movie's tomatoMeter expressed as Beta(α, β).

    Workflow
    --------
    1. fit(movies_df)               — learn from historical data
    2. get_prior(director, writer)  → (α, β)
    3. update(α, β, k_fresh, n)     → (α_post, β_post)  as each review batch arrives
    4. p_above_threshold(α, β, T)   → float              Kalshi trading signal

    Parameterisation
    ----------------
    Beta(α, β) is parameterised internally as (μ, κ):
      μ = α / (α + β)    prior mean fresh rate  [0, 1]
      κ = α + β          concentration — "phantom review count"
      α = μ * κ,  β = (1 − μ) * κ

    μ₀ comes from a linear regression on director/writer LOO averages.
    κ grows with combined filmography size: κ = kappa_base + kappa_scale * n_films.
    Both kappa_base and kappa_scale are calibrated by minimising Brier score on a
    held-out validation split during fit().

    Cold-start (debut director/writer with no history): prior mean falls back to the
    global population mean (~0.57), shrunk smoothly via shrink_k.

    Parameters
    ----------
    shrink_k    : int   Shrinkage toward global mean for sparse filmographies.
                        A person with shrink_k films gets their history weighted
                        equally with the global mean; fewer films → more shrinkage.
    kappa_base  : float Minimum κ for a debut filmmaker (calibrated during fit).
    kappa_scale : float κ increment per additional film (calibrated during fit).
    kappa_cap   : int   Cap on filmography count used to compute κ, so a prolific
                        director doesn't make the prior immovable.
    """

    _KAPPA_BASE_GRID  = [1, 2, 3, 5]
    _KAPPA_SCALE_GRID = [0.25, 0.5, 1.0, 1.5, 2.0]

    def __init__(
        self,
        shrink_k: int    = 3,
        kappa_base: float  = 2.0,
        kappa_scale: float = 1.0,
        kappa_cap: int   = 20,
    ):
        self.shrink_k    = shrink_k
        self.kappa_base  = kappa_base
        self.kappa_scale = kappa_scale
        self.kappa_cap   = kappa_cap

        self._global_mean: float | None          = None
        self._dir_stats:   pd.DataFrame | None   = None
        self._wri_stats:   pd.DataFrame | None   = None
        self._mu_model:    LinearRegression | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, movies_df: pd.DataFrame, random_state: int = 42) -> "BetaPriorModel":
        """
        Learn director/writer LOO averages, fit the μ regression, and
        calibrate κ hyperparameters via Brier score on a held-out split.

        Parameters
        ----------
        movies_df   DataFrame with columns: id, tomatoMeter, director, writer
        """
        df = movies_df.dropna(subset=["tomatoMeter"]).copy()
        self._global_mean = df["tomatoMeter"].mean() / 100.0

        self._dir_stats, dir_loo = self._compute_loo_avgs(df, "director")
        self._wri_stats, wri_loo = self._compute_loo_avgs(df, "writer")

        feat = df[["id", "tomatoMeter", "director", "writer"]].copy()
        feat["dir_loo"] = feat["id"].map(dir_loo).astype("float64") / 100.0
        feat["wri_loo"] = (feat["id"].map(wri_loo).astype("float64") / 100.0).fillna(self._global_mean)
        feat = feat.dropna(subset=["dir_loo"])

        # precompute per-movie filmography counts for kappa calibration
        feat["n_dir"] = feat["director"].apply(
            lambda d: self._lookup_person(d, self._dir_stats)[1]
        )
        feat["n_wri"] = feat["writer"].apply(
            lambda w: self._lookup_person(w, self._wri_stats)[1]
        )
        feat["n_films"] = (feat["n_dir"] + feat["n_wri"]).clip(upper=self.kappa_cap)

        X       = feat[["dir_loo", "wri_loo"]].values
        y       = feat["tomatoMeter"].values / 100.0
        n_films = feat["n_films"].values

        idx_tr, idx_val = train_test_split(
            np.arange(len(feat)), test_size=0.2, random_state=random_state
        )

        self._mu_model = LinearRegression().fit(X[idx_tr], y[idx_tr])
        self._calibrate_kappa(X[idx_val], y[idx_val], n_films[idx_val])

        return self

    def get_prior(
        self,
        director: str | None,
        writer:   str | None,
    ) -> tuple[float, float]:
        """
        Return (α, β) for a not-yet-released movie.

        Parameters
        ----------
        director    comma-separated director name(s), or None
        writer      comma-separated writer name(s), or None
        """
        self._check_fitted()
        assert self._dir_stats is not None and self._wri_stats is not None
        assert self._mu_model is not None
        mu    = self._predict_mu(director, writer)
        kappa = self._estimate_kappa(director, writer)
        return mu * kappa, (1.0 - mu) * kappa

    @staticmethod
    def update(
        alpha: float,
        beta:  float,
        k_fresh:   int,
        n_reviews: int,
    ) -> tuple[float, float]:
        """
        Exact closed-form Bayesian update.
        Pass cumulative k_fresh and n_reviews (not just the new batch).

        Example
        -------
        alpha, beta = model.get_prior("Ari Aster", "Ari Aster")
        # 3 reviews in: 2 fresh, 1 rotten
        alpha, beta = model.update(alpha, beta, k_fresh=2, n_reviews=3)
        """
        return alpha + k_fresh, beta + (n_reviews - k_fresh)

    @staticmethod
    def p_above_threshold(alpha: float, beta: float, threshold_pct: float) -> float:
        """
        P(final tomatoMeter > threshold_pct) under Beta(α, β).
        threshold_pct is on a 0–100 scale (e.g. 60 for Certified Fresh).
        """
        return 1.0 - beta_dist.cdf(threshold_pct / 100.0, alpha, beta)

    def summary(self, alpha: float, beta: float) -> dict:
        """Mean, mode, std, and 90 % credible interval of Beta(α, β), all in pct."""
        mean = alpha / (alpha + beta)
        mode = (alpha - 1) / (alpha + beta - 2) if alpha > 1 and beta > 1 else mean
        std  = np.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)))
        lo, hi = beta_dist.ppf([0.05, 0.95], alpha, beta)
        return {
            "mean_pct":  round(float(mean) * 100, 1),
            "mode_pct":  round(float(mode) * 100, 1),
            "std_pct":   round(float(std)  * 100, 1),
            "ci90_low":  round(float(lo)   * 100, 1),
            "ci90_high": round(float(hi)   * 100, 1),
            "kappa":     round(float(alpha + beta), 2),
        }

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BetaPriorModel":
        with open(path, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_loo_avgs(
        self,
        df: pd.DataFrame,
        person_col: str,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        For each movie, compute the leave-one-out average tomatoMeter for each
        person listed in person_col (comma-separated). Where a person has only
        one film, LOO is undefined → NaN (will shrink to global mean at inference).

        Returns
        -------
        stats       DataFrame indexed by person name, cols [total, count]
        loo_by_movie  Series indexed by movie id
        Ported from notebook cell 33.
        """
        exploded = df[["id", "tomatoMeter", person_col]].copy()
        exploded[person_col] = exploded[person_col].str.split(",")
        exploded = exploded.explode(person_col)
        exploded[person_col] = exploded[person_col].str.strip()
        exploded = exploded.dropna(subset=[person_col])
        exploded = exploded[exploded[person_col] != ""]

        stats = exploded.groupby(person_col).agg(
            total=("tomatoMeter", "sum"),
            count=("tomatoMeter", "count"),
        )

        exploded = exploded.join(stats, on=person_col)

        exploded["loo_avg"] = np.where(
            exploded["count"] > 1,
            (exploded["total"] - exploded["tomatoMeter"]) / (exploded["count"] - 1),
            np.nan,
        )

        loo_by_movie = exploded.groupby("id")["loo_avg"].mean()
        return stats, loo_by_movie

    def _lookup_person(
        self,
        names_str: str | None,
        stats: pd.DataFrame,
    ) -> tuple[float | None, int]:
        """
        For a comma-separated name string, return (career_avg_0_100, n_films).
        Uses the full career average (not LOO) since we're predicting a *new* film.
        """
        if not names_str or pd.isna(names_str):
            return None, 0

        names = [n.strip() for n in str(names_str).split(",") if n.strip()]
        avgs, counts = [], []
        for name in names:
            if name in stats.index:
                total = float(np.asarray(stats.at[name, "total"]).item())
                count = int(np.asarray(stats.at[name, "count"]).item())
                avgs.append(total / count)
                counts.append(count)

        raw_avg = float(np.mean(avgs)) if avgs else None
        n_films  = int(np.mean(counts)) if counts else 0
        return raw_avg, n_films

    def _shrink(self, raw_avg_100: float | None, n_films: int) -> float:
        """Blend career average (0–100 scale) toward global mean (0–1 scale)."""
        gm = self._global_mean
        if raw_avg_100 is None:
            return gm
        raw_01 = raw_avg_100 / 100.0
        return (raw_01 * n_films + gm * self.shrink_k) / (n_films + self.shrink_k)

    def _predict_mu(self, director: str | None, writer: str | None) -> float:
        dir_raw, dir_n = self._lookup_person(director, self._dir_stats)
        wri_raw, wri_n = self._lookup_person(writer,   self._wri_stats)

        dir_mu = self._shrink(dir_raw, dir_n)
        wri_mu = self._shrink(wri_raw, wri_n)

        mu = float(self._mu_model.predict([[dir_mu, wri_mu]])[0])
        return float(np.clip(mu, 0.01, 0.99))

    def _estimate_kappa(self, director: str | None, writer: str | None) -> float:
        _, dir_n = self._lookup_person(director, self._dir_stats)
        _, wri_n = self._lookup_person(writer,   self._wri_stats)
        n_films  = min(dir_n + wri_n, self.kappa_cap)
        return self.kappa_base + self.kappa_scale * n_films

    def _calibrate_kappa(
        self,
        X_val:       np.ndarray,
        y_val:       np.ndarray,
        n_films_val: np.ndarray,
    ) -> None:
        """
        Grid search over (κ_base, κ_scale) minimising Brier score at the 50%
        threshold on the validation set. Fully vectorised — no row iteration.
        """
        mu_vals = np.clip(self._mu_model.predict(X_val), 0.01, 0.99)
        actual  = (y_val > 0.5).astype(float)
        best_brier = np.inf

        for kb in self._KAPPA_BASE_GRID:
            for ks in self._KAPPA_SCALE_GRID:
                kappa   = kb + ks * n_films_val
                alpha_v = mu_vals * kappa
                beta_v  = (1.0 - mu_vals) * kappa
                p_above = 1.0 - beta_dist.cdf(0.5, alpha_v, beta_v)
                brier   = float(np.mean((p_above - actual) ** 2))
                if brier < best_brier:
                    best_brier       = brier
                    self.kappa_base  = float(kb)
                    self.kappa_scale = float(ks)

    def _check_fitted(self) -> None:
        if self._mu_model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
