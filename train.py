from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
from forecast_core import ingest, features
from forecast_core.response_curves import fit_hill
from forecast_core.bayesian_predict import CompiledModel, BayesianForecaster
from forecast_core.config import get_rng, SEED


def _series_table(feats: pd.DataFrame):
    return feats.groupby(["channel", "campaign_type", "campaign"], sort=True)


def fit_fallback(feats: pd.DataFrame, n_draws: int, rng) -> CompiledModel:
    series = []
    last_date = str(feats["date"].max().date())
    for (channel, ctype, camp), g in _series_table(feats):
        daily_rev = g.groupby("date")["revenue"].sum()
        daily_spend = g.groupby("date")["spend"].sum()
        base = float(daily_rev.mean())
        base_sd = float(daily_rev.std() or base * 0.1)
        # seasonal day-of-week deviations from mean
        dow_mean = g.groupby(g["date"].dt.dayofweek)["revenue"].mean()
        seasonal = np.array([float(dow_mean.get(d, base)) - base for d in range(7)])
        hp = fit_hill(daily_spend.to_numpy(),
                      np.clip(daily_rev.to_numpy() - base, 0, None), rng)
        series.append({
            "channel": channel, "campaign_type": ctype, "campaign": camp,
            "baseline_draws": rng.normal(base, max(base_sd, 1.0), n_draws),
            "seasonal_dow": seasonal[None, :] + rng.normal(
                0, abs(base_sd) * 0.1 + 1e-6, (n_draws, 7)),
            "hill": {"alpha": rng.normal(hp["alpha"], hp["alpha"] * 0.1 + 1e-6, n_draws),
                     "kappa": np.full(n_draws, hp["kappa"]),
                     "slope": np.full(n_draws, hp["slope"])},
            "sigma_log": np.full(n_draws, 0.1),
            "recent_spend": float(daily_spend.tail(14).mean() or 0.0),
        })
    return CompiledModel(series=series, n_draws=n_draws, last_date=last_date,
                         calibration={})


def fit_bayesian(feats: pd.DataFrame, draws: int = 1000, tune: int = 1000) -> CompiledModel:
    """Full hierarchical fit. Requires requirements-train.txt (PyMC)."""
    import pymc as pm  # imported lazily so the scored path never needs PyMC

    series = []
    last_date = str(feats["date"].max().date())
    n_draws = draws * 2  # 2 chains
    for (channel, ctype, camp), g in _series_table(feats):
        d = g.groupby("date").agg(revenue=("revenue", "sum"),
                                  spend=("spend", "sum")).reset_index()
        y = d["revenue"].to_numpy()
        spend = d["spend"].to_numpy()
        dow = d["date"].dt.dayofweek.to_numpy()
        with pm.Model():
            base = pm.Normal("base", mu=float(y.mean()), sigma=float(y.std() + 1.0))
            s_dow = pm.Normal("s_dow", 0.0, sigma=float(y.std() + 1.0), shape=7)
            alpha = pm.HalfNormal("alpha", sigma=float(max(y.max(), 1.0)))
            kappa = pm.HalfNormal("kappa", sigma=float(
                max(np.median(spend[spend > 0]) if (spend > 0).any() else 1.0, 1.0)))
            slope = pm.TruncatedNormal("slope", mu=1.0, sigma=0.5, lower=0.1, upper=5.0)
            sigma = pm.HalfNormal("sigma", sigma=0.5)
            incr = alpha * spend**slope / (kappa**slope + spend**slope + 1e-9)
            mu = pm.math.log(pm.math.clip(base + s_dow[dow] + incr, 1e-6, np.inf))
            pm.Lognormal("obs", mu=mu, sigma=sigma, observed=np.clip(y, 1e-6, None))
            idata = pm.sample(draws=draws, tune=tune, chains=2, cores=1,
                              random_seed=SEED, progressbar=False)
        post = idata.posterior

        def flat(name):
            return post[name].to_numpy().reshape(-1, *post[name].shape[2:])

        n_draws = flat("base").shape[0]
        series.append({
            "channel": channel, "campaign_type": ctype, "campaign": camp,
            "baseline_draws": flat("base"),
            "seasonal_dow": flat("s_dow"),
            "hill": {"alpha": flat("alpha"), "kappa": flat("kappa"),
                     "slope": flat("slope")},
            "sigma_log": flat("sigma"),
            "recent_spend": float(d["spend"].tail(14).mean() or 0.0),
        })
    return CompiledModel(series=series, n_draws=n_draws, last_date=last_date,
                         calibration={})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/sample")
    ap.add_argument("--out", default="pickle/model.pkl")
    ap.add_argument("--method", choices=["bayesian", "fallback"], default="bayesian")
    ap.add_argument("--draws", type=int, default=1000)
    args = ap.parse_args(argv)
    feats = features.build_feature_frame(ingest.load_data(args.data_dir))
    if args.method == "bayesian":
        model = fit_bayesian(feats, draws=args.draws)
    else:
        model = fit_fallback(feats, n_draws=2000, rng=get_rng(SEED))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    BayesianForecaster(model).save(args.out)
    print(f"[train] {args.method} model with {len(model.series)} series -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
