from __future__ import annotations
import argparse
import os
import collections
import numpy as np
import pandas as pd
from forecast_core import ingest, features
from forecast_core.response_curves import fit_hill
from forecast_core.bayesian_predict import CompiledModel, BayesianForecaster
from forecast_core.config import get_rng, SEED, PAID_CHANNELS


def _finite(x: float, default: float) -> float:
    x = float(x)
    return x if np.isfinite(x) else float(default)


def _group_point_estimates(frame: pd.DataFrame):
    """Estimate relative seasonality, Hill shape, and noise from a group's
    campaigns, normalizing each campaign by its own mean so they pool/transfer."""
    pairs_x, pairs_y = [], []          # normalized (spend, revenue) for Hill shape
    dow_vals = {d: [] for d in range(7)}
    logr = []
    for _, gc in frame.groupby("campaign"):
        d = gc.groupby("date").agg(rev=("revenue", "sum"), spend=("spend", "sum"))
        mr = float(d["rev"].mean())
        if not np.isfinite(mr) or mr <= 0:
            continue
        yn = (d["rev"] / mr).to_numpy()
        for dd, yy in zip(d.index.dayofweek.to_numpy(), yn):
            dow_vals[int(dd)].append(float(yy))
        logr.extend(np.log(np.clip(yn, 1e-3, None)).tolist())
        ms = float(d["spend"].mean())
        if np.isfinite(ms) and ms > 0:
            xn = (d["spend"] / ms).to_numpy()
            pairs_x.extend(xn.tolist())
            pairs_y.extend(yn.tolist())

    seas = np.array([(np.mean(dow_vals[d]) if dow_vals[d] else 1.0) for d in range(7)])
    seas = seas / (seas.mean() if seas.mean() > 0 else 1.0)

    kappa_rel, slope = 2.0, 1.0
    if len(pairs_x) >= 8:
        hp = fit_hill(np.array(pairs_x), np.clip(np.array(pairs_y), 0, None), get_rng(SEED))
        kappa_rel = min(max(_finite(hp["kappa"], 2.0), 0.1), 20.0)
        slope = min(max(_finite(hp["slope"], 1.0), 0.3), 3.0)

    sigma = _finite(np.std(logr), 0.2) if len(logr) > 2 else 0.2
    sigma = min(max(sigma, 0.05), 1.0)
    return seas, kappa_rel, slope, sigma


def _draws_from_estimates(seas, kappa_rel, slope, sigma, n_draws, rng) -> dict:
    return {
        "seasonal_mult": seas[None, :] * np.exp(rng.normal(0, 0.05, (n_draws, 7))),
        "kappa_rel": np.clip(kappa_rel * np.exp(rng.normal(0, 0.20, n_draws)), 0.05, None),
        "slope": np.clip(slope + rng.normal(0, 0.10, n_draws), 0.2, 4.0),
        "sigma_log": np.full(n_draws, sigma),
    }


def _build_groups(feats, n_draws, rng, draws_fn):
    groups, channel_groups = {}, {}
    for (ch, ct), g in feats.groupby(["channel", "campaign_type"]):
        groups[(ch, ct)] = draws_fn(g, n_draws, rng)
    for ch, g in feats.groupby("channel"):
        channel_groups[ch] = draws_fn(g, n_draws, rng)
    global_group = draws_fn(feats, n_draws, rng)
    last_date = str(pd.to_datetime(feats["date"]).max().date())
    return CompiledModel(groups=groups, channel_groups=channel_groups,
                         global_group=global_group, n_draws=n_draws,
                         last_date=last_date, paid_channels=PAID_CHANNELS)


def fit_fallback(feats: pd.DataFrame, n_draws: int, rng) -> CompiledModel:
    def draws_fn(g, nd, rng_):
        seas, kappa_rel, slope, sigma = _group_point_estimates(g)
        return _draws_from_estimates(seas, kappa_rel, slope, sigma, nd, rng_)
    return _build_groups(feats, n_draws, rng, draws_fn)


def fit_bayesian(feats: pd.DataFrame, draws: int = 1000, tune: int = 1000) -> CompiledModel:
    """Bayesian day-of-week seasonality + noise per group (requires PyMC; runs on
    Linux/Colab). The saturation shape is fit deterministically, since response
    curves are weakly identified and full MMM is out of scope."""
    import pymc as pm  # imported lazily so the scored path never needs PyMC
    n_draws = draws * 2  # 2 chains

    def draws_fn(g, nd, rng_):
        seas_pt, kappa_rel, slope, sigma_pt = _group_point_estimates(g)
        # Build normalized daily revenue with day-of-week index across campaigns.
        ys, dows = [], []
        for _, gc in g.groupby("campaign"):
            d = gc.groupby("date").agg(rev=("revenue", "sum"))
            mr = float(d["rev"].mean())
            if not np.isfinite(mr) or mr <= 0:
                continue
            ys.extend((d["rev"] / mr).clip(lower=1e-3).tolist())
            dows.extend(d.index.dayofweek.tolist())
        if len(ys) < 14:
            return _draws_from_estimates(seas_pt, kappa_rel, slope, sigma_pt, nd, rng_)
        ys = np.asarray(ys); dows = np.asarray(dows)
        with pm.Model():
            dow = pm.Normal("dow", mu=0.0, sigma=0.3, shape=7)
            sigma = pm.HalfNormal("sigma", sigma=0.5)
            mu = dow[dows]
            pm.Lognormal("obs", mu=mu, sigma=sigma, observed=ys)
            idata = pm.sample(draws=draws, tune=tune, chains=2, cores=1,
                              random_seed=SEED, progressbar=False)
        post = idata.posterior
        dow_draws = np.exp(post["dow"].to_numpy().reshape(-1, 7))      # (nd,7) multipliers
        dow_draws = dow_draws / dow_draws.mean(axis=1, keepdims=True)
        sig_draws = post["sigma"].to_numpy().reshape(-1)
        nd2 = dow_draws.shape[0]
        return {
            "seasonal_mult": dow_draws,
            "kappa_rel": np.clip(kappa_rel * np.exp(rng_.normal(0, 0.2, nd2)), 0.05, None),
            "slope": np.clip(slope + rng_.normal(0, 0.1, nd2), 0.2, 4.0),
            "sigma_log": sig_draws,
        }
    return _build_groups(feats, n_draws, get_rng(SEED), draws_fn)


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
    print(f"[train] {args.method} model: {len(model.groups)} (channel,type) groups, "
          f"{len(model.channel_groups)} channels -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
