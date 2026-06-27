from __future__ import annotations
import argparse
from forecast_core import features, aggregate, output_schema
from forecast_core.bayesian_predict import BayesianForecaster
from forecast_core.config import get_rng, PAID_CHANNELS, HORIZONS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    feats = features.read_features(args.features)
    fc = BayesianForecaster.load(args.model)
    paid = getattr(fc.model, "paid_channels", PAID_CHANNELS)

    by_horizon = {}
    for h in HORIZONS:
        rev, spend, series_meta = fc.predict_from_features(feats, h, budget_plan=None,
                                                           rng=get_rng())
        by_horizon[h] = aggregate.aggregate_levels(rev, spend, series_meta, paid)

    df = output_schema.build_predictions(by_horizon)
    output_schema.write_predictions(df, args.output)
    print(f"[predict] wrote {len(df)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
