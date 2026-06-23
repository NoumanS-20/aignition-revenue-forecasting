from __future__ import annotations
import argparse
import sys
from forecast_core import ingest, validate, features


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    df = ingest.load_data(args.data_dir)
    report = validate.validate_campaigns(df)
    print(f"[validate] rows={report.n_rows} campaigns={report.n_campaigns} "
          f"dates={report.date_min}..{report.date_max} ok={report.ok}")
    for issue in report.issues:
        print(f"[validate] ISSUE: {issue}", file=sys.stderr)
    if not report.ok:
        return 2
    feats = features.build_feature_frame(df)
    features.write_features(feats, args.out)
    print(f"[features] wrote {len(feats)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
