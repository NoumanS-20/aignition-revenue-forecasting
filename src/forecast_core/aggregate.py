from __future__ import annotations
import numpy as np


def _accumulate(groups, key, rev, spend):
    r, s = groups.get(key, (None, 0.0))
    r = rev.copy() if r is None else r + rev
    groups[key] = (r, s + spend)


def aggregate_levels(revenue_draws, spend_totals, series_meta, paid_channels):
    """Aggregate per-series revenue draws into level/entity/metric draws.

    Inputs are keyed by a unique series id (e.g. "channel::campaign"); campaign
    names can repeat across channels, so the campaign-level entity is qualified
    as "channel:campaign". ROAS is a ratio of summed revenue to summed spend.
    """
    groups: dict = {}   # (level, entity) -> (revenue_draws, spend_total)
    paid = set(paid_channels)
    for sid, rev in revenue_draws.items():
        m = series_meta[sid]
        spend = float(spend_totals[sid])
        # campaign level (channel-qualified so identical names don't collide)
        _accumulate(groups, ("campaign", f'{m["channel"]}:{m["campaign"]}'), rev, spend)
        # campaign_type level (aggregated across channels)
        _accumulate(groups, ("campaign_type", m["campaign_type"]), rev, spend)
        # channel level
        _accumulate(groups, ("channel", m["channel"]), rev, spend)
        # total (paid only, per documented assumption)
        if m["channel"] in paid:
            _accumulate(groups, ("total", "all"), rev, spend)

    out: dict = {}
    for (level, entity), (rev, spend) in groups.items():
        out[(level, entity, "revenue")] = rev
        out[(level, entity, "roas")] = (rev / spend if spend > 0
                                        else np.full_like(rev, np.nan))
    return out
