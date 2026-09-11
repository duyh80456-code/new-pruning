from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _finite_mean(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.mean(values[np.isfinite(values)])) if np.isfinite(values).any() else float("nan")


def _format_stat(value: float) -> str:
    return f"{value:.3f}" if np.isfinite(value) else "not estimable"


def write_smoke_report(root_output: Path, seed_summaries: list[dict], combined: pd.DataFrame) -> Path:
    report_dir = root_output / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    systematic = np.asarray([x["systematic_geometry_spearman"] for x in seed_summaries], float)
    pearson = np.asarray(
        [x["sensitivity_accuracy_correlation"]["pearson"]["estimate"] for x in seed_summaries], float
    )
    spearman = np.asarray(
        [x["sensitivity_accuracy_correlation"]["spearman"]["estimate"] for x in seed_summaries], float
    )
    cliff_count = sum(x["compression_cliff_count"] for x in seed_summaries)
    regressions = []
    for summary in seed_summaries:
        regressions.extend(summary["regressions"])
    reg = pd.DataFrame(regressions)
    acc_reg = reg[reg["target"] == "accuracy_degradation"]
    cov = acc_reg.loc[acc_reg["predictors"] == "coverage only", "in_sample_r2"].astype(float).mean()
    combo = acc_reg.loc[
        acc_reg["predictors"] == "coverage + geometry + fit", "in_sample_r2"
    ].astype(float).mean()
    oracle_n = int(combined["oracle_gap_if_available"].notna().sum())
    lines = [
        "# Smoke-test go/no-go report",
        "",
        "> This is an automatic diagnostic, not a claim of statistical significance. "
        "Repeat across the configured seeds and inspect confidence intervals before drawing conclusions.",
        "",
        "## Experimental integrity",
        "",
        "Training used only widths `0.25, 0.50, 0.75, 1.00`. Dense intermediate widths were "
        "used only for post-training BN calibration and evaluation. `ID_fit_proxy` is explicitly a "
        "proxy against the shared full-width teacher representation, not an independently trained reference.",
        "",
        "## Questions",
        "",
        f"1. **Does geometry change systematically with budget?** Mean Spearman correlation between "
        f"distance-to-full-width and width distance: `{_format_stat(_finite_mean(systematic))}`.",
        "",
        f"2. **Are there identifiable compression cliffs?** `{cliff_count}` candidates were flagged "
        "by the configured robust-z rule across all seeds. See each seed's `compression_cliffs.csv`.",
        "",
        f"3. **Does geometric sensitivity correlate with accuracy degradation?** Mean Pearson: "
        f"`{_format_stat(_finite_mean(pearson))}`; mean Spearman: "
        f"`{_format_stat(_finite_mean(spearman))}`. Per-seed bootstrap "
        "intervals are in `analysis_summary.json`.",
        "",
        f"4. **Does geometry add information beyond coverage/FLOPs?** Exploratory in-sample R2 is "
        f"`{_format_stat(cov)}` for coverage alone and `{_format_stat(combo)}` for coverage + geometry + fit. Treat this "
        "as diagnostic because adjacent budgets are not independent samples.",
        "",
        f"5. **Does geometry predict selected-budget oracle gap?** `{oracle_n}` oracle observations are "
        + ("available; inspect `regression_comparison.csv`." if oracle_n else "available. Enable `oracle.enabled` and rerun for this answer."),
        "",
        "## Go/no-go",
        "",
    ]
    finite_corr = _finite_mean(np.abs(spearman))
    if np.isfinite(finite_corr) and finite_corr >= 0.3:
        lines.append(
            "**GO for replication, not yet for geometry regularization.** The smoke result contains a "
            "non-trivial monotonic signal. Confirm it over all seeds and oracle budgets first."
        )
    else:
        lines.append(
            "**NO-GO for geometry regularization at this checkpoint.** The current smoke result does "
            "not show a sufficiently stable sensitivity/performance association. Validate training quality, "
            "feature normalization, and seed consistency before adding method complexity."
        )
    path = report_dir / "smoke_test_report.md"
    path.write_text("\n".join(lines) + "\n")
    return path
