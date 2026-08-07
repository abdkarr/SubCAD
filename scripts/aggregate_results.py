"""
Aggregate the results produced by experiment scripts into the format to be used
for reporting. Aggregations for the following experiments are implemented, as
one subcommand each: `planted-attacks` (`planted_attacks.py`) and
`planted-attacks-with-experts` (`planted_attacks_with_experts.py`). Both write
the same four CSVs (Dataset x Method, one row per dataset, "<method>"/
"<method>_ci" columns, unless noted otherwise) to `research/reports/<experiment>/`:

- label-fusion.csv        — Accuracy, for every baseline/competitor/proposed method
- adversary-detection.csv — WorkerROCAUC, for the competitors and the best proposed method
- target-detection.csv    — TaskROCAUC, for the best proposed method only
- significance.csv        — per dataset/metric, whether the best method is
  significantly better than the second-best (Wilcoxon signed-rank test)
"""

import tomllib

from pathlib import Path

import click
import numpy as np
import pandas as pd
from scipy import stats

import subcad

INPUT_DIR = Path(subcad.PROJECT_DIR, "research", "data", "inputs")
OUTPUT_DIR = Path(subcad.PROJECT_DIR, "research", "data", "outputs")
REPORTS_DIR = Path(subcad.PROJECT_DIR, "research", "reports")


def _mean_ci(values: pd.Series, confidence: float) -> pd.Series:
    """Mean and half-width of a t-distribution confidence interval."""
    values = values.dropna().to_numpy()
    if len(values) == 0:
        return pd.Series({"mean": np.nan, "ci": np.nan})
    if len(values) == 1:
        return pd.Series({"mean": values[0], "ci": 0.0})

    mean = np.mean(values)
    half_width = stats.sem(values) * stats.t.ppf((1 + confidence) / 2, len(values) - 1)
    return pd.Series({"mean": mean, "ci": half_width})


def _summarize(
    results: pd.DataFrame,
    metric: str,
    datasets: list[str],
    methods: list[str],
    confidence: float,
) -> pd.DataFrame:
    """Pivot one metric into a Dataset x Method table of mean/CI columns."""
    if metric not in results.columns:
        return pd.DataFrame({"Dataset": datasets})

    summary = (
        results.groupby(["Dataset", "MethodKey"])[metric]
        .apply(_mean_ci, confidence=confidence)
        .unstack(-1)  # "mean"/"ci" -> columns
        .unstack("MethodKey")
    )
    summary.columns = [
        method if stat == "mean" else f"{method}_ci" for stat, method in summary.columns
    ]

    ordered_cols = [
        col for method in methods for col in (method, f"{method}_ci")
    ]
    summary = summary.reindex(index=datasets, columns=ordered_cols)
    summary.index.name = "Dataset"

    # Drop methods for which this metric was never produced (all-NaN columns).
    summary = summary.dropna(axis=1, how="all")

    return summary.reset_index()


def _best_vs_second_best(
    results: pd.DataFrame, metric: str, datasets: list[str], methods: list[str]
) -> pd.DataFrame:
    """Per dataset, test whether the best method significantly outperforms
    the second-best method (by mean `metric`), restricted to `methods`.

    Uses the paired Wilcoxon signed-rank test: seeds are matched across
    methods (the same seed generates the same corrupted dataset regardless of
    method), and performance metrics like accuracy/AUROC are bounded and not
    reliably normal, so a non-parametric paired test is preferred over a
    paired t-test (Demsar, 2006).
    """
    if metric not in results.columns:
        return pd.DataFrame()

    rows = []
    for dataset in datasets:
        subset = results[
            (results["Dataset"] == dataset) & (results["MethodKey"].isin(methods))
        ]
        means = (
            subset.groupby("MethodKey")[metric]
            .mean()
            .dropna()
            .sort_values(ascending=False)
        )
        if len(means) < 2:
            continue
        best_method, second_method = means.index[0], means.index[1]

        paired = pd.concat(
            [
                subset[subset["MethodKey"] == best_method].set_index("Seed")[metric],
                subset[subset["MethodKey"] == second_method].set_index("Seed")[metric],
            ],
            axis=1,
            keys=["best", "second"],
        ).dropna()

        if len(paired) < 2 or np.allclose(paired["best"], paired["second"]):
            p_value = np.nan
        else:
            _, p_value = stats.wilcoxon(paired["best"], paired["second"])

        rows.append(
            {
                "Dataset": dataset,
                "Metric": metric,
                "BestMethod": best_method,
                "BestMean": means.loc[best_method],
                "SecondBestMethod": second_method,
                "SecondBestMean": means.loc[second_method],
                "NPairs": len(paired),
                "PValue": p_value,
                "Significant": bool(p_value < 0.05) if pd.notna(p_value) else False,
            }
        )

    return pd.DataFrame(rows)


@click.group()
def cli():
    pass


@cli.command("planted-attacks")
@click.option("--config", default="configs/planted_attacks.toml", show_default=True)
def planted_attacks(config):
    """Aggregate results produced by `planted_attacks.py`."""
    exp_name = "planted_attacks"
    confidence = 0.95

    cfg_path = Path(config)
    if not cfg_path.is_absolute():
        cfg_path = Path(subcad.PROJECT_DIR) / cfg_path
    cfg = tomllib.loads(cfg_path.read_text())
    data_cfg = cfg.get("dataset", {})

    datasets = ["rte", "sp", "web", "dog", "temp", "adult2"]
    baselines = ["glad", "ebcc", "la-onepass", "la-twopass", "mv", "ds"]
    competitors = ["dacs", "repalg-soft", "repalg-hard", "mmsr"]
    proposed = ["subcad-wgsmv", "subcad-wgsds"]
    methods = baselines + competitors + proposed

    # Read the results
    rows = []
    for dataset in datasets:
        adv_frac = data_cfg[dataset]["adv_frac"]
        target_frac = data_cfg[dataset]["target_frac"]
        target_obs = data_cfg[dataset]["target_obs"]
        camo_reliability = data_cfg[dataset]["camo_reliability"]
        for method in methods:
            results_dir = Path(
                OUTPUT_DIR, exp_name, f"{dataset}",
                f"af={adv_frac}_tf={target_frac}_to={target_obs}_cr={camo_reliability}",
                f"{method}"
            )
            for seed_file in sorted(results_dir.glob("seed-*.csv")):
                df = pd.read_csv(seed_file)
                df["Dataset"] = dataset
                df["MethodKey"] = method
                df["Seed"] = int(seed_file.stem.split("-")[1])
                rows.append(df)
    results = pd.concat(rows, ignore_index=True)

    out_dir = Path(REPORTS_DIR, exp_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate label fusion methods
    label_fusion_summary = _summarize(
        results, "Accuracy", datasets, methods, confidence
    )
    label_fusion_summary.to_csv(
        Path(out_dir, "label-fusion.csv"), index=False, float_format="%.4f"
    )

    # Aggregate adversary detection methods
    adversary_detectors = competitors + [proposed[-1]]
    adversary_detection_summary = _summarize(
        results, "WorkerROCAUC", datasets, adversary_detectors, confidence
    )
    adversary_detection_summary.to_csv(
        Path(out_dir, "adversary-detection.csv"), index=False, float_format="%.4f"
    )

    # Aggregate task detection methods
    target_detectors = [proposed[-1]]
    target_detection_summary = _summarize(
        results, "TaskROCAUC", datasets, target_detectors, confidence
    )
    target_detection_summary.to_csv(
        Path(out_dir, "target-detection.csv"), index=False, float_format="%.4f"
    )

    # Significance testing
    significance = pd.concat([
        _best_vs_second_best(results, "Accuracy", datasets, methods),
        _best_vs_second_best(results, "WorkerROCAUC", datasets, adversary_detectors),
    ], ignore_index=True)
    significance.to_csv(
        Path(out_dir, "significance.csv"), index=False, float_format="%.4f"
    )

    print(f"Wrote summaries to {out_dir}")


@cli.command("planted-attacks-with-experts")
@click.option(
    "--config", default="configs/planted_attacks_with_experts.toml", show_default=True
)
def planted_attacks_with_experts(config):
    """Aggregate results produced by `planted_attacks_with_experts.py`."""
    exp_name = "planted_attacks_with_experts"
    confidence = 0.95

    cfg_path = Path(config)
    if not cfg_path.is_absolute():
        cfg_path = Path(subcad.PROJECT_DIR) / cfg_path
    cfg = tomllib.loads(cfg_path.read_text())
    data_cfg = cfg.get("dataset", {})

    datasets = ["rte", "sp", "web", "dog", "temp", "adult2"]
    baselines = ["glad", "ebcc", "la-onepass", "la-twopass", "mv", "ds"]
    competitors = ["dacs", "repalg-soft", "repalg-hard", "mmsr"]
    proposed = ["subcad-wgsmv", "subcad-wgsds"]
    methods = baselines + competitors + proposed

    # Read the results
    rows = []
    for dataset in datasets:
        adv_frac = data_cfg[dataset]["adv_frac"]
        target_frac = data_cfg[dataset]["target_frac"]
        target_obs = data_cfg[dataset]["target_obs"]
        camo_reliability = data_cfg[dataset]["camo_reliability"]
        expert_frac = data_cfg[dataset]["expert_frac"]
        expert_reliability = data_cfg[dataset]["expert_reliability"]
        for method in methods:
            results_dir = Path(
                OUTPUT_DIR, exp_name, f"{dataset}",
                f"af={adv_frac}_tf={target_frac}_to={target_obs}_cr={camo_reliability}_"
                f"ef={expert_frac}_er={expert_reliability}",
                f"{method}"
            )
            for seed_file in sorted(results_dir.glob("seed-*.csv")):
                df = pd.read_csv(seed_file)
                df["Dataset"] = dataset
                df["MethodKey"] = method
                df["Seed"] = int(seed_file.stem.split("-")[1])
                rows.append(df)
    results = pd.concat(rows, ignore_index=True)

    out_dir = Path(REPORTS_DIR, exp_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate label fusion methods
    label_fusion_summary = _summarize(
        results, "Accuracy", datasets, methods, confidence
    )
    label_fusion_summary.to_csv(
        Path(out_dir, "label-fusion.csv"), index=False, float_format="%.4f"
    )

    # Aggregate adversary detection methods
    adversary_detectors = competitors + [proposed[-1]]
    adversary_detection_summary = _summarize(
        results, "WorkerROCAUC", datasets, adversary_detectors, confidence
    )
    adversary_detection_summary.to_csv(
        Path(out_dir, "adversary-detection.csv"), index=False, float_format="%.4f"
    )

    # Aggregate task detection methods
    target_detectors = [proposed[-1]]
    target_detection_summary = _summarize(
        results, "TaskROCAUC", datasets, target_detectors, confidence
    )
    target_detection_summary.to_csv(
        Path(out_dir, "target-detection.csv"), index=False, float_format="%.4f"
    )

    # Significance testing
    significance = pd.concat([
        _best_vs_second_best(results, "Accuracy", datasets, methods),
        _best_vs_second_best(results, "WorkerROCAUC", datasets, adversary_detectors),
    ], ignore_index=True)
    significance.to_csv(
        Path(out_dir, "significance.csv"), index=False, float_format="%.4f"
    )

    print(f"Wrote summaries to {out_dir}")


if __name__ == "__main__":
    cli()
