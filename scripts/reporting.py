"""
Summaries of the planted-attack experiments, used by the `aggregate` subcommand of
`planted_attacks.py` and `planted_attacks_with_experts.py`: reads the per-seed result
files of an experiment and writes six CSVs (Dataset x Method, one row per dataset,
"<method>"/"<method>_ci" columns with the mean and the half-width of the 95% t-interval
over the seeds, unless noted otherwise):

- label-fusion.csv              - Accuracy, for every baseline/competitor/proposed method
- adversary-detection-auroc.csv - WorkerROCAUC, for the competitors and the best proposed method
- adversary-detection-auprc.csv - WorkerAP (average precision), same methods
- target-detection-auroc.csv    - TaskROCAUC, for the best proposed method only
- target-detection-auprc.csv    - TaskAP (average precision), same method
- significance.csv              - per dataset/metric, whether the best method is
  significantly better than the second-best (Wilcoxon signed-rank test)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATASETS = ["rte", "sp", "web", "dog", "temp", "adult2"]
BASELINES = ["glad", "ebcc", "la-onepass", "la-twopass", "mv", "ds"]
COMPETITORS = ["dacs", "repalg-soft", "repalg-hard", "mmsr"]
PROPOSED = ["subcad-wgsmv", "subcad-wgsds"]


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


def _read_results(exp_dir: Path, setting, datasets: list[str], methods: list[str]):
    """Read every per-seed result CSV of an experiment into one long table.

    `setting(dataset)` returns the name of the directory (below the dataset
    directory) holding the results for that dataset's data-generation settings.
    """
    rows = []
    for dataset in datasets:
        for method in methods:
            results_dir = Path(exp_dir, dataset, setting(dataset), method)
            for seed_file in sorted(results_dir.glob("seed-*.csv")):
                df = pd.read_csv(seed_file)
                df["Dataset"] = dataset
                df["MethodKey"] = method
                df["Seed"] = int(seed_file.stem.split("-")[1])
                rows.append(df)
    return pd.concat(rows, ignore_index=True)


def write_reports(exp_dir: Path, setting, out_dir: Path, confidence: float = 0.95) -> None:
    """Summarize the results below `exp_dir` into the six report CSVs in `out_dir`."""
    datasets = DATASETS
    methods = BASELINES + COMPETITORS + PROPOSED
    adversary_detectors = COMPETITORS + [PROPOSED[-1]]
    target_detectors = [PROPOSED[-1]]

    results = _read_results(exp_dir, setting, datasets, methods)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(df: pd.DataFrame, name: str) -> None:
        df.to_csv(Path(out_dir, name), index=False, float_format="%.4f")

    save(_summarize(results, "Accuracy", datasets, methods, confidence), "label-fusion.csv")
    save(
        _summarize(results, "WorkerROCAUC", datasets, adversary_detectors, confidence),
        "adversary-detection-auroc.csv",
    )
    save(
        _summarize(results, "WorkerAP", datasets, adversary_detectors, confidence),
        "adversary-detection-auprc.csv",
    )
    save(
        _summarize(results, "TaskROCAUC", datasets, target_detectors, confidence),
        "target-detection-auroc.csv",
    )
    save(
        _summarize(results, "TaskAP", datasets, target_detectors, confidence),
        "target-detection-auprc.csv",
    )

    significance = pd.concat(
        [
            _best_vs_second_best(results, "Accuracy", datasets, methods),
            _best_vs_second_best(results, "WorkerROCAUC", datasets, adversary_detectors),
            _best_vs_second_best(results, "WorkerAP", datasets, adversary_detectors),
        ],
        ignore_index=True,
    )
    save(significance, "significance.csv")

    print(f"Wrote summaries to {out_dir}")
