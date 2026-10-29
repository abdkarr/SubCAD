"""
Experiment testing detection/selection/aggregation methods on a real
crowdsourcing dataset corrupted with synthetic adversaries, with an
additional pool of synthetic "expert" honest annotators mixed in.

`run` runs one (dataset, method, seed) and writes (overwriting) its metrics to
`research/data/outputs/planted_attacks_with_experts/<dataset>/<setting>/<method>/seed-<seed>.csv`.
`aggregate` summarizes the results of all seeds.
"""

from pathlib import Path

import click
import numpy as np
import pandas as pd

import subcad
from dispatcher import LOADER, METHODS, evaluate
from reporting import write_reports

EXP_NAME = "planted_attacks_with_experts"
INPUT_DIR = Path(subcad.PROJECT_DIR, "research", "data", "inputs")
OUTPUT_DIR = Path(subcad.PROJECT_DIR, "research", "data", "outputs", EXP_NAME)
REPORTS_DIR = Path(subcad.PROJECT_DIR, "research", "reports", EXP_NAME)

# Settings of TSP paper
ADV_FRAC = 0.2  # fraction of the workers that are adversaries
TARGET_FRAC = 0.2  # fraction of the tasks that are targeted
CAMO_RELIABILITY = 2
EXPERT_FRAC = 0.2  # fraction of the workers that are experts
EXPERT_RELIABILITY = 4
# probability that an adversary labels a targeted task (rho_a)
TARGET_OBS = {"rte": 0.3, "sp": 0.3, "web": 0.3, "dog": 0.6, "adult2": 0.6, "temp": 0.6}


def setting(dataset):
    """Name of the output folder of the settings used for `dataset`."""
    return (
        f"af={ADV_FRAC}_tf={TARGET_FRAC}_to={TARGET_OBS[dataset]}_cr={CAMO_RELIABILITY}"
        f"_ef={EXPERT_FRAC}_er={EXPERT_RELIABILITY}"
    )


@click.group()
def cli():
    pass


@cli.command()
@click.option("--dataset", type=click.Choice(list(TARGET_OBS)), required=True)
@click.option("--method", type=click.Choice(list(METHODS)), required=True)
@click.option("--seed", default=1, show_default=True)
def run(dataset, method, seed):
    """Run one method on one corrupted copy of a dataset."""
    target_obs = TARGET_OBS[dataset]

    # Load real data as the honest workers
    honest_responses, gt_labels = LOADER[dataset](INPUT_DIR)

    n_honests, n_tasks = honest_responses.shape
    honest_obs = np.mean(np.count_nonzero(honest_responses, axis=1) / n_tasks)
    n_adversaries = int(ADV_FRAC * n_honests / (1 - ADV_FRAC - EXPERT_FRAC))
    n_experts = int(EXPERT_FRAC * n_honests / (1 - ADV_FRAC - EXPERT_FRAC))

    rng = np.random.default_rng(seed)

    # Generate synthetic adversaries
    adv_responses, gt_targeted = subcad.data.make_adversaries(
        gt_labels,
        n_adversaries,
        target_frac=TARGET_FRAC,
        camo_obs=honest_obs,
        target_obs=target_obs,
        camo_reliability=CAMO_RELIABILITY,
        random_state=rng,
    )

    # Generate synthetic experts, i.e. camouflage-only "adversaries" that
    # never target a task, just honest annotators with a set reliability.
    expert_responses, _ = subcad.data.make_adversaries(
        gt_labels,
        n_experts,
        target_frac=0.0,
        camo_obs=honest_obs,
        target_obs=0.0,
        camo_reliability=EXPERT_RELIABILITY,
        random_state=rng,
    )

    # Combine honest, adversary and expert responses with shuffling
    response_mat = np.vstack([honest_responses, adv_responses, expert_responses])
    gt_adversaries = np.hstack(
        [np.zeros(n_honests), np.ones(n_adversaries), np.zeros(n_experts)]
    )

    idx = np.arange(response_mat.shape[0])
    rng.shuffle(idx)
    response_mat = response_mat[idx, :]
    gt_adversaries = gt_adversaries[idx]

    # RepAlg is given the true fraction of adversaries (oracle)
    options = {"adv_frac": ADV_FRAC} if method.startswith("repalg") else {}
    result = METHODS[method](response_mat, **options)
    row = evaluate(result, gt_adversaries, gt_targeted, gt_labels)

    out_path = OUTPUT_DIR / dataset / setting(dataset) / method / f"seed-{seed}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out_path, index=False, float_format="%.4f")


@cli.command()
@click.option("--out-dir", type=click.Path(path_type=Path), default=REPORTS_DIR, show_default=True)
def aggregate(out_dir):
    """Summarize the results of all seeds: mean and 95% CI, significance tests."""
    write_reports(OUTPUT_DIR, setting, out_dir)


if __name__ == "__main__":
    cli()
