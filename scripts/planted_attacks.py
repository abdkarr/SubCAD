"""
Experiment testing detection/selection/aggregation methods on a real
crowdsourcing dataset corrupted with synthetic adversaries (Sec. VI-B).

One call runs one (dataset, method, seed) and writes (overwriting) its metrics to
`research/data/outputs/planted_attacks/<dataset>/<setting>/<method>/seed-<seed>.csv`.
The data-generation settings are the paper's (below); other settings are covered
by `sweeps.py`. `aggregate_results.py planted-attacks` summarizes the results.
"""

from pathlib import Path

import click
import pandas as pd

import subcad
from dispatcher import METHODS, evaluate
from planting import LOADER, plant_attacks

EXP_NAME = "planted_attacks"
INPUT_DIR = Path(subcad.PROJECT_DIR, "research", "data", "inputs")
OUTPUT_DIR = Path(subcad.PROJECT_DIR, "research", "data", "outputs", EXP_NAME)

# Settings of TSP paper
ADV_FRAC = 0.2  # fraction of the workers that are adversaries
TARGET_FRAC = 0.2  # fraction of the tasks that are targeted
CAMO_RELIABILITY = 2
# probability that an adversary labels a targeted task (rho_a)
TARGET_OBS = {"rte": 0.3, "sp": 0.3, "web": 0.3, "dog": 0.6, "adult2": 0.6, "temp": 0.6}


@click.command()
@click.option("--dataset", type=click.Choice(list(TARGET_OBS)), required=True)
@click.option("--method", type=click.Choice(list(METHODS)), required=True)
@click.option("--seed", default=1, show_default=True)
def cli(dataset, method, seed):
    target_obs = TARGET_OBS[dataset]
    honest_responses, gt_labels = LOADER[dataset](INPUT_DIR)
    response_mat, gt_adversaries, gt_targeted = plant_attacks(
        honest_responses,
        gt_labels,
        seed,
        adv_frac=ADV_FRAC,
        target_frac=TARGET_FRAC,
        target_obs=target_obs,
        camo_reliability=CAMO_RELIABILITY,
    )

    # RepAlg is given the true fraction of adversaries (oracle)
    options = {"adv_frac": ADV_FRAC} if method.startswith("repalg") else {}
    result = METHODS[method](response_mat, **options)
    row = evaluate(result, gt_adversaries, gt_targeted, gt_labels)

    setting = f"af={ADV_FRAC}_tf={TARGET_FRAC}_to={target_obs}_cr={CAMO_RELIABILITY}"
    out_path = OUTPUT_DIR / dataset / setting / method / f"seed-{seed}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out_path, index=False, float_format="%.4f")


if __name__ == "__main__":
    cli()
