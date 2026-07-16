"""
Experiment testing detection/selection/aggregation methods on a real
crowdsourcing dataset corrupted with synthetic adversaries, with an
additional pool of synthetic "expert" honest annotators mixed in.
"""

import tomllib

from pathlib import Path

import click
import numpy as np
import pandas as pd

import subcad
import dispatcher

EXP_NAME = "planted_attacks_with_experts"
INPUT_DIR = Path(subcad.PROJECT_DIR, "research", "data", "inputs")
OUTPUT_DIR = Path(subcad.PROJECT_DIR, "research", "data", "outputs", EXP_NAME)


def _load_config(
    config: str, dataset: str | None, method: str | None
) -> tuple[dict, dict, dict]:
    cfg_path = Path(config)
    if not cfg_path.is_absolute():
        cfg_path = Path(subcad.PROJECT_DIR) / cfg_path
    cfg = tomllib.loads(cfg_path.read_text())

    dataset_cfg = cfg.get("dataset", {}).get(dataset)
    if dataset_cfg is None:
        raise click.BadParameter(
            f"[dataset.{dataset}] not found in {config}", param_hint="--dataset"
        )

    method_cfg = cfg.get("method", {}).get(method)
    if method_cfg is None:
        raise click.BadParameter(
            f"[method.{method}] not found in {config}", param_hint="--method"
        )

    return dataset_cfg, method_cfg


@click.command()
@click.option("--config", default="configs/planted_attacks_with_experts.toml")
@click.option(
    "--dataset", default="rte", help="Key into [dataset.*] section of the config TOML."
)
@click.option(
    "--method",
    default="subcad-wgdmv",
    help="Key into [method.*] section of the config TOML.",
)
@click.option("--seed", default=1)
def cli(config, dataset, method, seed):
    data_cfg, method_cfg = _load_config(config, dataset, method)

    # Data generation parameters
    dataset_name = data_cfg["name"]
    adv_frac = data_cfg["adv_frac"]
    target_frac = data_cfg["target_frac"]
    target_obs = data_cfg["target_obs"]
    camo_reliability = data_cfg["camo_reliability"]
    expert_frac = data_cfg["expert_frac"]
    expert_reliability = data_cfg["expert_reliability"]

    # Load real data as the honest workers
    honest_responses, gt_labels = dispatcher.LOADER[dataset_name](INPUT_DIR)

    n_honests, n_tasks = honest_responses.shape
    honest_obs = np.mean(np.count_nonzero(honest_responses, axis=1) / n_tasks)
    n_adversaries = int(adv_frac * n_honests / (1 - adv_frac - expert_frac))
    n_experts = int(expert_frac * n_honests / (1 - adv_frac - expert_frac))

    rng = np.random.default_rng(seed)

    # Generate synthetic adversaries
    adv_responses, gt_targeted = subcad.data.make_adversaries(
        gt_labels,
        n_adversaries,
        target_frac=target_frac,
        camo_obs=honest_obs,
        target_obs=target_obs,
        camo_reliability=camo_reliability,
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
        camo_reliability=expert_reliability,
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

    # Detection / Selection / Aggregation with Evaluation
    method_name = method_cfg["name"]
    perfs = []
    for result in dispatcher.REGISTRY[method_name](response_mat, method_cfg):
        perfs.append(
            dispatcher.evaluate(result, gt_adversaries, gt_targeted, gt_labels)
        )

    # Save the performance
    save_dir = (
        f"{dataset_name}/"
        f"af={adv_frac}_tf={target_frac}_to={target_obs}_cr={camo_reliability}_"
        f"ef={expert_frac}_er={expert_reliability}/"
        f"{method}"
    )
    save_file = f"seed-{seed:d}.csv"
    out_path = Path(OUTPUT_DIR, save_dir, save_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(perfs).to_csv(out_path, index=False, float_format="%.4f")


if __name__ == "__main__":
    cli()
