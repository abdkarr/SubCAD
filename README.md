## Datasets

`subcad.data` provides `fetch_*` functions that download the crowdsourcing benchmark
datasets used in experiments from the following hosts:

- **RTE** (`fetch_rte`) — Snow, O'Connor, Jurafsky, Ng, "Cheap and Fast — But is it
  Good? Evaluating Non-Expert Annotations for Natural Language Tasks" (EMNLP 2008),
  via a [Wayback Machine snapshot](https://web.archive.org/web/20230331023329/https://sites.google.com/site/nlpannotations/all_collected_data.tgz)
  of the authors' data page.
- **Temporal Ordering / temp** (`fetch_temp`) — same archive as RTE above (Snow et
  al., 2008).
- **Dog** (`fetch_dog`) / **Web Search Relevance Judging / web** (`fetch_web`) —
  Zhou, Platt, Basu, Mao, "Learning from the Wisdom of Crowds by Minimax Entropy"
  (NeurIPS 2012), vendored in [maqqbu/MMSR](https://github.com/maqqbu/MMSR) (Ma &
  Olshevsky, "Adversarial Crowdsourcing Through Robust Rank-One Matrix Completion",
  NeurIPS 2020).
- **AdultContent2 / adult2** (`fetch_adult2`) — Ipeirotis,
  [Get-Another-Label](https://github.com/ipeirotis/Get-Another-Label).
- **Sentence Polarity / sp** (`fetch_sp`) — Rodrigues, Pereira, Ribeiro, "Learning
  from Multiple Annotators: Distinguishing Good from Random Labelers" (Pattern
  Recognition Letters, 2013), from the authors'
  [fprodrigues.com/mturk-datasets.tar.gz](http://fprodrigues.com/mturk-datasets.tar.gz)
  (itself derived from Pang & Lee's sentence polarity corpus, relabeled via
  Mechanical Turk).

> **Disclaimer:** These datasets are hosted by their original authors or third
> parties, not by this project. `subcad` only provides code to download and parse
> them for research use — it does not redistribute the data itself, and makes no
> claim about the licensing or redistribution terms of any of them. Check each
> source's own terms before using the data outside of a research context.

## Experiment configs

Experiments driven by `scripts/dispatcher.py` (e.g. via `scripts/planted_attacks.py`)
are configured with a TOML file (see `configs/planted_attacks.toml`) containing
`[dataset.*]` and `[method.*]` sections. Each `[method.*]` section is passed as the
`cfg` dict to a runner in `dispatcher.REGISTRY`, keyed by its `name`.

For `name = "subcad"` (`dispatcher.run_subcad`), a method section must set:

- `kind` — `"binary"` or `"weighted"`, the bipartite graph construction used by the
  detector.
- `detector` — `"greedy"`, `"greedypp"`, or `"spectral"`.
- `selector` — `"density"` or `"spectral"`.
- `aggregator` — `"mv"` (Weighted Majority Voting) or `"ds"` (Weighted Dawid-Skene).

`detector = "greedypp"` additionally requires an `iteration` search value, and the
aggregator accepts optional `adv_frac`/`target_frac`/`scale` search values (falling
back to the selector's own size estimate for `adv_frac`/`target_frac`, and `5.0` for
`scale`, if omitted). Each of these accepts one of four forms, resolved by
`dispatcher._resolve_array`:

- `iteration = 10` — a single value.
- `iterations = [5, 10, 20]` — an explicit list to sweep over.
- `iteration_logspace = [start, end, num]` — `np.logspace(start, end, num)`.
- `iteration_linspace = [start, end, num]` — `np.linspace(start, end, num)`.

(same pattern for `adv_frac`/`target_frac`/`scale`). `run_subcad` yields one
`ModelResult` per combination in the sweep.

Example:

```toml
[method.subcad-wgdmv]
name = "subcad"
kind = "weighted"
detector = "greedy"
selector = "density"
aggregator = "mv"

[method.subcad-wgppdmv]
name = "subcad"
kind = "weighted"
detector = "greedypp"
selector = "density"
aggregator = "mv"
iteration = 10
```
