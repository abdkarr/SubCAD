import os

from pathlib import Path

import requests
import pandas as pd
import numpy as np
import numpy.typing as npt

from sklearn.utils import Bunch

from ..base import _preprocess

_BASE_URL = (
    "https://raw.githubusercontent.com/maqqbu/MMSR/"
    "c85b4cc2419cd553cb2efea03ad62124f94aba32/original_datasets/"
)


def fetch_web(
    data_home: os.PathLike | str,
    download: bool = True,
    return_X_y: bool = False,
) -> Bunch | tuple[npt.NDArray, npt.NDArray]:
    """Fetch the Web Search Relevance Judging dataset (Zhou, Basu, Mao, Platt,
    "Learning from the Wisdom of Crowds by Minimax Entropy", NeurIPS 2012).

    This function looks for the data files `data_home/web/web_crowd.txt` and
    `data_home/web/web_truth.txt` to read the web dataset. If these files do not
    exist and parameter `download` is True, it will attempt to download them from
    [maqqbu/MMSR](https://github.com/maqqbu/MMSR) — the authors' own repository
    for Ma & Olshevsky, "Adversarial Crowdsourcing Through Robust Rank-One Matrix
    Completion", NeurIPS 2020, which vendors this dataset — pinned to a specific
    commit so the downloaded content can't change underneath this function.

    Not every task has a gold label in `web_truth.txt` (12 of the 2665 tasks are
    missing one) — unlike `fetch_rte`/`fetch_temp`, those tasks are kept (with
    `target == 0`, the same "no ground truth" sentinel used for missing worker
    responses) rather than dropped, matching the original `web.mat` shipped
    alongside this package.

    ??? Example
        The following code uses `data` folder under current working directory
        as `data_home` and loads (and downloads if not exist) the web data.

        ```python
        from pathlib import Path
        import subcad

        data_home = Path("data")
        response_mat, gt_labels = subcad.data.fetch_web(data_home, return_X_y=True)
        ```

    Parameters
    ----------
    data_home :
        The directory under which to look for `web` folder
    download :
        Whether to download the data from the remote server if
        `data_home/web/web_crowd.txt` or `data_home/web/web_truth.txt` is not found
    return_X_y :
        If True, returns `(response_mat, gt_labels)` instead of a
        [`sklearn.utils.Bunch`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html)
        object, mirroring the `return_X_y` convention used by
        `sklearn.datasets` loaders.

    Returns
    -------
    data : Bunch | tuple[npt.NDArray, npt.NDArray]
        If `return_X_y` is False, a `Bunch` with the following fields:

        - `data` : $(M, N)$ dimensional matrix where `data[i, j]` is the label
          provided by ith worker for jth task. `data[i, j] = 0` if no label is
          provided by the ith worker for jth task.
        - `target` : $(N, )$ dimensional vector where `target[i]` is the
          ground truth label of ith task, or `0` if no gold label exists for it.

        If `return_X_y` is True, the `(data, target)` tuple is returned directly.

    Raises
    ------
    Exception
        If remote server not available or `data_home/web/web_crowd.txt` /
        `data_home/web/web_truth.txt` is not found.
    """

    dataset_dir = Path(data_home, "web")
    crowd_file = Path(dataset_dir, "web_crowd.txt")
    truth_file = Path(dataset_dir, "web_truth.txt")

    # Check if web is already downloaded, if not download
    if (not crowd_file.exists() or not truth_file.exists()) and download:
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for filename, dest in (("web_crowd.txt", crowd_file), ("web_truth.txt", truth_file)):
            response = requests.get(_BASE_URL + filename)
            if response.status_code == requests.codes.ok:
                with open(dest, mode="wb") as f:
                    f.write(response.content)
            else:
                raise Exception("Remote not available to download web dataset.")

    try:
        crowd_df = pd.read_csv(
            crowd_file, sep="\t", header=None, names=["task_id", "worker_id", "response"]
        )
        gt_df = pd.read_csv(
            truth_file, sep="\t", header=None, names=["task_id", "gt"]
        ).drop_duplicates(["task_id"])
    except:
        raise Exception("`web_crowd.txt`/`web_truth.txt` files not found.")

    # Create response matrix
    task_ids = crowd_df["task_id"].unique()
    worker_ids = crowd_df["worker_id"].unique()
    class_ids = crowd_df["response"].unique()

    n_tasks = len(task_ids)
    n_workers = len(worker_ids)

    task_to_idx = {t: i for i, t in enumerate(task_ids)}
    worker_to_idx = {t: i for i, t in enumerate(worker_ids)}
    class_to_idx = {t: i + 1 for i, t in enumerate(class_ids)}

    rows = [worker_to_idx[t] for t in crowd_df["worker_id"]]
    cols = [task_to_idx[t] for t in crowd_df["task_id"]]
    vals = [class_to_idx[t] for t in crowd_df["response"]]

    response_mat = np.zeros((n_workers, n_tasks), dtype=np.int64)
    response_mat[rows, cols] = vals

    # Convert ground truth labels to class indices. Tasks missing from
    # web_truth.txt are left at the default 0 ("no ground truth") rather than
    # dropped, since web_truth.txt doesn't cover every task in web_crowd.txt.
    gt_labels = np.zeros(n_tasks, dtype=np.int64)
    for _, row in gt_df.iterrows():
        if row["task_id"] in task_to_idx:
            gt_labels[task_to_idx[row["task_id"]]] = class_to_idx[row["gt"]]

    response_mat = _preprocess(response_mat)

    if return_X_y:
        return response_mat, gt_labels
    return Bunch(data=response_mat, target=gt_labels)
