import os

from pathlib import Path

import requests
import pandas as pd
import numpy as np
import numpy.typing as npt

from sklearn.utils import Bunch

from ..base import _preprocess

_BASE_URL = (
    "https://raw.githubusercontent.com/ipeirotis/Get-Another-Label/"
    "9fe6c56a7a157dba1a12018f696bad56806125b7/data/AdultContent2/"
)


def fetch_adult2(
    data_home: os.PathLike | str,
    download: bool = True,
    return_X_y: bool = False,
) -> Bunch | tuple[npt.NDArray, npt.NDArray]:
    """Fetch the AdultContent2 dataset (Ipeirotis, "Get Another Label").

    This function looks for the data files `data_home/adult2/labels.txt` and
    `data_home/adult2/gold.txt` to read the adult2 dataset. If these files do not
    exist and parameter `download` is True, it will attempt to download them from
    [ipeirotis/Get-Another-Label](https://github.com/ipeirotis/Get-Another-Label) —
    the original author's own repository — pinned to a specific commit so the
    downloaded content can't change underneath this function.

    `labels.txt` covers many more tasks (URLs) than `gold.txt` has gold labels
    for. Only tasks with a gold label are kept — mirroring how `fetch_rte` drops
    tasks without ground truth — since the tasks without a gold label are the
    ones this dataset's benchmark usage doesn't score against anyway.

    ??? Example
        The following code uses `data` folder under current working directory
        as `data_home` and loads (and downloads if not exist) the adult2 data.

        ```python
        from pathlib import Path
        import subcad

        data_home = Path("data")
        response_mat, gt_labels = subcad.data.fetch_adult2(data_home, return_X_y=True)
        ```

    Parameters
    ----------
    data_home :
        The directory under which to look for `adult2` folder
    download :
        Whether to download the data from the remote server if
        `data_home/adult2/labels.txt` or `data_home/adult2/gold.txt` is not found
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
          ground truth label of ith task.

        If `return_X_y` is True, the `(data, target)` tuple is returned directly.

    Raises
    ------
    Exception
        If remote server not available or `data_home/adult2/labels.txt` /
        `data_home/adult2/gold.txt` is not found.
    """

    dataset_dir = Path(data_home, "adult2")
    labels_file = Path(dataset_dir, "labels.txt")
    gold_file = Path(dataset_dir, "gold.txt")

    # Check if adult2 is already downloaded, if not download
    if (not labels_file.exists() or not gold_file.exists()) and download:
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for filename, dest in (("labels.txt", labels_file), ("gold.txt", gold_file)):
            response = requests.get(_BASE_URL + filename)
            if response.status_code == requests.codes.ok:
                with open(dest, mode="wb") as f:
                    f.write(response.content)
            else:
                raise Exception("Remote not available to download adult2 dataset.")

    try:
        response_df = pd.read_csv(
            labels_file, sep="\t", header=None, names=["worker_id", "task_id", "response"]
        )
        gt_df = pd.read_csv(
            gold_file, sep="\t", header=None, names=["task_id", "gt"]
        ).drop_duplicates(["task_id"])
    except:
        raise Exception("`labels.txt`/`gold.txt` files not found.")

    # Delete tasks without ground truth information from responses, and
    # duplicate (worker, task) annotations (the same worker labeling the same
    # task more than once)
    response_df = response_df[response_df["task_id"].isin(gt_df["task_id"])]
    response_df = response_df.drop_duplicates(["worker_id", "task_id"])

    # Create response matrix
    task_ids = response_df["task_id"].unique()
    worker_ids = response_df["worker_id"].unique()
    class_ids = response_df["response"].unique()

    n_tasks = len(task_ids)
    n_workers = len(worker_ids)

    task_to_idx = {t: i for i, t in enumerate(task_ids)}
    worker_to_idx = {t: i for i, t in enumerate(worker_ids)}
    class_to_idx = {t: i + 1 for i, t in enumerate(class_ids)}

    rows = [worker_to_idx[t] for t in response_df["worker_id"]]
    cols = [task_to_idx[t] for t in response_df["task_id"]]
    vals = [class_to_idx[t] for t in response_df["response"]]

    response_mat = np.zeros((n_workers, n_tasks), dtype=np.int64)
    response_mat[rows, cols] = vals

    # Convert ground truth labels to class indices
    gt_labels = np.zeros(n_tasks, dtype=np.int64)
    for _, row in gt_df.iterrows():
        if row["task_id"] in task_to_idx:
            gt_labels[task_to_idx[row["task_id"]]] = class_to_idx[row["gt"]]

    response_mat = _preprocess(response_mat)

    if return_X_y:
        return response_mat, gt_labels
    return Bunch(data=response_mat, target=gt_labels)
