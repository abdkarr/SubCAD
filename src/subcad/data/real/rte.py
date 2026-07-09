import os
import tarfile

from pathlib import Path

import requests
import pandas as pd
import numpy as np
import numpy.typing as npt

from sklearn.utils import Bunch

from ..base import _preprocess


def fetch_rte(
    data_home: os.PathLike | str,
    download: bool = True,
    return_X_y: bool = False,
) -> Bunch | tuple[npt.NDArray, npt.NDArray]:
    """Fetch the RTE dataset.

    This function looks for the data file `data_home/rte/rte.standardized.tsv`
    to read RTE dataset. If this file does not exist and parameter `download`
    is True, it will attempt to download the file from the remote server at
    [here](https://web.archive.org/web/20230331023329/https://sites.google.com/site/nlpannotations/).

    ??? Example
        The following code uses `data` folder under current working directory
        as `data_home` and loads (and downloads if not exist) the RTE data.

        ```python
        from pathlib import Path
        import subcad

        data_home = Path("data")
        response_mat, gt_labels = subcad.data.fetch_rte(data_home, return_X_y=True)
        ```

    Parameters
    ----------
    data_home :
        The directory under which to look for `rte` folder
    download :
        Whether to download the data from the remote server if
        `data_home/rte/rte.standardized.tsv` is not found
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
        If remote server not available or `data_home/rte/rte.standardized.tsv` is
        not found.
    """

    download_url = (
        "https://web.archive.org/web/20230331023329/"
        + "https://sites.google.com/site/nlpannotations/all_collected_data.tgz"
        + "?attredirects=0"
    )
    dataset_dir = Path(data_home, "rte")
    dataset_file = Path(dataset_dir, "rte.standardized.tsv")

    # Check if RTE is already downloaded, if not download
    if (not dataset_file.exists()) and download:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        zipped_file = Path(dataset_dir, "all_collected_data.tgz")

        with requests.get(download_url, stream=True) as response:
            if response.status_code == requests.codes.ok:
                with open(zipped_file, mode="wb") as f:
                    for chunk in response.iter_content(chunk_size=10 * 1024):
                        f.write(chunk)
            else:
                raise Exception("Remote not available to download RTE dataset.")

        # Unzip rte dataset from downloaded zip file, remove the rest
        with tarfile.open(zipped_file, "r:gz") as f:
            f.extractall(dataset_dir, members=["rte.standardized.tsv"], filter="data")

        zipped_file.unlink()

    try:
        raw_data = pd.read_csv(
            dataset_file,
            sep="\t",
            header=0,
            names=["annotation_id", "worker_id", "task_id", "response", "gt"],
        )
    except:
        raise Exception("`rte.standardized.tsv` file not found.")

    response_df = raw_data[["task_id", "worker_id", "response"]]
    gt_df = raw_data[["task_id", "gt"]].drop_duplicates(["task_id"])

    # Delete tasks without ground truth information from responses
    missing_tasks = np.setdiff1d(response_df["task_id"], gt_df["task_id"])
    missing_idx = np.isin(response_df["task_id"], missing_tasks)
    response_df = response_df.iloc[~missing_idx, :]

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
    for i, row in gt_df.iterrows():
        gt_labels[task_to_idx[row["task_id"]]] = class_to_idx[row["gt"]]

    response_mat = _preprocess(response_mat)

    if return_X_y:
        return response_mat, gt_labels
    return Bunch(data=response_mat, target=gt_labels)
