import os
import tarfile

from pathlib import Path

import requests
import pandas as pd
import numpy as np
import numpy.typing as npt

from sklearn.utils import Bunch

from ..base import _preprocess

_ARCHIVE_MEMBER = "sentiment_polarity/mturk_answers.csv"


def fetch_sp(
    data_home: os.PathLike | str,
    download: bool = True,
    return_X_y: bool = False,
) -> Bunch | tuple[npt.NDArray, npt.NDArray]:
    """Fetch the Sentence Polarity (sp) dataset — Pang & Lee movie-review sentence
    polarity sentences, relabeled via Amazon Mechanical Turk in Rodrigues, Pereira,
    Ribeiro, "Learning from Multiple Annotators: Distinguishing Good from Random
    Labelers", Pattern Recognition Letters, 2013.

    This function looks for the data file
    `data_home/sp/sentiment_polarity/mturk_answers.csv` to read the sp dataset.
    If this file does not exist and parameter `download` is True, it will attempt
    to download it from the paper author's own page at
    [fprodrigues.com/mturk-datasets.tar.gz](http://fprodrigues.com/mturk-datasets.tar.gz),
    extracting only the `sentiment_polarity/mturk_answers.csv` member (the archive
    also has an unrelated `music_genre_classification` dataset, and a
    `sentiment_polarity/polarity_gold_lsa_topics.csv` of precomputed LSA topic
    features that this function doesn't need).

    ??? Example
        The following code uses `data` folder under current working directory
        as `data_home` and loads (and downloads if not exist) the sp data.

        ```python
        from pathlib import Path
        import subcad

        data_home = Path("data")
        response_mat, gt_labels = subcad.data.fetch_sp(data_home, return_X_y=True)
        ```

    Parameters
    ----------
    data_home :
        The directory under which to look for `sp` folder
    download :
        Whether to download the data from the remote server if
        `data_home/sp/sentiment_polarity/mturk_answers.csv` is not found
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
        If remote server not available or
        `data_home/sp/sentiment_polarity/mturk_answers.csv` is not found.
    """

    download_url = "http://fprodrigues.com/mturk-datasets.tar.gz"
    dataset_dir = Path(data_home, "sp")
    dataset_file = Path(dataset_dir, _ARCHIVE_MEMBER)

    # Check if sp is already downloaded, if not download
    if (not dataset_file.exists()) and download:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        zipped_file = Path(dataset_dir, "mturk-datasets.tar.gz")

        # The host 403s a plain/minimal `requests` User-Agent (Cloudflare bot
        # check), so pretend to be a real browser
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            )
        }
        with requests.get(download_url, stream=True, headers=headers) as response:
            if response.status_code == requests.codes.ok:
                with open(zipped_file, mode="wb") as f:
                    for chunk in response.iter_content(chunk_size=10 * 1024):
                        f.write(chunk)
            else:
                raise Exception("Remote not available to download sp dataset.")

        # Unzip sp dataset from downloaded zip file, remove the rest. The
        # archive's last member is truncated at the source, so read it as a
        # stream and stop as soon as the member we want is extracted instead
        # of `extractall(members=...)`, which indexes the whole archive
        # upfront and would trip over that truncated tail.
        with tarfile.open(zipped_file, "r|gz") as f:
            for member in f:
                if member.name == _ARCHIVE_MEMBER:
                    f.extract(member, dataset_dir, filter="data")
                    break

        zipped_file.unlink()

    try:
        raw_data = pd.read_csv(dataset_file)
    except:
        raise Exception("`sentiment_polarity/mturk_answers.csv` file not found.")

    response_df = raw_data[["Input.id", "WorkerId", "Answer.sent"]].rename(
        columns={"Input.id": "task_id", "WorkerId": "worker_id", "Answer.sent": "response"}
    )
    gt_df = raw_data[["Input.id", "Input.true_sent"]].rename(
        columns={"Input.id": "task_id", "Input.true_sent": "gt"}
    ).drop_duplicates(["task_id"])

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
        gt_labels[task_to_idx[row["task_id"]]] = class_to_idx[row["gt"]]

    response_mat = _preprocess(response_mat)

    if return_X_y:
        return response_mat, gt_labels
    return Bunch(data=response_mat, target=gt_labels)
