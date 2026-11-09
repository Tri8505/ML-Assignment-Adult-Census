"""Download and load the official UCI Adult Census Income dataset.

The public UCI train/test split is preserved. Missing categorical values are
represented by ``pandas.NA`` and the test labels are normalized to match the
training labels.
"""

from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

UCI_DATASET_PAGE = "https://archive.ics.uci.edu/dataset/2/adult"
TRAIN_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)
TEST_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
)

COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

EXPECTED_TARGETS = {"<=50K", ">50K"}
DEFAULT_DATA_DIR = Path("data/raw")


class AdultDataError(RuntimeError):
    """Raised when the Adult data cannot be downloaded or validated."""


def _download_file(url: str, destination: Path) -> None:
    """Download one URL atomically to avoid leaving a partial cached file."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "ML-Assignment-Adult-Census/1.0"})

    try:
        with urlopen(request, timeout=60) as response, temporary_path.open("wb") as file:
            file.write(response.read())
        temporary_path.replace(destination)
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        temporary_path.unlink(missing_ok=True)
        raise AdultDataError(f"Không thể tải dữ liệu từ {url}: {error}") from error


def download_adult_data(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    force_download: bool = False,
) -> tuple[Path, Path]:
    """Download the official train/test files and return their local paths.

    Existing non-empty files are reused unless ``force_download`` is true.
    This makes repeated Colab runs faster while requiring no cloud drive.
    """

    destination = Path(data_dir)
    train_path = destination / "adult.data"
    test_path = destination / "adult.test"

    files = ((TRAIN_URL, train_path), (TEST_URL, test_path))
    for url, path in files:
        if force_download or not path.is_file() or path.stat().st_size == 0:
            _download_file(url, path)

    return train_path, test_path


def _read_partition(path: Path, *, is_test: bool) -> pd.DataFrame:
    """Parse and normalize one UCI partition."""

    try:
        frame = pd.read_csv(
            path,
            names=COLUMN_NAMES,
            header=None,
            skiprows=1 if is_test else 0,
            skipinitialspace=True,
            na_values=["?"],
            keep_default_na=True,
        )
    except (OSError, pd.errors.ParserError) as error:
        raise AdultDataError(f"Không thể đọc tệp dữ liệu {path}: {error}") from error

    frame = frame.dropna(how="all").reset_index(drop=True)
    text_columns = frame.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        frame[column] = frame[column].astype("string").str.strip()

    # UCI appends a period to labels in adult.test but not in adult.data.
    frame["income"] = frame["income"].str.removesuffix(".")
    _validate_partition(frame, path)
    return frame


def _validate_partition(frame: pd.DataFrame, source: Path) -> None:
    if frame.empty:
        raise AdultDataError(f"Tệp dữ liệu rỗng: {source}")
    if frame.columns.tolist() != COLUMN_NAMES:
        raise AdultDataError(f"Cấu trúc cột không hợp lệ trong {source}")

    observed_targets = set(frame["income"].dropna().unique())
    unexpected_targets = observed_targets - EXPECTED_TARGETS
    if unexpected_targets:
        raise AdultDataError(
            f"Nhãn mục tiêu không hợp lệ trong {source}: {sorted(unexpected_targets)}"
        )


def load_adult_data(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    force_download: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cleaned official training and test DataFrames.

    Examples
    --------
    >>> train_df, test_df = load_adult_data()
    >>> train_df.shape
    (32561, 15)
    """

    train_path, test_path = download_adult_data(
        data_dir, force_download=force_download
    )
    train_frame = _read_partition(train_path, is_test=False)
    test_frame = _read_partition(test_path, is_test=True)
    return train_frame, test_frame


def combine_partitions(
    train_frame: pd.DataFrame, test_frame: pd.DataFrame
) -> pd.DataFrame:
    """Combine partitions for descriptive EDA and retain their origin."""

    train = train_frame.assign(dataset_split="train")
    test = test_frame.assign(dataset_split="test")
    return pd.concat([train, test], ignore_index=True)
