import os

from polars.dataframe import DataFrame
from polars.lazyframe import LazyFrame
from polars.io.ipc import read_ipc, scan_ipc

_LOCAL_FOLDER = os.path.join(os.path.dirname(__file__), "")
# _REMOTE_FOLDER = "https://github.com/Zaf4/datasets/raw/main/feather/" # for tests
_REMOTE_FOLDER="https://github.com/pola-rs/polars/raw/main/examples/datasets/"
_EXTESION = ".feather"

_DATASETS = {
    "iris": f"{_REMOTE_FOLDER}iris{_EXTESION}",
    "mpg": f"{_REMOTE_FOLDER}mpg{_EXTESION}",
    "netflix": f"{_REMOTE_FOLDER}netflix{_EXTESION}",
    "starbucks": f"{_REMOTE_FOLDER}starbucks{_EXTESION}",
    "titanic": f"{_REMOTE_FOLDER}titanic{_EXTESION}",
}

_LOCAL_PATHS = {
    name: path.replace(_REMOTE_FOLDER, _LOCAL_FOLDER)
    for name, path in _DATASETS.items()
}


def load(
    name: str, *, lazy: bool = False, check_local: bool = True, cache: bool = True
) -> DataFrame | LazyFrame:
    """
    Loads dataset by with the given name if available.

    Parameters
    ----------
    name : str
        Name of the dataset to load.
    lazy : bool, optional
        Lazy loading,
        Default is  False

    check_local : bool, optional
        Whether to check if the dataset is available locally
        Default is True

    cache : bool, optional
        Whether to cache the dataset locally.
        Default is True

    Returns
    -------
    polars.DataFrame or polars.LazyFrame
    """
    # select the function to load the dataset
    loader = scan_ipc if lazy else read_ipc
    # check if the dataset is available
    if name not in _DATASETS.keys():
        msg = f"Dataset {name} not found."
        raise ValueError(msg)

    local_path = _LOCAL_PATHS.get(name)
    # load the dataset
    if check_local:
        if os.path.exists(local_path):
            frame = loader(local_path)
        else:
            frame = loader(_DATASETS.get(name))
    else:
        frame = loader(_DATASETS.get(name))

    # cache the dataset locally
    if cache and not os.path.exists(local_path):
        if lazy:
            frame.sink_ipc(local_path, compression="zstd")
        else:
            frame.write_ipc(local_path, compression="zstd")

    if os.path.exists(local_path):
        print(local_path)
    return frame


def available(option: str | None = None) -> list[str]:
    """
    List available datasets.

    Parameters
    ----------
    option : str, optional
        The option to list available datasets.
        Default is None

    Returns
    -------
    list of available datasets

    Raises
    ------
    ValueError
        If the option is not 'remote' or 'local'.

    """
    if option is None or option == "remote":
        return list(_DATASETS.keys())
    elif option == "local":
        return [x for x in _LOCAL_PATHS.keys() if os.path.exists(_LOCAL_PATHS.get(x))]
    else:
        msg = "Invalid option. Please use either 'remote' or 'local'."
        raise ValueError(msg)
