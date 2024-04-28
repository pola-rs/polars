from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from polars.dataframe import DataFrame
from polars.expr import Expr
from polars.selectors import exclude

if TYPE_CHECKING:
    import sys

    from torch import Tensor, memory_format

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self
try:
    import torch
    from torch.utils.data import TensorDataset
except ImportError:
    msg = (
        "Required package 'torch' not installed.\n"
        "Please install it using the command `pip install torch`."
    )
    raise ImportError(msg) from None


class PolarsDataset(TensorDataset):
    """Specialized TensorDataset for Polars DataFrames."""

    tensors: tuple[Tensor, ...]
    labels: Tensor | None
    features: Tensor

    def __init__(
        self,
        frame: DataFrame,
        *,
        label: str | Expr | Sequence[str | Expr] | None = None,
    ):
        """
        TensorDataset class specialized for use with Polars DataFrames.

        Parameters
        ----------
        frame
            Polars DataFrame containing the data that will be retrieved as Tensors.
        label
            One or more column names or expressions that label the feature data; results
            in `(features,label)` tuples, where all non-label columns are considered
            to be features. If no label is designated then each returned item is a
            simple `(features,)` tuple containing all row elements.

        Notes
        -----
        * The PolarsDataset supports indexing with integers, slices, ranges, lists, and
          integer Tensors.
        * When indexing with an object that represents multiple items (such as a list of
          row indexes or a slice) the return value is a sequence of tensor tuples.
        * Designating multi-element labels is supported.

        Examples
        --------
        >>> from torch.utils.data import DataLoader
        >>> df = pl.DataFrame(
        ...     data=[
        ...         (0, 1, 1.5),
        ...         (1, 0, -0.5),
        ...         (2, 0, 0.0),
        ...         (3, 1, -2.25),
        ...     ],
        ...     schema=["lbl", "feat1", "feat2"],
        ... )

        Create a Dataset from a Polars DataFrame, standardising the dtype and
        distinguishing the label/feature columns.

        >>> ds = df.to_torch("dataset", label="lbl", dtype=pl.Float32)
        >>> ds  # doctest: +IGNORE_RESULT
        <PolarsDataset [len:4, features:2, labels:1] at 0x156B033B0>
        >>> ds.features
        tensor([[ 1.0000,  1.5000],
                [ 0.0000, -0.5000],
                [ 0.0000,  0.0000],
                [ 1.0000, -2.2500]])
        >>> ds[0]
        (tensor([1.0000, 1.5000]), tensor(0.))

        The Dataset can be used standalone, or in conjunction with a DataLoader.

        >>> dl = DataLoader(ds, batch_size=2)
        >>> list(dl)
        [[tensor([[ 1.0000,  1.5000],
                  [ 0.0000, -0.5000]]),
          tensor([0., 1.])],
         [tensor([[ 0.0000,  0.0000],
                  [ 1.0000, -2.2500]]),
          tensor([2., 3.])]]

        Note that the label can be given as an expression as well as a column name,
        allowing for independent transform and dtype adjustment from the feature
        columns.

        >>> ds = df.to_torch(
        ...     "dataset",
        ...     dtype=pl.Float32,
        ...     label=(pl.col("lbl") * 8).cast(pl.Int16),
        ... )
        >>> ds[:2]
        (tensor([[ 1.0000,  1.5000],
                 [ 0.0000, -0.5000]]), tensor([0, 8], dtype=torch.int16))
        """
        if label is None:
            label_colnames = None
        else:
            label_colnames = []
            if isinstance(label, (str, Expr)):
                label = [label]

            for lbl in label:
                if isinstance(lbl, Expr):
                    if lbl.meta.has_multiple_outputs():
                        msg = f"label expression must have a single output; found {lbl}"
                        raise ValueError(msg)
                    label_colnames.append(lbl.meta.output_name())
                else:
                    label_colnames.append(lbl)

        if not label:
            self.labels = None
            self.features = frame.to_torch()
            self.tensors = (self.features,)
        else:
            label_frame = frame.select(*label)
            self.labels = (  # type: ignore[attr-defined]
                label_frame if len(label) > 1 else label_frame.to_series()
            ).to_torch()
            self.features = frame.select(exclude(label_colnames)).to_torch()
            self.tensors = (self.features, self.labels)  # type: ignore[assignment]

        self._n_labels = len(label_colnames or [])
        self._n_features = frame.width - self._n_labels

    def __copy__(self) -> Self:
        """Return a shallow copy of this PolarsDataset."""
        dummy_frame = DataFrame({"blank": [0]})
        dataset_copy = self.__class__(dummy_frame)
        for attr in (
            "tensors",
            "labels",
            "features",
            "_n_labels",
            "_n_features",
        ):
            setattr(dataset_copy, attr, getattr(self, attr))
        return dataset_copy

    def __repr__(self) -> str:
        """Return a string representation of the PolarsDataset."""
        return (
            f"<{type(self).__name__} "
            f"[len:{len(self)},"
            f" features:{self._n_features},"
            f" labels:{self._n_labels}"
            f"] at 0x{id(self):X}>"
        )

    def half(
        self,
        *,
        features: bool = True,
        labels: bool = True,
        memory_format: memory_format | None = None,
    ) -> Self:
        """
        Return a copy of this PolarsDataset with the numeric data converted to f16.

        Parameters
        ----------
        features
            Convert feature data to half precision (f16).
        labels
            Convert label data to half precision (f16).
        memory_format
            Desired memory format for the modified tensors; default if unset
            is `torch.preserve_format`.
        """
        ds = self.__copy__()
        if features:
            ds.features = self.features.to(torch.float16, memory_format=memory_format)
        if self.labels is not None:
            if labels:
                ds.labels = self.labels.to(torch.float16, memory_format=memory_format)
            ds.tensors = (ds.features, ds.labels)  # type: ignore[assignment]
        else:
            ds.tensors = (ds.features,)
        return ds

    @property
    def schema(self) -> dict[str, torch.dtype | None]:
        """Return the features/labels schema."""
        return {
            "features": self.features.dtype,
            "labels": self.labels.dtype if self.labels is not None else None,
        }
