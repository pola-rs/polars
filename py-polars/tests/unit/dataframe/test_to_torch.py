from __future__ import annotations

from typing import Any

import pytest

import polars as pl
import polars.selectors as cs
from polars.dependencies import _lazy_import

# don't import torch until an actual test is triggered (the decorator already
# ensures the tests aren't run locally, this will skip premature local import)
torch, _ = _lazy_import("torch")


@pytest.fixture()
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [1, 2, 2, 3],
            "y": [True, False, True, False],
            "z": [1.5, -0.5, 0.0, -2.0],
        },
        schema_overrides={"x": pl.Int8, "z": pl.Float32},
    )


@pytest.mark.ci_only()
class TestTorchIntegration:
    """Test coverage for `to_torch` conversions and `polars.ml.torch` classes."""

    def assert_tensor(self, actual: Any, expected: Any) -> None:
        torch.testing.assert_close(actual, expected)

    def test_to_torch_series(
        self,
    ) -> None:
        s = pl.Series("x", [1, 2, 3, 4], dtype=pl.Int8)
        t = s.to_torch()

        assert list(t.shape) == [4]
        self.assert_tensor(t, torch.tensor([1, 2, 3, 4], dtype=torch.int8))

        # note: torch doesn't natively support uint16/32/64.
        # confirm that we export to a suitable signed integer type
        s = s.cast(pl.UInt16)
        t = s.to_torch()
        self.assert_tensor(t, torch.tensor([1, 2, 3, 4], dtype=torch.int32))

        for dtype in (pl.UInt32, pl.UInt64):
            t = s.cast(dtype).to_torch()
            self.assert_tensor(t, torch.tensor([1, 2, 3, 4], dtype=torch.int64))

    def test_to_torch_tensor(self, df: pl.DataFrame) -> None:
        t1 = df.to_torch()
        t2 = df.to_torch("tensor")

        assert list(t1.shape) == [4, 3]
        assert (t1 == t2).all().item() is True

    def test_to_torch_dict(self, df: pl.DataFrame) -> None:
        td = df.to_torch("dict")

        assert list(td.keys()) == ["x", "y", "z"]

        self.assert_tensor(td["x"], torch.tensor([1, 2, 2, 3], dtype=torch.int8))
        self.assert_tensor(
            td["y"], torch.tensor([True, False, True, False], dtype=torch.bool)
        )
        self.assert_tensor(
            td["z"], torch.tensor([1.5, -0.5, 0.0, -2.0], dtype=torch.float32)
        )

    def test_to_torch_dataset(self, df: pl.DataFrame) -> None:
        ds = df.to_torch("dataset", dtype=pl.Float64)

        assert len(ds) == 4
        assert isinstance(ds, torch.utils.data.Dataset)
        assert repr(ds).startswith("<PolarsDataset [len:4, features:3, labels:0] at 0x")

        ts = ds[0]
        assert isinstance(ts, tuple)
        assert len(ts) == 1
        self.assert_tensor(ts[0], torch.tensor([1.0, 1.0, 1.5], dtype=torch.float64))

    def test_to_torch_dataset_feature_reorder(self, df: pl.DataFrame) -> None:
        ds = df.to_torch("dataset", label="x", features=["z", "y"])
        self.assert_tensor(
            torch.tensor(
                [
                    [1.5000, 1.0000],
                    [-0.5000, 0.0000],
                    [0.0000, 1.0000],
                    [-2.0000, 0.0000],
                ]
            ),
            ds.features,
        )
        self.assert_tensor(torch.tensor([1, 2, 2, 3], dtype=torch.int8), ds.labels)

    def test_to_torch_dataset_feature_subset(self, df: pl.DataFrame) -> None:
        ds = df.to_torch("dataset", label="x", features=["z"])
        self.assert_tensor(
            torch.tensor([[1.5000], [-0.5000], [0.0000], [-2.0000]]),
            ds.features,
        )
        self.assert_tensor(torch.tensor([1, 2, 2, 3], dtype=torch.int8), ds.labels)

    def test_to_torch_dataset_index_slice(self, df: pl.DataFrame) -> None:
        ds = df.to_torch("dataset")
        ts = ds[1:3]

        expected = (
            torch.tensor([[2.0000, 0.0000, -0.5000], [2.0000, 1.0000, 0.0000]]),
        )
        self.assert_tensor(expected, ts)

        ts = ds[::2]
        expected = (torch.tensor([[1.0000, 1.0000, 1.5000], [2.0, 1.0, 0.0]]),)
        self.assert_tensor(expected, ts)

    @pytest.mark.parametrize(
        "index",
        [
            [0, 3],
            range(0, 4, 3),
            slice(0, 4, 3),
        ],
    )
    def test_to_torch_dataset_index_multi(self, index: Any, df: pl.DataFrame) -> None:
        ds = df.to_torch("dataset")
        ts = ds[index]

        expected = (torch.tensor([[1.0, 1.0, 1.5], [3.0, 0.0, -2.0]]),)
        self.assert_tensor(expected, ts)
        assert ds.schema == {"features": torch.float32, "labels": None}

    def test_to_torch_dataset_index_range(self, df: pl.DataFrame) -> None:
        ds = df.to_torch("dataset")
        ts = ds[range(3, 0, -1)]

        expected = (
            torch.tensor([[3.0, 0.0, -2.0], [2.0, 1.0, 0.0], [2.0, 0.0, -0.5]]),
        )
        self.assert_tensor(expected, ts)

    def test_to_dataset_half_precision(self, df: pl.DataFrame) -> None:
        ds = df.to_torch("dataset", label="x")
        assert ds.schema == {"features": torch.float32, "labels": torch.int8}

        dsf16 = ds.half()
        assert dsf16.schema == {"features": torch.float16, "labels": torch.float16}

        # half precision across all data
        ts = dsf16[:3:2]
        expected = (
            torch.tensor([[1.0000, 1.5000], [1.0000, 0.0000]], dtype=torch.float16),
            torch.tensor([1.0, 2.0], dtype=torch.float16),
        )
        self.assert_tensor(expected, ts)

        # only apply half precision to the feature data
        dsf16 = ds.half(labels=False)
        assert dsf16.schema == {"features": torch.float16, "labels": torch.int8}

        ts = dsf16[:3:2]
        expected = (
            torch.tensor([[1.0000, 1.5000], [1.0000, 0.0000]], dtype=torch.float16),
            torch.tensor([1, 2], dtype=torch.int8),
        )
        self.assert_tensor(expected, ts)

        # only apply half precision to the label data
        dsf16 = ds.half(features=False)
        assert dsf16.schema == {"features": torch.float32, "labels": torch.float16}

        ts = dsf16[:3:2]
        expected = (
            torch.tensor([[1.0000, 1.5000], [1.0000, 0.0000]], dtype=torch.float32),
            torch.tensor([1.0, 2.0], dtype=torch.float16),
        )
        self.assert_tensor(expected, ts)

        # no labels
        dsf16 = df.to_torch("dataset").half()
        assert dsf16.schema == {"features": torch.float16, "labels": None}

        ts = dsf16[:3:2]
        expected = (  # type: ignore[assignment]
            torch.tensor(
                data=[[1.0000, 1.0000, 1.5000], [2.0000, 1.0000, 0.0000]],
                dtype=torch.float16,
            ),
        )
        self.assert_tensor(expected, ts)

    @pytest.mark.parametrize(
        ("label", "features"),
        [
            ("x", None),
            ("x", ["y", "z"]),
            (cs.by_dtype(pl.INTEGER_DTYPES), ~cs.by_dtype(pl.INTEGER_DTYPES)),
        ],
    )
    def test_to_torch_labelled_dataset(
        self, label: Any, features: Any, df: pl.DataFrame
    ) -> None:
        ds = df.to_torch("dataset", label=label, features=features)
        ts = next(iter(torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)))

        expected = [
            torch.tensor([[1.0, 1.5], [0.0, -0.5]]),
            torch.tensor([1, 2], dtype=torch.int8),
        ]
        assert len(ts) == len(expected)
        for actual, exp in zip(ts, expected):
            self.assert_tensor(exp, actual)

    def test_to_torch_labelled_dataset_expr(self, df: pl.DataFrame) -> None:
        ds = df.to_torch(
            "dataset",
            dtype=pl.Float64,
            label=(pl.col("x") * 8).cast(pl.Int16),
        )
        dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
        for data in (tuple(ds[:2]), tuple(next(iter(dl)))):
            expected = (
                torch.tensor(
                    [[1.0000, 1.5000], [0.0000, -0.5000]], dtype=torch.float64
                ),
                torch.tensor([8, 16], dtype=torch.int16),
            )
            assert len(data) == len(expected)
            for actual, exp in zip(data, expected):
                self.assert_tensor(exp, actual)

    def test_to_torch_labelled_dataset_multi(self, df: pl.DataFrame) -> None:
        ds = df.to_torch("dataset", label=["x", "y"])
        dl = torch.utils.data.DataLoader(ds, batch_size=3, shuffle=False)
        ts = list(dl)

        expected = [
            [
                torch.tensor([[1.5000], [-0.5000], [0.0000]]),
                torch.tensor([[1, 1], [2, 0], [2, 1]], dtype=torch.int8),
            ],
            [
                torch.tensor([[-2.0]]),
                torch.tensor([[3, 0]], dtype=torch.int8),
            ],
        ]
        assert len(ts) == len(expected)

        for actual, exp in zip(ts, expected):
            assert len(actual) == len(exp)
            for a, e in zip(actual, exp):
                self.assert_tensor(e, a)

    def test_misc_errors(self, df: pl.DataFrame) -> None:
        ds = df.to_torch("dataset")

        with pytest.raises(
            ValueError,
            match="invalid `return_type`: 'stroopwafel'",
        ):
            _res0 = df.to_torch("stroopwafel")  # type: ignore[call-overload]

        with pytest.raises(
            ValueError,
            match="does not support u16, u32, or u64 dtypes",
        ):
            _res1 = df.to_torch(dtype=pl.UInt16)

        with pytest.raises(
            IndexError,
            match="tensors used as indices must be long, int",
        ):
            _res2 = ds[torch.tensor([0, 3], dtype=torch.complex64)]
