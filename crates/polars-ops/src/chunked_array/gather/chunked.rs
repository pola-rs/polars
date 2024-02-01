use arrow::legacy::utils::CustomIterTools;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::utils::NoNull;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::index::ChunkId;
use polars_utils::slice::GetSaferUnchecked;

use crate::frame::IntoDf;

pub trait DfTake: IntoDf {
    /// Take elements by a slice of [`ChunkId`]s.
    /// # Safety
    /// Does not do any bound checks.
    /// `sorted` indicates if the chunks are sorted.
    unsafe fn _take_chunked_unchecked_seq(&self, idx: &[ChunkId], sorted: IsSorted) -> DataFrame {
        let cols = self
            .to_df()
            ._apply_columns(&|s| s.take_chunked_unchecked(idx, sorted));

        DataFrame::new_no_checks(cols)
    }
    /// Take elements by a slice of optional [`ChunkId`]s.
    /// # Safety
    /// Does not do any bound checks.
    unsafe fn _take_opt_chunked_unchecked_seq(&self, idx: &[Option<ChunkId>]) -> DataFrame {
        let cols = self
            .to_df()
            ._apply_columns(&|s| s.take_opt_chunked_unchecked(idx));

        DataFrame::new_no_checks(cols)
    }

    /// # Safety
    /// Doesn't perform any bound checks
    unsafe fn _take_chunked_unchecked(&self, idx: &[ChunkId], sorted: IsSorted) -> DataFrame {
        let cols = self
            .to_df()
            ._apply_columns_par(&|s| s.take_chunked_unchecked(idx, sorted));

        DataFrame::new_no_checks(cols)
    }

    /// # Safety
    /// Doesn't perform any bound checks
    unsafe fn _take_opt_chunked_unchecked(&self, idx: &[Option<ChunkId>]) -> DataFrame {
        let cols = self
            .to_df()
            ._apply_columns_par(&|s| s.take_opt_chunked_unchecked(idx));

        DataFrame::new_no_checks(cols)
    }
}

impl DfTake for DataFrame {}

/// Gather by [`ChunkId`]
pub trait TakeChunked {
    /// # Safety
    /// This function doesn't do any bound checks.
    unsafe fn take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Self;

    /// # Safety
    /// This function doesn't do any bound checks.
    unsafe fn take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Self;
}

impl TakeChunked for Series {
    unsafe fn take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Self {
        let phys = self.to_physical_repr();
        use DataType::*;
        match phys.dtype() {
            dt if dt.is_numeric() => {
                with_match_physical_numeric_polars_type!(phys.dtype(), |$T| {
                 let ca: &ChunkedArray<$T> = phys.as_ref().as_ref().as_ref();
                 ca.take_chunked_unchecked(by, sorted).into_series()
                })
            },
            Boolean => {
                let ca = phys.bool().unwrap();
                ca.take_chunked_unchecked(by, sorted).into_series()
            },
            Binary => {
                let ca = phys.binary().unwrap();
                ca.take_chunked_unchecked(by, sorted).into_series()
            },
            String => {
                let ca = phys.str().unwrap();
                ca.take_chunked_unchecked(by, sorted).into_series()
            },
            List(_) => {
                let ca = phys.list().unwrap();
                ca.take_chunked_unchecked(by, sorted).into_series()
            },
            #[cfg(feature = "dtype-array")]
            Array(_, _) => {
                let ca = phys.array().unwrap();
                ca.take_chunked_unchecked(by, sorted).into_series()
            },
            #[cfg(feature = "dtype-struct")]
            Struct(_) => {
                let ca = phys.struct_().unwrap();
                ca._apply_fields(|s| s.take_chunked_unchecked(by, sorted))
                    .into_series()
            },
            #[cfg(feature = "object")]
            Object(_, _) => take_unchecked_object(s, by, sorted),
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, _) => {
                let ca = phys.decimal().unwrap();
                let out = ca.0.take_chunked_unchecked(by, sorted);
                out.into_decimal_unchecked(ca.precision(), ca.scale())
                    .into_series()
            },
            Null => Series::new_null(self.name(), by.len()),
            _ => unreachable!(),
        }
    }

    unsafe fn take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Self {
        let phys = self.to_physical_repr();
        use DataType::*;
        match phys.dtype() {
            dt if dt.is_numeric() => {
                with_match_physical_numeric_polars_type!(phys.dtype(), |$T| {
                 let ca: &ChunkedArray<$T> = phys.as_ref().as_ref().as_ref();
                 ca.take_opt_chunked_unchecked(by).into_series()
                })
            },
            Boolean => {
                let ca = phys.bool().unwrap();
                ca.take_opt_chunked_unchecked(by).into_series()
            },
            Binary => {
                let ca = phys.binary().unwrap();
                ca.take_opt_chunked_unchecked(by).into_series()
            },
            String => {
                let ca = phys.str().unwrap();
                ca.take_opt_chunked_unchecked(by).into_series()
            },
            List(_) => {
                let ca = phys.list().unwrap();
                ca.take_opt_chunked_unchecked(by).into_series()
            },
            #[cfg(feature = "dtype-array")]
            Array(_, _) => {
                let ca = phys.array().unwrap();
                ca.take_opt_chunked_unchecked(by).into_series()
            },
            #[cfg(feature = "dtype-struct")]
            Struct(_) => {
                let ca = phys.struct_().unwrap();
                ca._apply_fields(|s| s.take_opt_chunked_unchecked(by))
                    .into_series()
            },
            #[cfg(feature = "object")]
            Object(_, _) => take_opt_unchecked_object(s, by),
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, _) => {
                let ca = phys.decimal().unwrap();
                let out = ca.0.take_opt_chunked_unchecked(by);
                out.into_decimal_unchecked(ca.precision(), ca.scale())
                    .into_series()
            },
            Null => Series::new_null(self.name(), by.len()),
            _ => unreachable!(),
        }
    }
}

trait Sealed {}
impl Sealed for UInt8Type {}
impl Sealed for UInt16Type {}
impl Sealed for UInt32Type {}
impl Sealed for UInt64Type {}
impl Sealed for Int8Type {}
impl Sealed for Int16Type {}
impl Sealed for Int32Type {}
impl Sealed for Int64Type {}
impl Sealed for Int128Type {}
impl Sealed for Float32Type {}
impl Sealed for Float64Type {}

impl<T> TakeChunked for ChunkedArray<T>
where
    T: PolarsNumericType + Sealed,
{
    unsafe fn take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Self {
        let mut ca = if self.null_count() == 0 {
            let arrs = self
                .downcast_iter()
                .map(|arr| arr.values().as_slice())
                .collect::<Vec<_>>();

            let ca: NoNull<Self> = by
                .iter()
                .map(|chunk_id| {
                    let (chunk_idx, array_idx) = chunk_id.extract();
                    let arr = arrs.get_unchecked_release(chunk_idx as usize);
                    *arr.get_unchecked_release(array_idx as usize)
                })
                .collect_trusted();

            ca.into_inner()
        } else {
            let arrs = self.downcast_iter().collect::<Vec<_>>();
            by.iter()
                .map(|chunk_id| {
                    let (chunk_idx, array_idx) = chunk_id.extract();
                    let arr = arrs.get_unchecked(chunk_idx as usize);
                    arr.get_unchecked(array_idx as usize)
                })
                .collect_trusted()
        };
        ca.rename(self.name());
        ca.set_sorted_flag(sorted);
        ca
    }

    unsafe fn take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Self {
        let arrs = self.downcast_iter().collect::<Vec<_>>();
        let mut ca: Self = by
            .iter()
            .map(|opt_idx| {
                opt_idx.and_then(|chunk_id| {
                    let (chunk_idx, array_idx) = chunk_id.extract();
                    let arr = arrs.get_unchecked_release(chunk_idx as usize);
                    arr.get_unchecked(array_idx as usize)
                })
            })
            .collect_trusted();

        ca.rename(self.name());
        ca
    }
}

impl TakeChunked for StringChunked {
    unsafe fn take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Self {
        self.as_binary()
            .take_chunked_unchecked(by, sorted)
            .to_string()
    }

    unsafe fn take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Self {
        self.as_binary().take_opt_chunked_unchecked(by).to_string()
    }
}

impl TakeChunked for BinaryOffsetChunked {
    unsafe fn take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Self {
        let arrs = self.downcast_iter().collect::<Vec<_>>();
        let mut ca: Self = by
            .iter()
            .map(|chunk_id| {
                let (chunk_idx, array_idx) = chunk_id.extract();
                let arr = arrs.get_unchecked(chunk_idx as usize);
                arr.get_unchecked(array_idx as usize)
            })
            .collect_trusted();
        ca.rename(self.name());
        ca.set_sorted_flag(sorted);
        ca
    }

    unsafe fn take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Self {
        let arrs = self.downcast_iter().collect::<Vec<_>>();
        let mut ca: Self = by
            .iter()
            .map(|opt_idx| {
                opt_idx.and_then(|chunk_id| {
                    let (chunk_idx, array_idx) = chunk_id.extract();
                    let arr = arrs.get_unchecked(chunk_idx as usize);
                    arr.get_unchecked(array_idx as usize)
                })
            })
            .collect_trusted();

        ca.rename(self.name());
        ca
    }
}

impl TakeChunked for BinaryChunked {
    unsafe fn take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Self {
        let arrs = self.downcast_iter().collect::<Vec<_>>();
        let mut ca: Self = by
            .iter()
            .map(|chunk_id| {
                let (chunk_idx, array_idx) = chunk_id.extract();
                let arr = arrs.get_unchecked(chunk_idx as usize);
                arr.get_unchecked(array_idx as usize)
            })
            .collect_trusted();
        ca.rename(self.name());
        ca.set_sorted_flag(sorted);
        ca
    }

    unsafe fn take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Self {
        let arrs = self.downcast_iter().collect::<Vec<_>>();
        let mut ca: Self = by
            .iter()
            .map(|opt_idx| {
                opt_idx.and_then(|chunk_id| {
                    let (chunk_idx, array_idx) = chunk_id.extract();
                    let arr = arrs.get_unchecked(chunk_idx as usize);
                    arr.get_unchecked(array_idx as usize)
                })
            })
            .collect_trusted();

        ca.rename(self.name());
        ca
    }
}

impl TakeChunked for BooleanChunked {
    unsafe fn take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Self {
        let arrs = self.downcast_iter().collect::<Vec<_>>();
        let mut ca: Self = by
            .iter()
            .map(|chunk_id| {
                let (chunk_idx, array_idx) = chunk_id.extract();
                let arr = arrs.get_unchecked(chunk_idx as usize);
                arr.get_unchecked(array_idx as usize)
            })
            .collect_trusted();
        ca.rename(self.name());
        ca.set_sorted_flag(sorted);
        ca
    }

    unsafe fn take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Self {
        let arrs = self.downcast_iter().collect::<Vec<_>>();
        let mut ca: Self = by
            .iter()
            .map(|opt_idx| {
                opt_idx.and_then(|chunk_id| {
                    let (chunk_idx, array_idx) = chunk_id.extract();
                    let arr = arrs.get_unchecked(chunk_idx as usize);
                    arr.get_unchecked(array_idx as usize)
                })
            })
            .collect_trusted();

        ca.rename(self.name());
        ca
    }
}

impl TakeChunked for ListChunked {
    unsafe fn take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Self {
        let arrs = self.downcast_iter().collect::<Vec<_>>();
        let mut ca: Self = by
            .iter()
            .map(|chunk_id| {
                let (chunk_idx, array_idx) = chunk_id.extract();
                let arr = arrs.get_unchecked(chunk_idx as usize);
                arr.get_unchecked(array_idx as usize)
            })
            .collect();
        ca.rename(self.name());
        ca.set_sorted_flag(sorted);
        ca
    }

    unsafe fn take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Self {
        let arrs = self.downcast_iter().collect::<Vec<_>>();
        let mut ca: Self = by
            .iter()
            .map(|opt_idx| {
                opt_idx.and_then(|chunk_id| {
                    let (chunk_idx, array_idx) = chunk_id.extract();
                    let arr = arrs.get_unchecked(chunk_idx as usize);
                    arr.get_unchecked(array_idx as usize)
                })
            })
            .collect();

        ca.rename(self.name());
        ca
    }
}

#[cfg(feature = "dtype-array")]
impl TakeChunked for ArrayChunked {
    unsafe fn take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Self {
        let arrs = self.downcast_iter().collect::<Vec<_>>();
        let iter = by.iter().map(|chunk_id| {
            let (chunk_idx, array_idx) = chunk_id.extract();
            let arr = arrs.get_unchecked(chunk_idx as usize);
            arr.get_unchecked(array_idx as usize)
        });
        let mut ca = Self::from_iter_and_args(
            iter,
            self.width(),
            by.len(),
            Some(self.inner_dtype()),
            self.name(),
        );
        ca.set_sorted_flag(sorted);
        ca
    }

    unsafe fn take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Self {
        let arrs = self.downcast_iter().collect::<Vec<_>>();
        let iter = by.iter().map(|opt_idx| {
            opt_idx.and_then(|chunk_id| {
                let (chunk_idx, array_idx) = chunk_id.extract();
                let arr = arrs.get_unchecked(chunk_idx as usize);
                arr.get_unchecked(array_idx as usize)
            })
        });

        Self::from_iter_and_args(
            iter,
            self.width(),
            by.len(),
            Some(self.inner_dtype()),
            self.name(),
        )
    }
}

#[cfg(feature = "object")]
unsafe fn take_unchecked_object(s: &Series, by: &[ChunkId], _sorted: IsSorted) -> Series {
    let DataType::Object(_, reg) = s.dtype() else {
        unreachable!()
    };
    let reg = reg.unwrap();
    let mut builder = (*reg.builder_constructor)(s.name(), by.len());

    by.iter().map(|chunk_id| {
        let (chunk_idx, array_idx) = chunk_id.extract();
        let object = s.get_object_chunked_unchecked(chunk_idx as usize, array_idx as usize);
        builder.append_option(object.map(|v| v.as_any()))
    });
    builder.to_series()
}

#[cfg(feature = "object")]
unsafe fn take_opt_unchecked_object(
    s: &Series,
    by: &[Option<ChunkId>],
    _sorted: IsSorted,
) -> Series {
    let DataType::Object(_, reg) = s.dtype() else {
        unreachable!()
    };
    let reg = reg.unwrap();
    let mut builder = (*reg.builder_constructor)(s.name(), by.len());

    by.iter().map(|chunk_id| match chunk_id {
        None => builder.append_null(),
        Some(chunk_id) => {
            let (chunk_idx, array_idx) = chunk_id.extract();
            let object = s.get_object_chunked_unchecked(chunk_idx as usize, array_idx as usize);
            builder.append_option(object.map(|v| v.as_any()))
        },
    });
    builder.to_series()
}
