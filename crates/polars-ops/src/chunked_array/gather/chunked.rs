use polars_core::prelude::*;
use polars_core::series::IsSorted;
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
            Object(_, _) => take_unchecked_object(&phys, by, sorted),
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
            Object(_, _) => take_opt_unchecked_object(&phys, by),
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

impl<T> TakeChunked for ChunkedArray<T>
where
    T: PolarsDataType,
{
    unsafe fn take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Self {
        let arrow_dtype = self.dtype().to_arrow(true);

        let mut out = if let Some(iter) = self.downcast_slices() {
            let targets = iter.collect::<Vec<_>>();
            let iter = by.iter().map(|chunk_id| {
                let (chunk_idx, array_idx) = chunk_id.extract();
                let vals = targets.get_unchecked_release(chunk_idx as usize);
                vals.get_unchecked_release(array_idx as usize).clone()
            });

            let arr = iter.collect_arr_trusted_with_dtype(arrow_dtype);
            ChunkedArray::with_chunk(self.name(), arr)
        } else {
            let targets = self.downcast_iter().collect::<Vec<_>>();
            let iter = by.iter().map(|chunk_id| {
                let (chunk_idx, array_idx) = chunk_id.extract();
                let vals = targets.get_unchecked_release(chunk_idx as usize);
                vals.get_unchecked(array_idx as usize)
            });
            let arr = iter.collect_arr_trusted_with_dtype(arrow_dtype);
            ChunkedArray::with_chunk(self.name(), arr)
        };
        out.set_sorted_flag(sorted);
        out
    }

    unsafe fn take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Self {
        let arrow_dtype = self.dtype().to_arrow(true);

        if let Some(iter) = self.downcast_slices() {
            let targets = iter.collect::<Vec<_>>();
            let arr = by
                .iter()
                .map(|chunk_id| {
                    chunk_id.map(|chunk_id| {
                        let (chunk_idx, array_idx) = chunk_id.extract();
                        let vals = *targets.get_unchecked_release(chunk_idx as usize);
                        vals.get_unchecked_release(array_idx as usize).clone()
                    })
                })
                .collect_arr_trusted_with_dtype(arrow_dtype);

            ChunkedArray::with_chunk(self.name(), arr)
        } else {
            let targets = self.downcast_iter().collect::<Vec<_>>();
            let arr = by
                .iter()
                .map(|chunk_id| {
                    chunk_id.and_then(|chunk_id| {
                        let (chunk_idx, array_idx) = chunk_id.extract();
                        let vals = *targets.get_unchecked_release(chunk_idx as usize);
                        vals.get_unchecked(array_idx as usize)
                    })
                })
                .collect_arr_trusted_with_dtype(arrow_dtype);

            ChunkedArray::with_chunk(self.name(), arr)
        }
    }
}

#[cfg(feature = "object")]
unsafe fn take_unchecked_object(s: &Series, by: &[ChunkId], _sorted: IsSorted) -> Series {
    let DataType::Object(_, reg) = s.dtype() else {
        unreachable!()
    };
    let reg = reg.as_ref().unwrap();
    let mut builder = (*reg.builder_constructor)(s.name(), by.len());

    by.iter().for_each(|chunk_id| {
        let (chunk_idx, array_idx) = chunk_id.extract();
        let object = s.get_object_chunked_unchecked(chunk_idx as usize, array_idx as usize);
        builder.append_option(object.map(|v| v.as_any()))
    });
    builder.to_series()
}

#[cfg(feature = "object")]
unsafe fn take_opt_unchecked_object(s: &Series, by: &[Option<ChunkId>]) -> Series {
    let DataType::Object(_, reg) = s.dtype() else {
        unreachable!()
    };
    let reg = reg.as_ref().unwrap();
    let mut builder = (*reg.builder_constructor)(s.name(), by.len());

    by.iter().for_each(|chunk_id| match chunk_id {
        None => builder.append_null(),
        Some(chunk_id) => {
            let (chunk_idx, array_idx) = chunk_id.extract();
            let object = s.get_object_chunked_unchecked(chunk_idx as usize, array_idx as usize);
            builder.append_option(object.map(|v| v.as_any()))
        },
    });
    builder.to_series()
}
