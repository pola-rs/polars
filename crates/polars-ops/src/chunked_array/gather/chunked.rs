use std::borrow::Cow;
use std::fmt::Debug;

use arrow::array::{Array, BinaryViewArray, View};
use arrow::bitmap::MutableBitmap;
use arrow::buffer::Buffer;
use arrow::legacy::trusted_len::TrustedLenPush;
use polars_core::prelude::gather::_update_gather_sorted_flag;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::slice::GetSaferUnchecked;

use crate::frame::IntoDf;

pub trait DfTake: IntoDf {
    /// Take elements by a slice of [`ChunkId`]s.
    ///
    /// # Safety
    /// Does not do any bound checks.
    /// `sorted` indicates if the chunks are sorted.
    unsafe fn _take_chunked_unchecked_seq(&self, idx: &[ChunkId], sorted: IsSorted) -> DataFrame {
        let cols = self
            .to_df()
            ._apply_columns(&|s| s.take_chunked_unchecked(idx, sorted));

        unsafe { DataFrame::new_no_checks(cols) }
    }
    /// Take elements by a slice of optional [`ChunkId`]s.
    ///
    /// # Safety
    /// Does not do any bound checks.
    unsafe fn _take_opt_chunked_unchecked_seq(&self, idx: &[NullableChunkId]) -> DataFrame {
        let cols = self
            .to_df()
            ._apply_columns(&|s| s.take_opt_chunked_unchecked(idx));

        unsafe { DataFrame::new_no_checks(cols) }
    }

    /// # Safety
    /// Doesn't perform any bound checks
    unsafe fn _take_chunked_unchecked(&self, idx: &[ChunkId], sorted: IsSorted) -> DataFrame {
        let cols = self
            .to_df()
            ._apply_columns_par(&|s| s.take_chunked_unchecked(idx, sorted));

        unsafe { DataFrame::new_no_checks(cols) }
    }

    /// # Safety
    /// Doesn't perform any bound checks
    ///
    /// Check for null state in `ChunkId`.
    unsafe fn _take_opt_chunked_unchecked(&self, idx: &[ChunkId]) -> DataFrame {
        let cols = self
            .to_df()
            ._apply_columns_par(&|s| s.take_opt_chunked_unchecked(idx));

        unsafe { DataFrame::new_no_checks(cols) }
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
    unsafe fn take_opt_chunked_unchecked(&self, by: &[ChunkId]) -> Self;
}

fn prepare_series(s: &Series) -> Cow<Series> {
    let phys = if s.dtype().is_nested() {
        Cow::Borrowed(s)
    } else {
        s.to_physical_repr()
    };
    // If this is hit the cast rechunked the data and the gather will OOB
    assert_eq!(
        phys.chunks().len(),
        s.chunks().len(),
        "implementation error"
    );
    phys
}

impl TakeChunked for Series {
    unsafe fn take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Self {
        let phys = prepare_series(self);
        use DataType::*;
        let out = match phys.dtype() {
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
                let out = take_unchecked_binview(ca, by, sorted);
                out.into_series()
            },
            String => {
                let ca = phys.str().unwrap();
                let ca = ca.as_binary();
                let out = take_unchecked_binview(&ca, by, sorted);
                out.to_string_unchecked().into_series()
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
                    .expect("infallible")
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
        };
        unsafe { out.cast_unchecked(self.dtype()).unwrap() }
    }

    /// Take function that checks of null state in `ChunkIdx`.
    unsafe fn take_opt_chunked_unchecked(&self, by: &[NullableChunkId]) -> Self {
        let phys = prepare_series(self);
        use DataType::*;
        let out = match phys.dtype() {
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
                let out = take_unchecked_binview_opt(ca, by);
                out.into_series()
            },
            String => {
                let ca = phys.str().unwrap();
                let ca = ca.as_binary();
                let out = take_unchecked_binview_opt(&ca, by);
                out.to_string_unchecked().into_series()
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
                    .expect("infallible")
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
        };
        unsafe { out.cast_unchecked(self.dtype()).unwrap() }
    }
}

impl<T> TakeChunked for ChunkedArray<T>
where
    T: PolarsDataType,
    T::Array: Debug,
{
    unsafe fn take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Self {
        let arrow_dtype = self.dtype().to_arrow(CompatLevel::newest());

        let mut out = if let Some(iter) = self.downcast_slices() {
            let targets = iter.collect::<Vec<_>>();
            let iter = by.iter().map(|chunk_id| {
                debug_assert!(
                    !chunk_id.is_null(),
                    "null chunks should not hit this branch"
                );
                let (chunk_idx, array_idx) = chunk_id.extract();
                let vals = targets.get_unchecked_release(chunk_idx as usize);
                vals.get_unchecked_release(array_idx as usize).clone()
            });

            let arr = iter.collect_arr_trusted_with_dtype(arrow_dtype);
            ChunkedArray::with_chunk(self.name(), arr)
        } else {
            let targets = self.downcast_iter().collect::<Vec<_>>();
            let iter = by.iter().map(|chunk_id| {
                debug_assert!(
                    !chunk_id.is_null(),
                    "null chunks should not hit this branch"
                );
                let (chunk_idx, array_idx) = chunk_id.extract();
                let vals = targets.get_unchecked_release(chunk_idx as usize);
                vals.get_unchecked(array_idx as usize)
            });
            let arr = iter.collect_arr_trusted_with_dtype(arrow_dtype);
            ChunkedArray::with_chunk(self.name(), arr)
        };
        let sorted_flag = _update_gather_sorted_flag(self.is_sorted_flag(), sorted);
        out.set_sorted_flag(sorted_flag);
        out
    }

    // Take function that checks of null state in `ChunkIdx`.
    unsafe fn take_opt_chunked_unchecked(&self, by: &[NullableChunkId]) -> Self {
        let arrow_dtype = self.dtype().to_arrow(CompatLevel::newest());

        if let Some(iter) = self.downcast_slices() {
            let targets = iter.collect::<Vec<_>>();
            let arr = by
                .iter()
                .map(|chunk_id| {
                    if chunk_id.is_null() {
                        None
                    } else {
                        let (chunk_idx, array_idx) = chunk_id.extract();
                        let vals = *targets.get_unchecked_release(chunk_idx as usize);
                        Some(vals.get_unchecked_release(array_idx as usize).clone())
                    }
                })
                .collect_arr_trusted_with_dtype(arrow_dtype);

            ChunkedArray::with_chunk(self.name(), arr)
        } else {
            let targets = self.downcast_iter().collect::<Vec<_>>();
            let arr = by
                .iter()
                .map(|chunk_id| {
                    if chunk_id.is_null() {
                        None
                    } else {
                        let (chunk_idx, array_idx) = chunk_id.extract();
                        let vals = *targets.get_unchecked_release(chunk_idx as usize);
                        vals.get_unchecked(array_idx as usize)
                    }
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
unsafe fn take_opt_unchecked_object(s: &Series, by: &[NullableChunkId]) -> Series {
    let DataType::Object(_, reg) = s.dtype() else {
        unreachable!()
    };
    let reg = reg.as_ref().unwrap();
    let mut builder = (*reg.builder_constructor)(s.name(), by.len());

    by.iter().for_each(|chunk_id| {
        if chunk_id.is_null() {
            builder.append_null()
        } else {
            let (chunk_idx, array_idx) = chunk_id.extract();
            let object = s.get_object_chunked_unchecked(chunk_idx as usize, array_idx as usize);
            builder.append_option(object.map(|v| v.as_any()))
        }
    });
    builder.to_series()
}

#[allow(clippy::unnecessary_cast)]
#[inline(always)]
unsafe fn rewrite_view(mut view: View, chunk_idx: IdxSize, buffer_offsets: &[u32]) -> View {
    if view.length > 12 {
        let base_offset = *buffer_offsets.get_unchecked_release(chunk_idx as usize);
        view.buffer_idx += base_offset;
    }
    view
}

fn create_buffer_offsets(ca: &BinaryChunked) -> Vec<u32> {
    let mut buffer_offsets = Vec::with_capacity(ca.chunks().len() + 1);
    let mut cumsum = 0u32;
    buffer_offsets.push(cumsum);
    buffer_offsets.extend(ca.downcast_iter().map(|arr| {
        cumsum += arr.data_buffers().len() as u32;
        cumsum
    }));
    buffer_offsets
}

#[allow(clippy::unnecessary_cast)]
unsafe fn take_unchecked_binview(
    ca: &BinaryChunked,
    by: &[ChunkId],
    sorted: IsSorted,
) -> BinaryChunked {
    let views = ca
        .downcast_iter()
        .map(|arr| arr.views().as_slice())
        .collect::<Vec<_>>();
    let buffer_offsets = create_buffer_offsets(ca);

    let buffers: Arc<[Buffer<u8>]> = ca
        .downcast_iter()
        .flat_map(|arr| arr.data_buffers().as_ref())
        .cloned()
        .collect();

    let (views, validity) = if ca.null_count() == 0 {
        let views = by
            .iter()
            .map(|chunk_id| {
                let (chunk_idx, array_idx) = chunk_id.extract();
                let array_idx = array_idx as usize;

                let target = *views.get_unchecked_release(chunk_idx as usize);
                let view = *target.get_unchecked_release(array_idx);
                rewrite_view(view, chunk_idx, &buffer_offsets)
            })
            .collect::<Vec<_>>();

        (views, None)
    } else {
        let targets = ca.downcast_iter().collect::<Vec<_>>();

        let mut mut_views = Vec::with_capacity(by.len());
        let mut validity = MutableBitmap::with_capacity(by.len());

        for id in by.iter() {
            let (chunk_idx, array_idx) = id.extract();
            let array_idx = array_idx as usize;

            let target = *targets.get_unchecked_release(chunk_idx as usize);
            if target.is_null_unchecked(array_idx) {
                mut_views.push_unchecked(View::default());
                validity.push_unchecked(false)
            } else {
                let target = *views.get_unchecked_release(chunk_idx as usize);
                let view = *target.get_unchecked_release(array_idx);
                let view = rewrite_view(view, chunk_idx, &buffer_offsets);
                mut_views.push_unchecked(view);
                validity.push_unchecked(true)
            }
        }

        (mut_views, Some(validity.freeze()))
    };

    let arr = BinaryViewArray::new_unchecked_unknown_md(
        ArrowDataType::BinaryView,
        views.into(),
        buffers,
        validity,
        None,
    )
    .maybe_gc();

    let mut out = BinaryChunked::with_chunk(ca.name(), arr);
    let sorted_flag = _update_gather_sorted_flag(ca.is_sorted_flag(), sorted);
    out.set_sorted_flag(sorted_flag);
    out
}

unsafe fn take_unchecked_binview_opt(ca: &BinaryChunked, by: &[NullableChunkId]) -> BinaryChunked {
    let views = ca
        .downcast_iter()
        .map(|arr| arr.views().as_slice())
        .collect::<Vec<_>>();
    let buffers: Arc<[Buffer<u8>]> = ca
        .downcast_iter()
        .flat_map(|arr| arr.data_buffers().as_ref())
        .cloned()
        .collect();
    let buffer_offsets = create_buffer_offsets(ca);

    let targets = ca.downcast_iter().collect::<Vec<_>>();

    let mut mut_views = Vec::with_capacity(by.len());
    let mut validity = MutableBitmap::with_capacity(by.len());

    let (views, validity) = if ca.null_count() == 0 {
        for id in by.iter() {
            if id.is_null() {
                mut_views.push_unchecked(View::default());
                validity.push_unchecked(false)
            } else {
                let (chunk_idx, array_idx) = id.extract();
                let array_idx = array_idx as usize;

                let target = *views.get_unchecked_release(chunk_idx as usize);
                let view = *target.get_unchecked_release(array_idx);
                let view = rewrite_view(view, chunk_idx, &buffer_offsets);

                mut_views.push_unchecked(view);
                validity.push_unchecked(true)
            }
        }
        (mut_views, Some(validity.freeze()))
    } else {
        for id in by.iter() {
            if id.is_null() {
                mut_views.push_unchecked(View::default());
                validity.push_unchecked(false)
            } else {
                let (chunk_idx, array_idx) = id.extract();
                let array_idx = array_idx as usize;

                let target = *targets.get_unchecked_release(chunk_idx as usize);
                if target.is_null_unchecked(array_idx) {
                    mut_views.push_unchecked(View::default());
                    validity.push_unchecked(false)
                } else {
                    let target = *views.get_unchecked_release(chunk_idx as usize);
                    let view = *target.get_unchecked_release(array_idx);
                    let view = rewrite_view(view, chunk_idx, &buffer_offsets);
                    mut_views.push_unchecked(view);
                    validity.push_unchecked(true);
                }
            }
        }

        (mut_views, Some(validity.freeze()))
    };

    let arr = BinaryViewArray::new_unchecked_unknown_md(
        ArrowDataType::BinaryView,
        views.into(),
        buffers,
        validity,
        None,
    )
    .maybe_gc();

    BinaryChunked::with_chunk(ca.name(), arr)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_binview_chunked_gather() {
        unsafe {
            // # Series without nulls;
            let mut s_1 = Series::new(
                "a",
                &["1 loooooooooooong string", "2 loooooooooooong string"],
            );
            let s_2 = Series::new(
                "a",
                &["11 loooooooooooong string", "22 loooooooooooong string"],
            );
            let s_3 = Series::new(
                "a",
                &[
                    "111 loooooooooooong string",
                    "222 loooooooooooong string",
                    "small", // this tests we don't mess with the inlined view
                ],
            );
            s_1.append(&s_2).unwrap();
            s_1.append(&s_3).unwrap();

            assert_eq!(s_1.n_chunks(), 3);

            // ## Ids without nulls;
            let by = [
                ChunkId::store(0, 0),
                ChunkId::store(0, 1),
                ChunkId::store(1, 1),
                ChunkId::store(1, 0),
                ChunkId::store(2, 0),
                ChunkId::store(2, 1),
                ChunkId::store(2, 2),
            ];

            let out = s_1.take_chunked_unchecked(&by, IsSorted::Not);
            let idx = IdxCa::new("", [0, 1, 3, 2, 4, 5, 6]);
            let expected = s_1.rechunk().take(&idx).unwrap();
            assert!(out.equals(&expected));

            // ## Ids with nulls;
            let by: [ChunkId; 4] = [
                ChunkId::null(),
                ChunkId::store(0, 1),
                ChunkId::store(1, 1),
                ChunkId::store(1, 0),
            ];
            let out = s_1.take_opt_chunked_unchecked(&by);

            let idx = IdxCa::new("", [None, Some(1), Some(3), Some(2)]);
            let expected = s_1.rechunk().take(&idx).unwrap();
            assert!(out.equals_missing(&expected));

            // # Series with nulls;
            let mut s_1 = Series::new(
                "a",
                &["1 loooooooooooong string 1", "2 loooooooooooong string 2"],
            );
            let s_2 = Series::new("a", &[Some("11 loooooooooooong string 11"), None]);
            s_1.append(&s_2).unwrap();

            // ## Ids without nulls;
            let by = [
                ChunkId::store(0, 0),
                ChunkId::store(0, 1),
                ChunkId::store(1, 1),
                ChunkId::store(1, 0),
            ];

            let out = s_1.take_chunked_unchecked(&by, IsSorted::Not);
            let idx = IdxCa::new("", [0, 1, 3, 2]);
            let expected = s_1.rechunk().take(&idx).unwrap();
            assert!(out.equals_missing(&expected));

            // ## Ids with nulls;
            let by: [ChunkId; 4] = [
                ChunkId::null(),
                ChunkId::store(0, 1),
                ChunkId::store(1, 1),
                ChunkId::store(1, 0),
            ];
            let out = s_1.take_opt_chunked_unchecked(&by);

            let idx = IdxCa::new("", [None, Some(1), Some(3), Some(2)]);
            let expected = s_1.rechunk().take(&idx).unwrap();
            assert!(out.equals_missing(&expected));
        }
    }
}
