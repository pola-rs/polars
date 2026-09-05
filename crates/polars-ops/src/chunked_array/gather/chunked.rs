#![allow(unsafe_op_in_unsafe_fn)]
use std::fmt::Debug;

use arrow::bitmap::BitmapBuilder;
use polars_array::builder::{PlArrayBuilder, ShareStrategy, builder_like};
use polars_core::prelude::gather::_update_gather_sorted_flag;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::utils::Container;
use polars_core::{with_match_categorical_physical_type, with_match_physical_numeric_polars_type};

use crate::frame::IntoDf;

/// Gather by [`ChunkId`]
pub trait TakeChunked {
    /// Gathers elements from a ChunkedArray, specifying for each element a
    /// chunk index and index within that chunk through ChunkId. If
    /// avoid_sharing is true the returned data should not share references
    /// with the original array (like shared buffers in views).
    ///
    /// # Safety
    /// This function doesn't do any bound checks.
    unsafe fn take_chunked_unchecked<const B: u64>(
        &self,
        by: &[ChunkId<B>],
        sorted: IsSorted,
        avoid_sharing: bool,
    ) -> Self;

    /// # Safety
    /// This function doesn't do any bound checks.
    unsafe fn take_opt_chunked_unchecked<const B: u64>(
        &self,
        by: &[ChunkId<B>],
        avoid_sharing: bool,
    ) -> Self;
}

impl TakeChunked for DataFrame {
    /// Take elements by a slice of [`ChunkId`]s.
    ///
    /// # Safety
    /// Does not do any bound checks.
    /// `sorted` indicates if the chunks are sorted.
    unsafe fn take_chunked_unchecked<const B: u64>(
        &self,
        idx: &[ChunkId<B>],
        sorted: IsSorted,
        avoid_sharing: bool,
    ) -> DataFrame {
        let cols = self
            .to_df()
            .apply_columns(|s| s.take_chunked_unchecked(idx, sorted, avoid_sharing));

        unsafe { DataFrame::new_unchecked_infer_height(cols) }
    }

    /// Take elements by a slice of optional [`ChunkId`]s.
    ///
    /// # Safety
    /// Does not do any bound checks.
    unsafe fn take_opt_chunked_unchecked<const B: u64>(
        &self,
        idx: &[ChunkId<B>],
        avoid_sharing: bool,
    ) -> DataFrame {
        let cols = self
            .to_df()
            .apply_columns(|s| s.take_opt_chunked_unchecked(idx, avoid_sharing));

        unsafe { DataFrame::new_unchecked_infer_height(cols) }
    }
}

pub trait TakeChunkedHorPar: IntoDf {
    /// # Safety
    /// Doesn't perform any bound checks
    unsafe fn _take_chunked_unchecked_hor_par<const B: u64>(
        &self,
        idx: &[ChunkId<B>],
        sorted: IsSorted,
    ) -> DataFrame {
        let cols = self
            .to_df()
            .apply_columns_par(|s| s.take_chunked_unchecked(idx, sorted, false));

        unsafe { DataFrame::new_unchecked_infer_height(cols) }
    }

    /// # Safety
    /// Doesn't perform any bound checks
    ///
    /// Check for null state in `ChunkId`.
    unsafe fn _take_opt_chunked_unchecked_hor_par<const B: u64>(
        &self,
        idx: &[ChunkId<B>],
    ) -> DataFrame {
        let cols = self
            .to_df()
            .apply_columns_par(|s| s.take_opt_chunked_unchecked(idx, false));

        unsafe { DataFrame::new_unchecked_infer_height(cols) }
    }
}

impl TakeChunkedHorPar for DataFrame {}

impl TakeChunked for Column {
    unsafe fn take_chunked_unchecked<const B: u64>(
        &self,
        by: &[ChunkId<B>],
        sorted: IsSorted,
        avoid_sharing: bool,
    ) -> Self {
        // @scalar-opt
        let s = self.as_materialized_series();
        let s = unsafe { s.take_chunked_unchecked(by, sorted, avoid_sharing) };
        s.into_column()
    }

    unsafe fn take_opt_chunked_unchecked<const B: u64>(
        &self,
        by: &[ChunkId<B>],
        avoid_sharing: bool,
    ) -> Self {
        // @scalar-opt
        let s = self.as_materialized_series();
        let s = unsafe { s.take_opt_chunked_unchecked(by, avoid_sharing) };
        s.into_column()
    }
}

impl TakeChunked for Series {
    unsafe fn take_chunked_unchecked<const B: u64>(
        &self,
        by: &[ChunkId<B>],
        sorted: IsSorted,
        avoid_sharing: bool,
    ) -> Self {
        use DataType::*;
        match self.dtype() {
            dt if dt.is_primitive_numeric() => {
                with_match_physical_numeric_polars_type!(self.dtype(), |$T| {
                    let ca: &ChunkedArray<$T> = self.as_ref().as_ref().as_ref();
                    ca.take_chunked_unchecked(by, sorted, avoid_sharing).into_series()
                })
            },
            Boolean => {
                let ca = self.bool().unwrap();
                ca.take_chunked_unchecked(by, sorted, avoid_sharing)
                    .into_series()
            },
            Binary => {
                let ca = self.binary().unwrap();
                ca.take_chunked_unchecked(by, sorted, avoid_sharing)
                    .into_series()
            },
            String => {
                let ca = self.str().unwrap();
                ca.take_chunked_unchecked(by, sorted, avoid_sharing)
                    .into_series()
            },
            List(_) => {
                let ca = self.list().unwrap();
                ca.take_chunked_unchecked(by, sorted, avoid_sharing)
                    .into_series()
            },
            #[cfg(feature = "dtype-array")]
            Array(_, _) => {
                let ca = self.array().unwrap();
                ca.take_chunked_unchecked(by, sorted, avoid_sharing)
                    .into_series()
            },
            #[cfg(feature = "dtype-struct")]
            Struct(_) => {
                let ca = self.struct_().unwrap();
                take_chunked_unchecked_struct(ca, by, sorted, avoid_sharing).into_series()
            },
            #[cfg(feature = "object")]
            Object(_) => take_unchecked_object(self, by, sorted),
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, _) => {
                let ca = self.decimal().unwrap();
                let out = ca.phys.take_chunked_unchecked(by, sorted, avoid_sharing);
                out.into_decimal_unchecked(ca.precision(), ca.scale())
                    .into_series()
            },
            #[cfg(feature = "dtype-date")]
            Date => {
                let ca = self.date().unwrap();
                ca.physical()
                    .take_chunked_unchecked(by, sorted, avoid_sharing)
                    .into_date()
                    .into_series()
            },
            #[cfg(feature = "dtype-datetime")]
            Datetime(u, z) => {
                let ca = self.datetime().unwrap();
                ca.physical()
                    .take_chunked_unchecked(by, sorted, avoid_sharing)
                    .into_datetime(*u, z.clone())
                    .into_series()
            },
            #[cfg(feature = "dtype-duration")]
            Duration(u) => {
                let ca = self.duration().unwrap();
                ca.physical()
                    .take_chunked_unchecked(by, sorted, avoid_sharing)
                    .into_duration(*u)
                    .into_series()
            },
            #[cfg(feature = "dtype-time")]
            Time => {
                let ca = self.time().unwrap();
                ca.physical()
                    .take_chunked_unchecked(by, sorted, avoid_sharing)
                    .into_time()
                    .into_series()
            },
            #[cfg(feature = "dtype-categorical")]
            Categorical(_, _) | Enum(_, _) => {
                with_match_categorical_physical_type!(self.dtype().cat_physical().unwrap(), |$C| {
                    let ca = self.cat::<$C>().unwrap();
                    CategoricalChunked::<$C>::from_cats_and_dtype_unchecked(
                        ca.physical().take_chunked_unchecked(by, sorted, avoid_sharing),
                        self.dtype().clone()
                    )
                    .into_series()
                })
            },
            Null => Series::new_null(self.name().clone(), by.len()),
            _ => unreachable!(),
        }
    }

    /// Take function that checks of null state in `ChunkIdx`.
    unsafe fn take_opt_chunked_unchecked<const B: u64>(
        &self,
        by: &[ChunkId<B>],
        avoid_sharing: bool,
    ) -> Self {
        use DataType::*;
        match self.dtype() {
            dt if dt.is_primitive_numeric() => {
                with_match_physical_numeric_polars_type!(self.dtype(), |$T| {
                 let ca: &ChunkedArray<$T> = self.as_ref().as_ref().as_ref();
                 ca.take_opt_chunked_unchecked(by, avoid_sharing).into_series()
                })
            },
            Boolean => {
                let ca = self.bool().unwrap();
                ca.take_opt_chunked_unchecked(by, avoid_sharing)
                    .into_series()
            },
            Binary => {
                let ca = self.binary().unwrap();
                ca.take_opt_chunked_unchecked(by, avoid_sharing)
                    .into_series()
            },
            String => {
                let ca = self.str().unwrap();
                ca.take_opt_chunked_unchecked(by, avoid_sharing)
                    .into_series()
            },
            List(_) => {
                let ca = self.list().unwrap();
                ca.take_opt_chunked_unchecked(by, avoid_sharing)
                    .into_series()
            },
            #[cfg(feature = "dtype-array")]
            Array(_, _) => {
                let ca = self.array().unwrap();
                ca.take_opt_chunked_unchecked(by, avoid_sharing)
                    .into_series()
            },
            #[cfg(feature = "dtype-struct")]
            Struct(_) => {
                let ca = self.struct_().unwrap();
                take_opt_chunked_unchecked_struct(ca, by, avoid_sharing).into_series()
            },
            #[cfg(feature = "object")]
            Object(_) => take_opt_unchecked_object(self, by, avoid_sharing),
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, _) => {
                let ca = self.decimal().unwrap();
                let out = ca.phys.take_opt_chunked_unchecked(by, avoid_sharing);
                out.into_decimal_unchecked(ca.precision(), ca.scale())
                    .into_series()
            },
            #[cfg(feature = "dtype-date")]
            Date => {
                let ca = self.date().unwrap();
                ca.physical()
                    .take_opt_chunked_unchecked(by, avoid_sharing)
                    .into_date()
                    .into_series()
            },
            #[cfg(feature = "dtype-datetime")]
            Datetime(u, z) => {
                let ca = self.datetime().unwrap();
                ca.physical()
                    .take_opt_chunked_unchecked(by, avoid_sharing)
                    .into_datetime(*u, z.clone())
                    .into_series()
            },
            #[cfg(feature = "dtype-duration")]
            Duration(u) => {
                let ca = self.duration().unwrap();
                ca.physical()
                    .take_opt_chunked_unchecked(by, avoid_sharing)
                    .into_duration(*u)
                    .into_series()
            },
            #[cfg(feature = "dtype-time")]
            Time => {
                let ca = self.time().unwrap();
                ca.physical()
                    .take_opt_chunked_unchecked(by, avoid_sharing)
                    .into_time()
                    .into_series()
            },
            #[cfg(feature = "dtype-categorical")]
            Categorical(_, _) | Enum(_, _) => {
                with_match_categorical_physical_type!(self.dtype().cat_physical().unwrap(), |$C| {
                    let ca = self.cat::<$C>().unwrap();
                    CategoricalChunked::<$C>::from_cats_and_dtype_unchecked(
                        ca.physical().take_opt_chunked_unchecked(by, avoid_sharing),
                        self.dtype().clone()
                    )
                    .into_series()
                })
            },
            Null => Series::new_null(self.name().clone(), by.len()),
            _ => unreachable!(),
        }
    }
}

/// The builder of the chunks of `ca`, with room for `by.len()` elements. A `ChunkedArray` always
/// has a chunk, which is the array the built one is shaped like.
fn gather_builder<T: PolarsDataType, const B: u64>(
    ca: &ChunkedArray<T>,
    by: &[ChunkId<B>],
) -> Box<dyn PlArrayBuilder> {
    let prototype = ca.chunks().first().expect("a ChunkedArray has a chunk");
    let mut builder = builder_like(&**prototype);
    builder.reserve(by.len());
    builder
}

/// Whether the gathered elements may keep pointing into the buffers they came from.
fn share_strategy(avoid_sharing: bool) -> ShareStrategy {
    if avoid_sharing {
        ShareStrategy::Never
    } else {
        ShareStrategy::Always
    }
}

impl<T> TakeChunked for ChunkedArray<T>
where
    T: PolarsDataType,
    T::Array: Debug,
{
    unsafe fn take_chunked_unchecked<const B: u64>(
        &self,
        by: &[ChunkId<B>],
        sorted: IsSorted,
        avoid_sharing: bool,
    ) -> Self {
        let mut builder = gather_builder(self, by);
        let share = share_strategy(avoid_sharing);

        for chunk_id in by {
            debug_assert!(
                !chunk_id.is_null(),
                "null chunks should not hit this branch"
            );
            let (chunk_idx, array_idx) = chunk_id.extract();
            let arr = self.downcast_get_unchecked(chunk_idx as usize);
            builder.subslice_extend(arr, array_idx as usize, 1, share);
        }

        // SAFETY: the builder was shaped like the chunks of this array, so what it froze is of
        // the same physical type.
        let mut out = self.with_chunks(vec![PlArrayBuilder::freeze(builder)]);
        let sorted_flag = _update_gather_sorted_flag(self.is_sorted_flag(), sorted);
        out.set_sorted_flag(sorted_flag);
        out
    }

    // Take function that checks of null state in `ChunkIdx`.
    unsafe fn take_opt_chunked_unchecked<const B: u64>(
        &self,
        by: &[ChunkId<B>],
        avoid_sharing: bool,
    ) -> Self {
        let mut builder = gather_builder(self, by);
        let share = share_strategy(avoid_sharing);

        for chunk_id in by {
            if chunk_id.is_null() {
                builder.extend_nulls(1);
                continue;
            }

            let (chunk_idx, array_idx) = chunk_id.extract();
            let arr = self.downcast_get_unchecked(chunk_idx as usize);
            builder.subslice_extend(arr, array_idx as usize, 1, share);
        }

        // SAFETY: as in `take_chunked_unchecked`.
        self.with_chunks(vec![PlArrayBuilder::freeze(builder)])
    }
}

#[cfg(feature = "object")]
unsafe fn take_unchecked_object<const B: u64>(
    s: &Series,
    by: &[ChunkId<B>],
    _sorted: IsSorted,
) -> Series {
    use polars_core::chunked_array::object::registry::get_object_builder;

    let mut builder = get_object_builder(s.name().clone(), by.len());

    by.iter().for_each(|chunk_id| {
        let (chunk_idx, array_idx) = chunk_id.extract();
        let object = s.get_object_chunked_unchecked(chunk_idx as usize, array_idx as usize);
        builder.append_option(object.map(|v| v.as_any()))
    });
    builder.to_series()
}

#[cfg(feature = "object")]
unsafe fn take_opt_unchecked_object<const B: u64>(
    s: &Series,
    by: &[ChunkId<B>],
    _allow_sharing: bool,
) -> Series {
    use polars_core::chunked_array::object::registry::get_object_builder;

    let mut builder = get_object_builder(s.name().clone(), by.len());

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

#[cfg(feature = "dtype-struct")]
unsafe fn take_chunked_unchecked_struct<const B: u64>(
    ca: &StructChunked,
    by: &[ChunkId<B>],
    sorted: IsSorted,
    avoid_sharing: bool,
) -> StructChunked {
    let fields = ca
        .fields_as_series()
        .iter()
        .map(|s| s.take_chunked_unchecked(by, sorted, avoid_sharing))
        .collect::<Vec<_>>();
    let mut out = StructChunked::from_series(ca.name().clone(), by.len(), fields.iter()).unwrap();

    if !ca.has_nulls() {
        return out;
    }

    let mut validity = BitmapBuilder::with_capacity(by.len());
    if ca.n_chunks() == 1 {
        let arr = ca.downcast_as_array();
        let bitmap = arr.validity().unwrap();
        for id in by.iter() {
            let (chunk_idx, array_idx) = id.extract();
            debug_assert!(chunk_idx == 0);
            validity.push_unchecked(bitmap.get_unchecked(array_idx as usize));
        }
    } else {
        for id in by.iter() {
            let (chunk_idx, array_idx) = id.extract();
            let arr = ca.downcast_get_unchecked(chunk_idx as usize);
            if let Some(bitmap) = arr.validity() {
                validity.push_unchecked(bitmap.get_unchecked(array_idx as usize));
            } else {
                validity.push_unchecked(true);
            }
        }
    }

    out.rechunk_mut(); // Should be a no-op.
    out.downcast_iter_mut()
        .next()
        .unwrap()
        .set_validity(validity.into_opt_validity().map(PlBitmap::from_bitmap));
    out
}

#[cfg(feature = "dtype-struct")]
unsafe fn take_opt_chunked_unchecked_struct<const B: u64>(
    ca: &StructChunked,
    by: &[ChunkId<B>],
    avoid_sharing: bool,
) -> StructChunked {
    let fields = ca
        .fields_as_series()
        .iter()
        .map(|s| s.take_opt_chunked_unchecked(by, avoid_sharing))
        .collect::<Vec<_>>();
    let mut out = StructChunked::from_series(ca.name().clone(), by.len(), fields.iter()).unwrap();

    let mut validity = BitmapBuilder::with_capacity(by.len());
    if ca.n_chunks() == 1 {
        let arr = ca.downcast_as_array();
        if let Some(bitmap) = arr.validity() {
            for id in by.iter() {
                if id.is_null() {
                    validity.push_unchecked(false);
                } else {
                    let (chunk_idx, array_idx) = id.extract();
                    debug_assert!(chunk_idx == 0);
                    validity.push_unchecked(bitmap.get_unchecked(array_idx as usize));
                }
            }
        } else {
            for id in by.iter() {
                validity.push_unchecked(!id.is_null());
            }
        }
    } else {
        for id in by.iter() {
            if id.is_null() {
                validity.push_unchecked(false);
            } else {
                let (chunk_idx, array_idx) = id.extract();
                let arr = ca.downcast_get_unchecked(chunk_idx as usize);
                if let Some(bitmap) = arr.validity() {
                    validity.push_unchecked(bitmap.get_unchecked(array_idx as usize));
                } else {
                    validity.push_unchecked(true);
                }
            }
        }
    }

    out.rechunk_mut(); // Should be a no-op.
    out.downcast_iter_mut()
        .next()
        .unwrap()
        .set_validity(validity.into_opt_validity().map(PlBitmap::from_bitmap));
    out
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_binview_chunked_gather() {
        unsafe {
            // # Series without nulls;
            let mut s_1 = Series::new(
                "a".into(),
                &["1 loooooooooooong string", "2 loooooooooooong string"],
            );
            let s_2 = Series::new(
                "a".into(),
                &["11 loooooooooooong string", "22 loooooooooooong string"],
            );
            let s_3 = Series::new(
                "a".into(),
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
            let by: [ChunkId<24>; 7] = [
                ChunkId::store(0, 0),
                ChunkId::store(0, 1),
                ChunkId::store(1, 1),
                ChunkId::store(1, 0),
                ChunkId::store(2, 0),
                ChunkId::store(2, 1),
                ChunkId::store(2, 2),
            ];

            let out = s_1.take_chunked_unchecked(&by, IsSorted::Not, true);
            let idx = IdxCa::new("".into(), [0, 1, 3, 2, 4, 5, 6]);
            let expected = s_1.rechunk().take(&idx).unwrap();
            assert!(out.equals(&expected));

            // ## Ids with nulls;
            let by: [ChunkId<24>; 4] = [
                ChunkId::null(),
                ChunkId::store(0, 1),
                ChunkId::store(1, 1),
                ChunkId::store(1, 0),
            ];
            let out = s_1.take_opt_chunked_unchecked(&by, true);

            let idx = IdxCa::new("".into(), [None, Some(1), Some(3), Some(2)]);
            let expected = s_1.rechunk().take(&idx).unwrap();
            assert!(out.equals_missing(&expected));

            // # Series with nulls;
            let mut s_1 = Series::new(
                "a".into(),
                &["1 loooooooooooong string 1", "2 loooooooooooong string 2"],
            );
            let s_2 = Series::new("a".into(), &[Some("11 loooooooooooong string 11"), None]);
            s_1.append(&s_2).unwrap();

            // ## Ids without nulls;
            let by: [ChunkId<24>; 4] = [
                ChunkId::store(0, 0),
                ChunkId::store(0, 1),
                ChunkId::store(1, 1),
                ChunkId::store(1, 0),
            ];

            let out = s_1.take_chunked_unchecked(&by, IsSorted::Not, true);
            let idx = IdxCa::new("".into(), [0, 1, 3, 2]);
            let expected = s_1.rechunk().take(&idx).unwrap();
            assert!(out.equals_missing(&expected));

            // ## Ids with nulls;
            let by: [ChunkId<24>; 4] = [
                ChunkId::null(),
                ChunkId::store(0, 1),
                ChunkId::store(1, 1),
                ChunkId::store(1, 0),
            ];
            let out = s_1.take_opt_chunked_unchecked(&by, true);

            let idx = IdxCa::new("".into(), [None, Some(1), Some(3), Some(2)]);
            let expected = s_1.rechunk().take(&idx).unwrap();
            assert!(out.equals_missing(&expected));
        }
    }

    #[test]
    #[cfg(feature = "dtype-categorical")]
    fn test_list_categorical_dtype_preserved_after_take() {
        use polars_core::prelude::*;

        unsafe {
            // Create List(String) and convert to List(Categorical)
            let mut builder = ListStringChunkedBuilder::new("a".into(), 2, 3);
            builder.append_values_iter(["a", "b"].iter().copied());
            builder.append_values_iter(["c", "d"].iter().copied());
            let list_str = builder.finish().into_series();

            let list_cat = list_str
                .list()
                .unwrap()
                .apply_to_inner(&|s| s.cast(&DataType::from_categories(Categories::global())))
                .unwrap()
                .into_series();

            // Append to create chunked series
            let mut chunked = list_cat.clone();
            chunked.append(&list_cat).unwrap();
            assert_eq!(chunked.n_chunks(), 2);

            // Perform chunked take
            let by: [ChunkId<24>; 2] = [ChunkId::store(0, 0), ChunkId::store(1, 0)];
            let out = chunked.take_chunked_unchecked(&by, IsSorted::Not, false);

            // Verify the Polars dtype is preserved
            // The bug was that List(Categorical) was becoming List(UInt32) after take
            assert!(
                matches!(out.dtype(), DataType::List(inner) if matches!(inner.as_ref(), DataType::Categorical(_, _))),
                "List(Categorical) dtype should be preserved after take_chunked_unchecked. Got: {:?}",
                out.dtype()
            );
        }
    }
}
