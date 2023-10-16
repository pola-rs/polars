use std::convert::TryFrom;

use arrow::array::*;
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::legacy::array::list::AnonymousBuilder;
use arrow::legacy::array::PolarsArray;
use arrow::legacy::bit_util::unset_bit_raw;
#[cfg(feature = "dtype-array")]
use arrow::legacy::is_valid::IsValid;
use arrow::legacy::prelude::*;
use arrow::legacy::trusted_len::TrustedLenPush;

#[cfg(feature = "dtype-array")]
use crate::chunked_array::builder::get_fixed_size_list_builder;
use crate::prelude::*;
use crate::series::implementations::null::NullChunked;

pub(crate) trait ExplodeByOffsets {
    fn explode_by_offsets(&self, offsets: &[i64]) -> Series;
}

unsafe fn unset_nulls(
    start: usize,
    last: usize,
    validity_values: &Bitmap,
    nulls: &mut Vec<usize>,
    empty_row_idx: &[usize],
    base_offset: usize,
) {
    for i in start..last {
        if !validity_values.get_bit_unchecked(i) {
            nulls.push(i + empty_row_idx.len() - base_offset);
        }
    }
}

fn get_capacity(offsets: &[i64]) -> usize {
    (offsets[offsets.len() - 1] - offsets[0] + 1) as usize
}

impl<T> ExplodeByOffsets for ChunkedArray<T>
where
    T: PolarsIntegerType,
{
    fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
        debug_assert_eq!(self.chunks.len(), 1);
        let arr = self.downcast_iter().next().unwrap();

        // make sure that we don't look beyond the sliced array
        let values = &arr.values().as_slice()[..offsets[offsets.len() - 1] as usize];

        let mut empty_row_idx = vec![];
        let mut nulls = vec![];

        let mut start = offsets[0] as usize;
        let base_offset = start;
        let mut last = start;

        let mut new_values = Vec::with_capacity(offsets[offsets.len() - 1] as usize - start + 1);

        // we check all the offsets and in the case a consecutive offset is the same,
        // e.g. 0, 1, 4, 4, 6
        // the 4 4, means that that is an empty row.
        // the empty row will be replaced with a None value.
        //
        // below we memcpy as much as possible and for the empty rows we add a default value
        // that value will later be masked out by the validity bitmap

        // in the case that the value array has got null values, we need to check every validity
        // value and collect the indices.
        // because the length of the array is not known, we first collect the null indexes, offset
        // with the insertion of empty rows (as None) and later create a validity bitmap
        if arr.has_validity() {
            let validity_values = arr.validity().unwrap();

            for &o in &offsets[1..] {
                let o = o as usize;
                if o == last {
                    if start != last {
                        #[cfg(debug_assertions)]
                        new_values.extend_from_slice(&values[start..last]);

                        #[cfg(not(debug_assertions))]
                        unsafe {
                            new_values.extend_from_slice(values.get_unchecked(start..last))
                        };

                        // Safety:
                        // we are in bounds
                        unsafe {
                            unset_nulls(
                                start,
                                last,
                                validity_values,
                                &mut nulls,
                                &empty_row_idx,
                                base_offset,
                            )
                        }
                    }

                    empty_row_idx.push(o + empty_row_idx.len() - base_offset);
                    new_values.push(T::Native::default());
                    start = o;
                }
                last = o;
            }

            // final null check
            // Safety:
            // we are in bounds
            unsafe {
                unset_nulls(
                    start,
                    last,
                    validity_values,
                    &mut nulls,
                    &empty_row_idx,
                    base_offset,
                )
            }
        } else {
            for &o in &offsets[1..] {
                let o = o as usize;
                if o == last {
                    if start != last {
                        #[cfg(debug_assertions)]
                        new_values.extend_from_slice(&values[start..last]);

                        #[cfg(not(debug_assertions))]
                        unsafe {
                            new_values.extend_from_slice(values.get_unchecked(start..last))
                        };
                    }

                    empty_row_idx.push(o + empty_row_idx.len() - base_offset);
                    new_values.push(T::Native::default());
                    start = o;
                }
                last = o;
            }
        }

        // add remaining values
        new_values.extend_from_slice(&values[start..]);

        let mut validity = MutableBitmap::with_capacity(new_values.len());
        validity.extend_constant(new_values.len(), true);
        let validity_slice = validity.as_slice().as_ptr() as *mut u8;

        for i in empty_row_idx {
            unsafe { unset_bit_raw(validity_slice, i) }
        }
        for i in nulls {
            unsafe { unset_bit_raw(validity_slice, i) }
        }
        let arr = PrimitiveArray::new(
            T::get_dtype().to_arrow(),
            new_values.into(),
            Some(validity.into()),
        );
        Series::try_from((self.name(), Box::new(arr) as ArrayRef)).unwrap()
    }
}

impl ExplodeByOffsets for Float32Chunked {
    fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
        self.apply_as_ints(|s| s.explode_by_offsets(offsets))
    }
}
impl ExplodeByOffsets for Float64Chunked {
    fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
        self.apply_as_ints(|s| s.explode_by_offsets(offsets))
    }
}

impl ExplodeByOffsets for NullChunked {
    fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
        let mut last_offset = offsets[0];

        let mut len = 0;
        for &offset in &offsets[1..] {
            // If offset == last_offset we have an empty list and a new row is inserted,
            // therefore we always increase at least 1.
            len += std::cmp::max(offset - last_offset, 1) as usize;
            last_offset = offset;
        }
        NullChunked::new(self.name.clone(), len).into_series()
    }
}

impl ExplodeByOffsets for BooleanChunked {
    fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
        debug_assert_eq!(self.chunks.len(), 1);
        let arr = self.downcast_iter().next().unwrap();

        let cap = get_capacity(offsets);
        let mut builder = BooleanChunkedBuilder::new(self.name(), cap);

        let mut start = offsets[0] as usize;
        let mut last = start;
        for &o in &offsets[1..] {
            let o = o as usize;
            if o == last {
                if start != last {
                    let vals = arr.slice_typed(start, last - start);

                    if vals.null_count() == 0 {
                        builder
                            .array_builder
                            .extend_trusted_len_values(vals.values_iter())
                    } else {
                        builder.array_builder.extend_trusted_len(vals.into_iter());
                    }
                }
                builder.append_null();
                start = o;
            }
            last = o;
        }
        let vals = arr.slice_typed(start, last - start);
        if vals.null_count() == 0 {
            builder
                .array_builder
                .extend_trusted_len_values(vals.values_iter())
        } else {
            builder.array_builder.extend_trusted_len(vals.into_iter());
        }
        builder.finish().into()
    }
}

impl ExplodeByOffsets for ListChunked {
    fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
        debug_assert_eq!(self.chunks.len(), 1);
        let arr = self.downcast_iter().next().unwrap();

        let cap = get_capacity(offsets);
        let inner_type = self.inner_dtype();

        let mut builder = arrow::legacy::array::list::AnonymousBuilder::new(cap);
        let mut owned = Vec::with_capacity(cap);
        let mut start = offsets[0] as usize;
        let mut last = start;

        let mut process_range = |start: usize, last: usize, builder: &mut AnonymousBuilder<'_>| {
            let vals = arr.slice_typed(start, last - start);
            for opt_arr in vals.into_iter() {
                match opt_arr {
                    None => builder.push_null(),
                    Some(arr) => {
                        unsafe {
                            // we create a pointer to evade the bck
                            let ptr = arr.as_ref() as *const dyn Array;
                            // safety: we preallocated
                            owned.push_unchecked(arr);
                            // safety: the pointer is still valid as `owned` will not reallocate
                            builder.push(&*ptr as &dyn Array);
                        }
                    },
                }
            }
        };

        for &o in &offsets[1..] {
            let o = o as usize;
            if o == last {
                if start != last {
                    process_range(start, last, &mut builder);
                }
                builder.push_null();
                start = o;
            }
            last = o;
        }
        process_range(start, last, &mut builder);
        let arr = builder.finish(Some(&inner_type.to_arrow())).unwrap();
        unsafe { self.copy_with_chunks(vec![Box::new(arr)], true, true) }.into_series()
    }
}

#[cfg(feature = "dtype-array")]
impl ExplodeByOffsets for ArrayChunked {
    fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
        debug_assert_eq!(self.chunks.len(), 1);
        let arr = self.downcast_iter().next().unwrap();

        let cap = get_capacity(offsets);
        let inner_type = self.inner_dtype();
        let mut builder =
            get_fixed_size_list_builder(&inner_type, cap, self.width(), self.name()).unwrap();

        let mut start = offsets[0] as usize;
        let mut last = start;
        for &o in &offsets[1..] {
            let o = o as usize;
            if o == last {
                if start != last {
                    let array = arr.slice_typed(start, last - start);
                    let values = array.values().as_ref();

                    for i in 0..array.len() {
                        unsafe {
                            if array.is_valid_unchecked(i) {
                                builder.push_unchecked(values, i)
                            } else {
                                builder.push_null()
                            }
                        }
                    }
                }
                unsafe {
                    builder.push_null();
                }
                start = o;
            }
            last = o;
        }
        let array = arr.slice_typed(start, last - start);
        let values = array.values().as_ref();
        for i in 0..array.len() {
            unsafe {
                if array.is_valid_unchecked(i) {
                    builder.push_unchecked(values, i)
                } else {
                    builder.push_null()
                }
            }
        }

        builder.finish().into()
    }
}

impl ExplodeByOffsets for Utf8Chunked {
    fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
        unsafe {
            self.as_binary()
                .explode_by_offsets(offsets)
                .cast_unchecked(&DataType::Utf8)
                .unwrap()
        }
    }
}

impl ExplodeByOffsets for BinaryChunked {
    fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
        debug_assert_eq!(self.chunks.len(), 1);
        let arr = self.downcast_iter().next().unwrap();

        let cap = get_capacity(offsets);
        let bytes_size = self.get_values_size();
        let mut builder = BinaryChunkedBuilder::new(self.name(), cap, bytes_size);

        let mut start = offsets[0] as usize;
        let mut last = start;
        for &o in &offsets[1..] {
            let o = o as usize;
            if o == last {
                if start != last {
                    let vals = arr.slice_typed(start, last - start);
                    if vals.null_count() == 0 {
                        builder
                            .builder
                            .extend_trusted_len_values(vals.values_iter())
                    } else {
                        builder.builder.extend_trusted_len(vals.into_iter());
                    }
                }
                builder.append_null();
                start = o;
            }
            last = o;
        }
        let vals = arr.slice_typed(start, last - start);
        if vals.null_count() == 0 {
            builder
                .builder
                .extend_trusted_len_values(vals.values_iter())
        } else {
            builder.builder.extend_trusted_len(vals.into_iter());
        }
        builder.finish().into()
    }
}

/// Convert Arrow array offsets to indexes of the original list
pub(crate) fn offsets_to_indexes(offsets: &[i64], capacity: usize) -> Vec<IdxSize> {
    if offsets.is_empty() {
        return vec![];
    }

    let mut idx = Vec::with_capacity(capacity);

    let mut last_idx = 0;
    for (offset_start, offset_end) in offsets.iter().zip(offsets[1..].iter()) {
        if idx.len() >= capacity {
            // significant speed-up in edge cases with many offsets,
            // no measurable overhead in typical case due to branch prediction
            break;
        }

        if offset_start == offset_end {
            // if the previous offset is equal to the current offset, we have an empty
            // list and we duplicate the previous index
            idx.push(last_idx);
        } else {
            let width = (offset_end - offset_start) as usize;
            for _ in 0..width {
                idx.push(last_idx);
            }
        }

        last_idx += 1;
    }

    // take the remaining values
    for _ in 0..capacity.saturating_sub(idx.len()) {
        idx.push(last_idx);
    }
    idx.truncate(capacity);
    idx
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::chunked_array::builder::get_list_builder;

    #[test]
    fn test_explode_list() -> PolarsResult<()> {
        let mut builder = get_list_builder(&DataType::Int32, 5, 5, "a")?;

        builder
            .append_series(&Series::new("", &[1, 2, 3, 3]))
            .unwrap();
        builder.append_series(&Series::new("", &[1])).unwrap();
        builder.append_series(&Series::new("", &[2])).unwrap();

        let ca = builder.finish();
        assert!(ca._can_fast_explode());

        // normal explode
        let exploded = ca.explode()?;
        let out: Vec<_> = exploded.i32()?.into_no_null_iter().collect();
        assert_eq!(out, &[1, 2, 3, 3, 1, 2]);

        // sliced explode
        let exploded = ca.slice(0, 1).explode()?;
        let out: Vec<_> = exploded.i32()?.into_no_null_iter().collect();
        assert_eq!(out, &[1, 2, 3, 3]);

        Ok(())
    }

    #[test]
    fn test_explode_list_nulls() -> PolarsResult<()> {
        let ca = Int32Chunked::from_slice_options("", &[None, Some(1), Some(2)]);
        let offsets = &[0, 3, 3];
        let out = ca.explode_by_offsets(offsets);
        assert_eq!(
            Vec::from(out.i32().unwrap()),
            &[None, Some(1), Some(2), None]
        );

        let ca = BooleanChunked::from_slice_options("", &[None, Some(true), Some(false)]);
        let out = ca.explode_by_offsets(offsets);
        assert_eq!(
            Vec::from(out.bool().unwrap()),
            &[None, Some(true), Some(false), None]
        );

        let ca = Utf8Chunked::from_slice_options("", &[None, Some("b"), Some("c")]);
        let out = ca.explode_by_offsets(offsets);
        assert_eq!(
            Vec::from(out.utf8().unwrap()),
            &[None, Some("b"), Some("c"), None]
        );
        Ok(())
    }

    #[test]
    fn test_explode_empty_list_slot() -> PolarsResult<()> {
        // primitive
        let mut builder = get_list_builder(&DataType::Int32, 5, 5, "a")?;
        builder.append_series(&Series::new("", &[1i32, 2])).unwrap();
        builder
            .append_series(&Int32Chunked::from_slice("", &[]).into_series())
            .unwrap();
        builder.append_series(&Series::new("", &[3i32])).unwrap();

        let ca = builder.finish();
        let exploded = ca.explode()?;
        assert_eq!(
            Vec::from(exploded.i32()?),
            &[Some(1), Some(2), None, Some(3)]
        );

        // more primitive
        let mut builder = get_list_builder(&DataType::Int32, 5, 5, "a")?;
        builder.append_series(&Series::new("", &[1i32])).unwrap();
        builder
            .append_series(&Int32Chunked::from_slice("", &[]).into_series())
            .unwrap();
        builder.append_series(&Series::new("", &[2i32])).unwrap();
        builder
            .append_series(&Int32Chunked::from_slice("", &[]).into_series())
            .unwrap();
        builder.append_series(&Series::new("", &[3, 4i32])).unwrap();

        let ca = builder.finish();
        let exploded = ca.explode()?;
        assert_eq!(
            Vec::from(exploded.i32()?),
            &[Some(1), None, Some(2), None, Some(3), Some(4)]
        );

        // utf8
        let mut builder = get_list_builder(&DataType::Utf8, 5, 5, "a")?;
        builder.append_series(&Series::new("", &["abc"])).unwrap();
        builder
            .append_series(
                &<Utf8Chunked as NewChunkedArray<Utf8Type, &str>>::from_slice("", &[])
                    .into_series(),
            )
            .unwrap();
        builder.append_series(&Series::new("", &["de"])).unwrap();
        builder
            .append_series(
                &<Utf8Chunked as NewChunkedArray<Utf8Type, &str>>::from_slice("", &[])
                    .into_series(),
            )
            .unwrap();
        builder.append_series(&Series::new("", &["fg"])).unwrap();
        builder
            .append_series(
                &<Utf8Chunked as NewChunkedArray<Utf8Type, &str>>::from_slice("", &[])
                    .into_series(),
            )
            .unwrap();

        let ca = builder.finish();
        let exploded = ca.explode()?;
        assert_eq!(
            Vec::from(exploded.utf8()?),
            &[Some("abc"), None, Some("de"), None, Some("fg"), None]
        );

        // boolean
        let mut builder = get_list_builder(&DataType::Boolean, 5, 5, "a")?;
        builder.append_series(&Series::new("", &[true])).unwrap();
        builder
            .append_series(&BooleanChunked::from_slice("", &[]).into_series())
            .unwrap();
        builder.append_series(&Series::new("", &[false])).unwrap();
        builder
            .append_series(&BooleanChunked::from_slice("", &[]).into_series())
            .unwrap();
        builder
            .append_series(&Series::new("", &[true, true]))
            .unwrap();

        let ca = builder.finish();
        let exploded = ca.explode()?;
        assert_eq!(
            Vec::from(exploded.bool()?),
            &[Some(true), None, Some(false), None, Some(true), Some(true)]
        );

        Ok(())
    }

    #[test]
    fn test_row_offsets() {
        let offsets = &[0, 1, 2, 2, 3, 4, 4];
        let out = offsets_to_indexes(offsets, 6);
        assert_eq!(out, &[0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_empty_row_offsets() {
        let offsets = &[0, 0];
        let out = offsets_to_indexes(offsets, 0);
        let expected: Vec<IdxSize> = Vec::new();
        assert_eq!(out, expected);
    }

    #[test]
    fn test_row_offsets_over_capacity() {
        let offsets = &[0, 1, 1, 2, 2];
        let out = offsets_to_indexes(offsets, 2);
        assert_eq!(out, &[0, 1]);
    }

    #[test]
    fn test_row_offsets_nonzero_first_offset() {
        let offsets = &[3, 6, 8];
        let out = offsets_to_indexes(offsets, 10);
        assert_eq!(out, &[0, 0, 0, 1, 1, 2, 2, 2, 2, 2]);
    }
}
