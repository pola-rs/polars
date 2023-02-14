use std::convert::TryFrom;

use arrow::array::*;
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::offset::OffsetsBuffer;
use polars_arrow::array::PolarsArray;
use polars_arrow::bit_util::unset_bit_raw;
use polars_arrow::prelude::*;

use crate::chunked_array::builder::AnonymousOwnedListBuilder;
use crate::prelude::*;

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

impl ExplodeByOffsets for BooleanChunked {
    fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
        debug_assert_eq!(self.chunks.len(), 1);
        let arr = self.downcast_iter().next().unwrap();

        let cap = ((arr.len() as f32) * 1.5) as usize;
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

        let cap = ((arr.len() as f32) * 1.5) as usize;
        let inner_type = self.inner_dtype();
        let mut builder = AnonymousOwnedListBuilder::new(self.name(), cap, Some(inner_type));

        let mut start = offsets[0] as usize;
        let mut last = start;
        for &o in &offsets[1..] {
            let o = o as usize;
            if o == last {
                if start != last {
                    let vals = arr.slice_typed(start, last - start);
                    let ca = unsafe { ListChunked::from_chunks("", vec![Box::new(vals)]) };
                    for s in &ca {
                        builder.append_opt_series(s.as_ref())
                    }
                }
                builder.append_null();
                start = o;
            }
            last = o;
        }
        let vals = arr.slice_typed(start, last - start);
        let ca = unsafe { ListChunked::from_chunks("", vec![Box::new(vals)]) };
        for s in &ca {
            builder.append_opt_series(s.as_ref())
        }
        builder.finish().into()
    }
}
impl ExplodeByOffsets for Utf8Chunked {
    fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
        debug_assert_eq!(self.chunks.len(), 1);
        let arr = self.downcast_iter().next().unwrap().clone();

        let cap = ((arr.len() as f32) * 1.5) as usize;
        let bytes_size = self.get_values_size();
        let mut builder = Utf8ChunkedBuilder::new(self.name(), cap, bytes_size);

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

#[cfg(feature = "dtype-binary")]
impl ExplodeByOffsets for BinaryChunked {
    fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
        debug_assert_eq!(self.chunks.len(), 1);
        let arr = self.downcast_iter().next().unwrap();

        let cap = ((arr.len() as f32) * 1.5) as usize;
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

    // `value_count` counts the taken values from the list values
    // and are the same unit as `offsets`
    // we also add the start offset as a list can be sliced
    let mut value_count = offsets[0];
    // `empty_count` counts the duplicates taken because of empty list
    let mut empty_count = 0usize;
    let mut last_idx = 0;

    for offset in &offsets[1..] {
        // this get all the elements up till offsets
        while value_count < *offset {
            value_count += 1;
            idx.push(last_idx)
        }

        // then we compute the previous offsets
        // Safety:
        // we started iterating from 1, so there is always a previous offset
        // we take the pointer to the previous element and deref that to get
        // the previous offset
        let previous_offset = unsafe { *(offset as *const i64).offset(-1) };

        // if the previous offset is equal to the current offset we have an empty
        // list and we duplicate previous index
        if previous_offset == *offset {
            empty_count += 1;
            idx.push(last_idx);
        }

        last_idx += 1;
    }

    // take the remaining values
    for _ in 0..(capacity - (value_count - offsets[0]) as usize - empty_count) {
        idx.push(last_idx);
    }
    idx
}

impl ChunkExplode for ListChunked {
    fn explode_and_offsets(&self) -> PolarsResult<(Series, OffsetsBuffer<i64>)> {
        // A list array's memory layout is actually already 'exploded', so we can just take the values array
        // of the list. And we also return a slice of the offsets. This slice can be used to find the old
        // list layout or indexes to expand the DataFrame in the same manner as the 'explode' operation
        let ca = self.rechunk();
        let listarr: &LargeListArray = ca
            .downcast_iter()
            .next()
            .ok_or_else(|| PolarsError::NoData("cannot explode empty list".into()))?;
        let offsets_buf = listarr.offsets().clone();
        let offsets = listarr.offsets().as_slice();
        let mut values = listarr.values().clone();

        let mut s = if ca._can_fast_explode() {
            // ensure that the value array is sliced
            // as a list only slices its offsets on a slice operation

            // we only do this in fast-explode as for the other
            // branch the offsets must coincide with the values.
            if !offsets.is_empty() {
                let start = offsets[0] as usize;
                let len = offsets[offsets.len() - 1] as usize - start;
                // safety:
                // we are in bounds
                values = unsafe { values.sliced_unchecked(start, len) };
            }
            Series::try_from((self.name(), values)).unwrap()
        } else {
            // during tests
            // test that this code branch is not hit with list arrays that could be fast exploded
            #[cfg(test)]
            {
                let mut last = offsets[0];
                let mut has_empty = false;
                for &o in &offsets[1..] {
                    if o == last {
                        has_empty = true;
                    }
                    last = o;
                }
                if !has_empty && offsets[0] == 0 {
                    panic!("could have fast exploded")
                }
            }

            let values = Series::try_from((self.name(), values)).unwrap();
            values.explode_by_offsets(offsets)
        };
        debug_assert_eq!(s.name(), self.name());
        // make sure we restore the logical type
        match self.inner_dtype() {
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(rev_map) => {
                let cats = s.u32().unwrap().clone();
                // safety:
                // rev_map is from same array, so we are still in bounds
                s = unsafe {
                    CategoricalChunked::from_cats_and_rev_map_unchecked(cats, rev_map.unwrap())
                        .into_series()
                };
            }
            #[cfg(feature = "dtype-date")]
            DataType::Date => s = s.into_date(),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(tu, tz) => s = s.into_datetime(tu, tz),
            #[cfg(feature = "dtype-duration")]
            DataType::Duration(tu) => s = s.into_duration(tu),
            #[cfg(feature = "dtype-time")]
            DataType::Time => s = s.into_time(),
            _ => {}
        }

        Ok((s, offsets_buf))
    }
}

impl ChunkExplode for Utf8Chunked {
    fn explode_and_offsets(&self) -> PolarsResult<(Series, OffsetsBuffer<i64>)> {
        // A list array's memory layout is actually already 'exploded', so we can just take the values array
        // of the list. And we also return a slice of the offsets. This slice can be used to find the old
        // list layout or indexes to expand the DataFrame in the same manner as the 'explode' operation
        let ca = self.rechunk();
        let array: &Utf8Array<i64> = ca
            .downcast_iter()
            .next()
            .ok_or_else(|| PolarsError::NoData("cannot explode empty str".into()))?;

        let values = array.values();
        let old_offsets = array.offsets().clone();

        let (new_offsets, validity) = if let Some(validity) = array.validity() {
            // capacity estimate
            let capacity = self.get_values_size() + validity.unset_bits();

            let old_offsets = old_offsets.as_slice();
            let mut old_offset = old_offsets[0];
            let mut new_offsets = Vec::with_capacity(capacity + 1);
            new_offsets.push(old_offset);

            let mut bitmap = MutableBitmap::with_capacity(capacity);
            let values = values.as_slice();
            for (&offset, valid) in old_offsets[1..].iter().zip(validity) {
                // safety:
                // new_offsets already has a single value, so -1 is always in bounds
                let latest_offset = unsafe { *new_offsets.get_unchecked(new_offsets.len() - 1) };

                if valid {
                    debug_assert!(old_offset as usize <= values.len());
                    debug_assert!(offset as usize <= values.len());
                    let val = unsafe { values.get_unchecked(old_offset as usize..offset as usize) };

                    // take the string value and find the char offsets
                    // create a new offset value for each char boundary
                    // safety:
                    // we know we have string data.
                    let str_val = unsafe { std::str::from_utf8_unchecked(val) };

                    let char_offsets = str_val
                        .char_indices()
                        .skip(1)
                        .map(|t| t.0 as i64 + latest_offset);

                    // extend the chars
                    // also keep track of the amount of offsets added
                    // as we must update the validity bitmap
                    let len_before = new_offsets.len();
                    new_offsets.extend(char_offsets);
                    new_offsets.push(latest_offset + str_val.len() as i64);
                    bitmap.extend_constant(new_offsets.len() - len_before, true);
                } else {
                    // no data, just add old offset and set null bit
                    new_offsets.push(latest_offset);
                    bitmap.push(false)
                }
                old_offset = offset;
            }

            (new_offsets.into(), bitmap.into())
        } else {
            // fast(er) explode

            // we cannot naively explode, because there might be empty strings.

            // capacity estimate
            let capacity = self.get_values_size();
            let old_offsets = old_offsets.as_slice();
            let mut old_offset = old_offsets[0];
            let mut new_offsets = Vec::with_capacity(capacity + 1);
            new_offsets.push(old_offset);

            let values = values.as_slice();
            for &offset in &old_offsets[1..] {
                // safety:
                // new_offsets already has a single value, so -1 is always in bounds
                let latest_offset = unsafe { *new_offsets.get_unchecked(new_offsets.len() - 1) };
                debug_assert!(old_offset as usize <= values.len());
                debug_assert!(offset as usize <= values.len());
                let val = unsafe { values.get_unchecked(old_offset as usize..offset as usize) };

                // take the string value and find the char offsets
                // create a new offset value for each char boundary
                // safety:
                // we know we have string data.
                let str_val = unsafe { std::str::from_utf8_unchecked(val) };

                let char_offsets = str_val
                    .char_indices()
                    .skip(1)
                    .map(|t| t.0 as i64 + latest_offset);

                // extend the chars
                new_offsets.extend(char_offsets);
                new_offsets.push(latest_offset + str_val.len() as i64);
                old_offset = offset;
            }

            (new_offsets.into(), None)
        };

        let array = unsafe {
            Utf8Array::<i64>::from_data_unchecked_default(new_offsets, values.clone(), validity)
        };

        let new_arr = Box::new(array) as ArrayRef;

        let s = Series::try_from((self.name(), new_arr)).unwrap();
        Ok((s, old_offsets))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::chunked_array::builder::get_list_builder;

    #[test]
    fn test_explode_list() -> PolarsResult<()> {
        let mut builder = get_list_builder(&DataType::Int32, 5, 5, "a")?;

        builder.append_series(&Series::new("", &[1, 2, 3, 3]));
        builder.append_series(&Series::new("", &[1]));
        builder.append_series(&Series::new("", &[2]));

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
        builder.append_series(&Series::new("", &[1i32, 2]));
        builder.append_series(&Int32Chunked::from_slice("", &[]).into_series());
        builder.append_series(&Series::new("", &[3i32]));

        let ca = builder.finish();
        let exploded = ca.explode()?;
        assert_eq!(
            Vec::from(exploded.i32()?),
            &[Some(1), Some(2), None, Some(3)]
        );

        // more primitive
        let mut builder = get_list_builder(&DataType::Int32, 5, 5, "a")?;
        builder.append_series(&Series::new("", &[1i32]));
        builder.append_series(&Int32Chunked::from_slice("", &[]).into_series());
        builder.append_series(&Series::new("", &[2i32]));
        builder.append_series(&Int32Chunked::from_slice("", &[]).into_series());
        builder.append_series(&Series::new("", &[3, 4i32]));

        let ca = builder.finish();
        let exploded = ca.explode()?;
        assert_eq!(
            Vec::from(exploded.i32()?),
            &[Some(1), None, Some(2), None, Some(3), Some(4)]
        );

        // utf8
        let mut builder = get_list_builder(&DataType::Utf8, 5, 5, "a")?;
        builder.append_series(&Series::new("", &["abc"]));
        builder.append_series(
            &<Utf8Chunked as NewChunkedArray<Utf8Type, &str>>::from_slice("", &[]).into_series(),
        );
        builder.append_series(&Series::new("", &["de"]));
        builder.append_series(
            &<Utf8Chunked as NewChunkedArray<Utf8Type, &str>>::from_slice("", &[]).into_series(),
        );
        builder.append_series(&Series::new("", &["fg"]));
        builder.append_series(
            &<Utf8Chunked as NewChunkedArray<Utf8Type, &str>>::from_slice("", &[]).into_series(),
        );

        let ca = builder.finish();
        let exploded = ca.explode()?;
        assert_eq!(
            Vec::from(exploded.utf8()?),
            &[Some("abc"), None, Some("de"), None, Some("fg"), None]
        );

        // boolean
        let mut builder = get_list_builder(&DataType::Boolean, 5, 5, "a")?;
        builder.append_series(&Series::new("", &[true]));
        builder.append_series(&BooleanChunked::from_slice("", &[]).into_series());
        builder.append_series(&Series::new("", &[false]));
        builder.append_series(&BooleanChunked::from_slice("", &[]).into_series());
        builder.append_series(&Series::new("", &[true, true]));

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
}
