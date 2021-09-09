use crate::prelude::*;
use arrow::array::{ArrayRef, BooleanArray, BooleanBufferBuilder};
use arrow::buffer::MutableBuffer;
use arrow::datatypes::{ArrowNativeType, ToByteSlice};
use arrow::util::bit_util::unset_bit_raw;
use arrow::{
    array::{Array, ArrayData, LargeListArray, LargeStringArray},
    buffer::Buffer,
};
use itertools::Itertools;
use polars_arrow::bit_util;
use polars_arrow::is_valid::IsValid;
use std::convert::TryFrom;
use std::ops::Deref;

pub struct ExplodedOffsets {
    buf: Buffer,
    offset: usize,
    len: usize,
}

impl ExplodedOffsets {
    pub(crate) fn value_offsets(&self) -> &[i64] {
        // Soundness
        //     Buffer holds i64 data and offset and len are copied from ArrayData
        unsafe {
            std::slice::from_raw_parts(
                (self.buf.as_ptr() as *const i64).add(self.offset),
                self.len + 1,
            )
        }
    }
}

pub(crate) trait ExplodeByOffsets {
    fn explode_by_offsets(&self, offsets: &[i64]) -> Series;
}

impl<T> ExplodeByOffsets for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Default + ArrowNativeType,
{
    fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
        debug_assert_eq!(self.chunks.len(), 1);
        let arr = self.downcast_iter().next().unwrap();
        let values = arr.values();

        let mut new_values = AlignedVec::with_capacity(((values.len() as f32) * 1.5) as usize);
        let mut empty_row_idx = vec![];
        let mut nulls = vec![];

        let mut start = offsets[0] as usize;
        let mut last = start;
        // we check all the offsets and in the case a consecutive offset is the same,
        // e.g. 0, 1, 4, 4, 6
        // the 4 4, means that that is an empty row.
        // the empty row will be replaced with a None value.
        //
        // below we memcpy as much as possible and for the empty rows we add a default value
        // that value will later be masked out by the validity bitmap

        // in the case that the value array has got null values, we need to check every validity
        // value and collect the indices.
        // because the length of the array is not known, we first collect the null indexes, ofsetted
        // with the insertion of empty rows (as None) and later create a validity bitmap
        if arr.null_count() > 0 {
            let validity_values = arr.data_ref().null_buffer().unwrap();
            let offset = arr.offset();

            for &o in &offsets[1..] {
                let o = o as usize;
                if o == last {
                    if start != last {
                        for i in start..last {
                            if unsafe { validity_values.is_null_unchecked(i + offset) } {
                                nulls.push(i + empty_row_idx.len());
                            }
                        }
                        new_values.extend_memcpy(&values[start..last])
                    }

                    empty_row_idx.push(o + empty_row_idx.len());
                    new_values.push(T::Native::default());
                    start = o;
                }
                last = o;
            }

            // final null check
            for i in start..last {
                if unsafe { validity_values.is_null_unchecked(i + offset) } {
                    nulls.push(i + empty_row_idx.len());
                }
            }
        } else {
            for &o in &offsets[1..] {
                let o = o as usize;
                if o == last {
                    if start != last {
                        new_values.extend_memcpy(&values[start..last])
                    }

                    empty_row_idx.push(o + empty_row_idx.len());
                    new_values.push(T::Native::default());
                    start = o;
                }
                last = o;
            }
        }
        new_values.extend_memcpy(&values[start..]);

        let num_bytes = bit_util::ceil(last, 8);
        let mut validity = MutableBuffer::new(num_bytes).with_bitset(num_bytes, true);
        let validity_slice = validity.as_mut_ptr();

        for i in empty_row_idx {
            unsafe { unset_bit_raw(validity_slice, i) }
        }
        for i in nulls {
            unsafe { unset_bit_raw(validity_slice, i) }
        }
        let arr = new_values.into_primitive_array::<T>(Some(validity.into()));
        Series::try_from((self.name(), Arc::new(arr) as ArrayRef)).unwrap()
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
                    let vals = arr.slice(start, last - start);
                    let vals_ref = vals.as_any().downcast_ref::<BooleanArray>().unwrap();
                    for val in vals_ref {
                        builder.append_option(val)
                    }
                }
                builder.append_null();
                start = o;
            }
            last = o;
        }
        let vals = arr.slice(start, last - start);
        let vals_ref = vals.as_any().downcast_ref::<BooleanArray>().unwrap();
        for val in vals_ref {
            builder.append_option(val)
        }
        builder.finish().into()
    }
}
impl ExplodeByOffsets for ListChunked {
    fn explode_by_offsets(&self, _offsets: &[i64]) -> Series {
        panic!("cannot explode List of Lists")
    }
}
impl ExplodeByOffsets for Utf8Chunked {
    fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
        debug_assert_eq!(self.chunks.len(), 1);
        let arr = self.downcast_iter().next().unwrap();

        let cap = ((arr.len() as f32) * 1.5) as usize;
        let bytes_size = self.get_values_size();
        let mut builder = Utf8ChunkedBuilder::new(self.name(), cap, bytes_size);

        let mut start = offsets[0] as usize;
        let mut last = start;
        for &o in &offsets[1..] {
            let o = o as usize;
            if o == last {
                if start != last {
                    let vals = arr.slice(start, last - start);
                    let vals_ref = vals.as_any().downcast_ref::<LargeStringArray>().unwrap();
                    for val in vals_ref {
                        builder.append_option(val)
                    }
                }
                builder.append_null();
                start = o;
            }
            last = o;
        }
        let vals = arr.slice(start, last - start);
        let vals_ref = vals.as_any().downcast_ref::<LargeStringArray>().unwrap();
        for val in vals_ref {
            builder.append_option(val)
        }
        builder.finish().into()
    }
}
impl ExplodeByOffsets for CategoricalChunked {
    #[inline(never)]
    fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
        let ca: CategoricalChunked = self
            .deref()
            .explode_by_offsets(offsets)
            .cast_with_dtype(&DataType::Categorical)
            .unwrap()
            .categorical()
            .unwrap()
            .clone();
        ca.set_state(self).into()
    }
}

/// Convert Arrow array offsets to indexes of the original list
pub(crate) fn offsets_to_indexes(offsets: &[i64], capacity: usize) -> AlignedVec<u32> {
    let mut idx = AlignedVec::with_capacity(capacity);

    let mut count = 0;
    let mut last_idx = 0;
    for &offset in offsets.iter().skip(1) {
        while count < offset {
            count += 1;
            idx.push(last_idx)
        }
        last_idx += 1;
    }
    for _ in 0..(capacity - count as usize) {
        idx.push(last_idx);
    }
    idx
}

impl ChunkExplode for ListChunked {
    fn explode_and_offsets(&self) -> Result<(Series, ExplodedOffsets)> {
        // A list array's memory layout is actually already 'exploded', so we can just take the values array
        // of the list. And we also return a slice of the offsets. This slice can be used to find the old
        // list layout or indexes to expand the DataFrame in the same manner as the 'explode' operation
        let ca = self.rechunk();
        let listarr: &LargeListArray = ca
            .downcast_iter()
            .next()
            .ok_or_else(|| PolarsError::NoData("cannot explode empty list".into()))?;
        let offsets = listarr.value_offsets();
        let offsets_buf = listarr.data_ref().buffers()[0].clone();

        let values = listarr
            .values()
            .slice(listarr.offset(), (offsets[offsets.len() - 1]) as usize);

        let s = if ca.can_fast_explode() {
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
                if !has_empty {
                    panic!()
                }
            }

            let values = Series::try_from(("", values)).unwrap();
            values.explode_by_offsets(offsets)
        };
        let offsets = ExplodedOffsets {
            buf: offsets_buf,
            offset: listarr.offset(),
            len: listarr.len(),
        };
        Ok((s, offsets))
    }
}

impl ChunkExplode for Utf8Chunked {
    fn explode_and_offsets(&self) -> Result<(Series, ExplodedOffsets)> {
        // A list array's memory layout is actually already 'exploded', so we can just take the values array
        // of the list. And we also return a slice of the offsets. This slice can be used to find the old
        // list layout or indexes to expand the DataFrame in the same manner as the 'explode' operation
        let ca = self.rechunk();
        let stringarr: &LargeStringArray = ca
            .downcast_iter()
            .next()
            .ok_or_else(|| PolarsError::NoData("cannot explode empty str".into()))?;
        let offsets_buf = stringarr.data().buffers()[0].clone();
        let str_values_buf = stringarr.value_data();

        let offsets = stringarr.value_offsets();

        // Because the strings are u8 stored but really are utf8 data we need to traverse the utf8 to
        // get the chars indexes
        let str_data = unsafe { std::str::from_utf8_unchecked(str_values_buf.as_slice()) };
        // iterator over index and chars, we take only the index
        // todo! directly create a buffer from an aligned vec or a mutable buffer
        let mut new_offsets = str_data.char_indices().map(|t| t.0 as i64).collect_vec();
        // somehow I don't get the last value if we don't add this one.
        new_offsets.push(str_data.len() as i64);

        // first buffer are the offsets. We now have only a single offset
        // second buffer is the actual values buffer
        let mut builder = ArrayData::builder(ArrowDataType::LargeUtf8)
            .len(new_offsets.len() - 1)
            .add_buffer(Buffer::from(new_offsets.to_byte_slice()))
            .add_buffer(str_values_buf);

        // the old bitmap doesn't fit on the exploded array, so we need to create a new one.
        if self.null_count() > 0 {
            let capacity = new_offsets.len();
            let mut bitmap_builder = BooleanBufferBuilder::new(new_offsets.len());

            let mut count = 0;
            let mut last_idx = 0;
            let mut last_valid = stringarr.is_valid(last_idx);
            for &offset in offsets.iter().skip(1) {
                while count < offset {
                    count += 1;
                    bitmap_builder.append(last_valid);
                }
                last_idx += 1;
                last_valid = stringarr.is_valid(last_idx);
            }
            for _ in 0..(capacity - count as usize) {
                bitmap_builder.append(last_valid);
            }
            builder = builder.null_bit_buffer(bitmap_builder.finish());
        }
        let arr_data = builder.build();

        let new_arr = Arc::new(LargeStringArray::from(arr_data)) as ArrayRef;

        let s = Series::try_from((self.name(), new_arr)).unwrap();
        let offsets = ExplodedOffsets {
            buf: offsets_buf,
            offset: stringarr.offset(),
            len: stringarr.len(),
        };
        Ok((s, offsets))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::chunked_array::builder::get_list_builder;

    #[test]
    fn test_explode_list() -> Result<()> {
        let mut builder = get_list_builder(&DataType::Int32, 5, 5, "a");

        builder.append_series(&Series::new("", &[1, 2, 3, 3]));
        builder.append_series(&Series::new("", &[1]));
        builder.append_series(&Series::new("", &[2]));

        let ca = builder.finish();
        assert!(ca.can_fast_explode());

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
    fn test_explode_empty_list_slot() -> Result<()> {
        // primitive
        let mut builder = get_list_builder(&DataType::Int32, 5, 5, "a");
        builder.append_series(&Series::new("", &[1i32, 2]));
        builder.append_series(&Int32Chunked::new_from_slice("", &[]).into_series());
        builder.append_series(&Series::new("", &[3i32]));

        let ca = builder.finish();
        let exploded = ca.explode()?;
        assert_eq!(
            Vec::from(exploded.i32()?),
            &[Some(1), Some(2), None, Some(3)]
        );

        // more primitive
        let mut builder = get_list_builder(&DataType::Int32, 5, 5, "a");
        builder.append_series(&Series::new("", &[1i32]));
        builder.append_series(&Int32Chunked::new_from_slice("", &[]).into_series());
        builder.append_series(&Series::new("", &[2i32]));
        builder.append_series(&Int32Chunked::new_from_slice("", &[]).into_series());
        builder.append_series(&Series::new("", &[3, 4i32]));

        let ca = builder.finish();
        let exploded = ca.explode()?;
        assert_eq!(
            Vec::from(exploded.i32()?),
            &[Some(1), None, Some(2), None, Some(3), Some(4)]
        );

        // primitive with nulls and empty rows
        let mut builder = get_list_builder(&DataType::Int32, 5, 5, "a");
        builder.append_series(&Series::new("", &[Some(1i32), None, Some(2)]));
        builder.append_series(&Int32Chunked::new_from_slice("", &[]).into_series());
        builder.append_series(&Series::new("", &[2i32]));
        builder.append_series(&Int32Chunked::new_from_slice("", &[]).into_series());
        builder.append_series(&Series::new("", &[3, 4i32]));

        let ca = builder.finish();
        let exploded = ca.explode()?;
        assert_eq!(
            Vec::from(exploded.i32()?),
            &[
                Some(1),
                None,
                Some(2),
                None,
                Some(2),
                None,
                Some(3),
                Some(4)
            ]
        );

        // utf8
        let mut builder = get_list_builder(&DataType::Utf8, 5, 5, "a");
        builder.append_series(&Series::new("", &["abc"]));
        builder.append_series(
            &<Utf8Chunked as NewChunkedArray<Utf8Type, &str>>::new_from_slice("", &[])
                .into_series(),
        );
        builder.append_series(&Series::new("", &["de"]));
        builder.append_series(
            &<Utf8Chunked as NewChunkedArray<Utf8Type, &str>>::new_from_slice("", &[])
                .into_series(),
        );
        builder.append_series(&Series::new("", &["fg"]));
        builder.append_series(
            &<Utf8Chunked as NewChunkedArray<Utf8Type, &str>>::new_from_slice("", &[])
                .into_series(),
        );

        let ca = builder.finish();
        let exploded = ca.explode()?;
        assert_eq!(
            Vec::from(exploded.utf8()?),
            &[Some("abc"), None, Some("de"), None, Some("fg"), None]
        );

        // boolean
        let mut builder = get_list_builder(&DataType::Boolean, 5, 5, "a");
        builder.append_series(&Series::new("", &[true]));
        builder.append_series(&BooleanChunked::new_from_slice("", &[]).into_series());
        builder.append_series(&Series::new("", &[false]));
        builder.append_series(&BooleanChunked::new_from_slice("", &[]).into_series());
        builder.append_series(&Series::new("", &[true, true]));

        let ca = builder.finish();
        let exploded = ca.explode()?;
        assert_eq!(
            Vec::from(exploded.bool()?),
            &[Some(true), None, Some(false), None, Some(true), Some(true)]
        );

        Ok(())
    }
}
