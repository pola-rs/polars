// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::bitmap::Bitmap;
use crate::buffer::Buffer;
use crate::{
    array::PrimitiveArray,
    bitmap::{utils::SlicesIterator, MutableBitmap},
    types::NativeType,
};

use super::super::SortOptions;

/// # Safety
/// `indices[i] < values.len()` for all i
#[inline]
fn k_element_sort_inner<T, F>(values: &mut [T], descending: bool, limit: usize, mut cmp: F)
where
    T: NativeType,
    F: FnMut(&T, &T) -> std::cmp::Ordering,
{
    if descending {
        let (before, _, _) = values.select_nth_unstable_by(limit, |x, y| cmp(y, x));
        before.sort_unstable_by(|x, y| cmp(x, y));
    } else {
        let (before, _, _) = values.select_nth_unstable_by(limit, |x, y| cmp(x, y));
        before.sort_unstable_by(|x, y| cmp(x, y));
    }
}

fn sort_values<T, F>(values: &mut [T], mut cmp: F, descending: bool, limit: usize)
where
    T: NativeType,
    F: FnMut(&T, &T) -> std::cmp::Ordering,
{
    if limit != values.len() {
        return k_element_sort_inner(values, descending, limit, cmp);
    }

    if descending {
        values.sort_unstable_by(|x, y| cmp(y, x));
    } else {
        values.sort_unstable_by(cmp);
    };
}

fn sort_nullable<T, F>(
    values: &[T],
    validity: &Bitmap,
    cmp: F,
    options: &SortOptions,
    limit: usize,
) -> (Buffer<T>, Option<Bitmap>)
where
    T: NativeType,
    F: FnMut(&T, &T) -> std::cmp::Ordering,
{
    assert!(limit <= values.len());
    if options.nulls_first && limit < validity.unset_bits() {
        let buffer = vec![T::default(); limit];
        let bitmap = MutableBitmap::from_trusted_len_iter(std::iter::repeat(false).take(limit));
        return (buffer.into(), bitmap.into());
    }

    let nulls = std::iter::repeat(false).take(validity.unset_bits());
    let valids = std::iter::repeat(true).take(values.len() - validity.unset_bits());

    let mut buffer = Vec::<T>::with_capacity(values.len());
    let mut new_validity = MutableBitmap::with_capacity(values.len());
    let slices = SlicesIterator::new(validity);

    if options.nulls_first {
        // validity is [0,0,0,...,1,1,1,1]
        new_validity.extend_from_trusted_len_iter(nulls.chain(valids).take(limit));

        // extend buffer with constants followed by non-null values
        buffer.resize(validity.unset_bits(), T::default());
        for (start, len) in slices {
            buffer.extend_from_slice(&values[start..start + len])
        }

        // sort values
        sort_values(
            &mut buffer.as_mut_slice()[validity.unset_bits()..],
            cmp,
            options.descending,
            limit - validity.unset_bits(),
        );
    } else {
        // validity is [1,1,1,...,0,0,0,0]
        new_validity.extend_from_trusted_len_iter(valids.chain(nulls).take(limit));

        // extend buffer with non-null values
        for (start, len) in slices {
            buffer.extend_from_slice(&values[start..start + len])
        }

        // sort all non-null values
        sort_values(
            buffer.as_mut_slice(),
            cmp,
            options.descending,
            limit - validity.unset_bits(),
        );

        if limit > values.len() - validity.unset_bits() {
            // extend remaining with nulls
            buffer.resize(buffer.len() + validity.unset_bits(), T::default());
        }
    };
    // values are sorted, we can now truncate the remaining.
    buffer.truncate(limit);
    buffer.shrink_to_fit();

    (buffer.into(), new_validity.into())
}

/// Sorts a [`PrimitiveArray`] according to `cmp` comparator and [`SortOptions`].
pub fn sort_by<T, F>(
    array: &PrimitiveArray<T>,
    cmp: F,
    options: &SortOptions,
    limit: Option<usize>,
) -> PrimitiveArray<T>
where
    T: NativeType,
    F: FnMut(&T, &T) -> std::cmp::Ordering,
{
    let limit = limit.unwrap_or_else(|| array.len());
    let limit = limit.min(array.len());

    let values = array.values();
    let validity = array.validity();

    let (buffer, validity) = if let Some(validity) = validity {
        sort_nullable(values, validity, cmp, options, limit)
    } else {
        let mut buffer = Vec::<T>::new();
        buffer.extend_from_slice(values);

        sort_values(buffer.as_mut_slice(), cmp, options.descending, limit);
        buffer.truncate(limit);
        buffer.shrink_to_fit();

        (buffer.into(), None)
    };
    PrimitiveArray::<T>::new(array.data_type().clone(), buffer, validity)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::array::ord;
    use crate::array::PrimitiveArray;
    use crate::datatypes::DataType;

    fn test_sort_primitive_arrays<T>(
        data: &[Option<T>],
        data_type: DataType,
        options: SortOptions,
        expected_data: &[Option<T>],
    ) where
        T: NativeType + std::cmp::Ord,
    {
        let input = PrimitiveArray::<T>::from(data).to(data_type.clone());
        let expected = PrimitiveArray::<T>::from(expected_data).to(data_type.clone());
        let output = sort_by(&input, ord::total_cmp, &options, None);
        assert_eq!(expected, output);

        // with limit
        let expected = PrimitiveArray::<T>::from(&expected_data[..3]).to(data_type);
        let output = sort_by(&input, ord::total_cmp, &options, Some(3));
        assert_eq!(expected, output)
    }

    #[test]
    fn ascending_nulls_first() {
        test_sort_primitive_arrays::<i8>(
            &[None, Some(3), Some(5), Some(2), Some(3), None],
            DataType::Int8,
            SortOptions {
                descending: false,
                nulls_first: true,
            },
            &[None, None, Some(2), Some(3), Some(3), Some(5)],
        );
    }

    #[test]
    fn ascending_nulls_last() {
        test_sort_primitive_arrays::<i8>(
            &[None, Some(3), Some(5), Some(2), Some(3), None],
            DataType::Int8,
            SortOptions {
                descending: false,
                nulls_first: false,
            },
            &[Some(2), Some(3), Some(3), Some(5), None, None],
        );
    }

    #[test]
    fn descending_nulls_first() {
        test_sort_primitive_arrays::<i8>(
            &[None, Some(3), Some(5), Some(2), Some(3), None],
            DataType::Int8,
            SortOptions {
                descending: true,
                nulls_first: true,
            },
            &[None, None, Some(5), Some(3), Some(3), Some(2)],
        );
    }

    #[test]
    fn descending_nulls_last() {
        test_sort_primitive_arrays::<i8>(
            &[None, Some(3), Some(5), Some(2), Some(3), None],
            DataType::Int8,
            SortOptions {
                descending: true,
                nulls_first: false,
            },
            &[Some(5), Some(3), Some(3), Some(2), None, None],
        );
    }
}
