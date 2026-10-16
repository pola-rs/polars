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

use polars_arrow::array::builder::{ShareStrategy, make_builder};
use polars_arrow::array::{self, Array, ArrayFromIterDtype, ListArray, StaticArray};
use polars_arrow::compute::utils::combine_validities_and;
use polars_arrow::datatypes::{IdxArr, PhysicalType};
use polars_arrow::offset::{Offset, OffsetsBuffer};
use polars_utils::IdxSize;

use super::bitmap::take_bitmap_nulls_unchecked;

/// Mean row length from which copying rows beats an element gather (fixed-width children).
const ROW_COPY_MIN_LEN: usize = 32;
/// Mean row length from which concatenating row slices wins (other children).
const SLICE_CONCAT_MIN_LEN: usize = 64;

/// `take` implementation for ListArrays
pub(super) unsafe fn take_unchecked<I: Offset>(
    values: &ListArray<I>,
    indices: &IdxArr,
) -> ListArray<I>
where
    ListArray<I>: StaticArray + ArrayFromIterDtype<std::option::Option<Box<dyn array::Array>>>,
{
    let child = values.values();
    if !super::has_native_take(child.dtype()) || child.len() > IdxSize::MAX as usize {
        return super::take_unchecked_impl_generic(values, indices, &ListArray::new_null);
    }
    if values.null_count() == values.len() || indices.null_count() == indices.len() {
        return ListArray::new_null(values.dtype().clone(), indices.len());
    }
    let offsets = values.offsets().as_slice();
    let src_len = offsets[offsets.len() - 1].to_usize() - offsets[0].to_usize();
    let fixed_width = matches!(
        child.dtype().to_physical_type(),
        PhysicalType::Primitive(_) | PhysicalType::Boolean
    );
    if !fixed_width && src_len >= SLICE_CONCAT_MIN_LEN * values.len() {
        return super::take_unchecked_impl_generic(values, indices, &ListArray::new_null);
    }

    let validity = values
        .validity()
        .map(|v| take_bitmap_nulls_unchecked(v, indices));
    let validity = combine_validities_and(validity.as_ref(), indices.validity());

    // Null rows get length 0; their index may be garbage.
    let row_range = |i: &IdxSize| {
        let i = *i as usize;
        let start = offsets.get_unchecked(i).to_usize();
        (start, offsets.get_unchecked(i + 1).to_usize() - start)
    };
    let is_valid = |j: usize| validity.as_ref().is_none_or(|v| v.get_bit_unchecked(j));

    let mut new_offsets = Vec::with_capacity(indices.len() + 1);
    new_offsets.push(I::zero());
    let new_child = if fixed_width && src_len >= ROW_COPY_MIN_LEN * values.len() {
        let mut builder = make_builder(child.dtype());
        builder.reserve(indices.len() * src_len / values.len());
        for (j, i) in indices.values().iter().enumerate() {
            if is_valid(j) {
                let (start, len) = row_range(i);
                builder.subslice_extend(child.as_ref(), start, len, ShareStrategy::Always);
            }
            new_offsets.push(I::from_as_usize(builder.len()));
        }
        builder.freeze_reset()
    } else {
        let mut child_idx = Vec::<IdxSize>::with_capacity(indices.len() * src_len / values.len());
        for (j, i) in indices.values().iter().enumerate() {
            if is_valid(j) {
                let (start, len) = row_range(i);
                child_idx.extend(start as IdxSize..(start + len) as IdxSize);
            }
            new_offsets.push(I::from_as_usize(child_idx.len()));
        }
        super::take_unchecked(child.as_ref(), &IdxArr::from_vec(child_idx))
    };

    ListArray::new(
        values.dtype().clone(),
        OffsetsBuffer::new_unchecked(new_offsets.into()),
        new_child,
        validity,
    )
}

#[cfg(test)]
mod tests {
    use polars_arrow::array::{
        Array, BooleanArray, FixedSizeListArray, Int64Array, ListArray, NullArray, PrimitiveArray,
        StructArray, Utf8Array, Utf8ViewArray,
    };
    use polars_arrow::bitmap::Bitmap;
    use polars_arrow::datatypes::{ArrowDataType, Field};
    use polars_utils::IdxSize;

    fn list(
        values: Box<dyn Array>,
        offsets: Vec<i64>,
        validity: Option<&[bool]>,
    ) -> ListArray<i64> {
        let dtype = ListArray::<i64>::default_datatype(values.dtype().clone());
        ListArray::new(
            dtype,
            offsets.try_into().unwrap(),
            values,
            validity.map(|v| Bitmap::from_iter(v.iter().copied())),
        )
    }

    /// Null indices point far out of bounds to check they are never read.
    fn indices(idx: &[Option<IdxSize>]) -> PrimitiveArray<IdxSize> {
        let values = idx
            .iter()
            .map(|i| i.unwrap_or(IdxSize::MAX))
            .collect::<Vec<_>>();
        let validity = Bitmap::from_iter(idx.iter().map(|i| i.is_some()));
        PrimitiveArray::new(ArrowDataType::IDX_DTYPE, values.into(), Some(validity))
    }

    fn check(arr: &ListArray<i64>, idx: &[Option<IdxSize>]) -> ListArray<i64> {
        let idx = indices(idx);
        let (native, generic) = unsafe {
            (
                super::take_unchecked(arr, &idx),
                super::super::take_unchecked_impl_generic(arr, &idx, &ListArray::new_null),
            )
        };
        assert_eq!(native, generic);
        assert_eq!(native.dtype(), arr.dtype());
        native
    }

    const IDX: &[&[Option<IdxSize>]] = &[
        &[],
        &[None, None],
        &[Some(0)],
        &[Some(3), Some(1), Some(1), None, Some(0), Some(2)],
        &[Some(2), Some(2), Some(3)],
    ];

    #[test]
    fn test_list_gather_primitive() {
        let values = Int64Array::from_slice([1, 2, 3, 4, 5, 6, 7]).boxed();
        // Row 2 is null with a non-empty range, row 1 is empty.
        let arr = list(
            values,
            vec![0, 2, 2, 5, 7],
            Some(&[true, true, false, true]),
        );
        for idx in IDX {
            check(&arr, idx);
        }

        let out = check(&arr, &[Some(2), Some(0), None, Some(3)]);
        assert_eq!(out.offsets().as_slice(), &[0, 0, 2, 2, 4]);
        assert_eq!(out.values().len(), 4);
    }

    #[test]
    fn test_list_gather_sliced() {
        let values = Int64Array::from_slice([1, 2, 3, 4, 5, 6, 7, 8]).boxed();
        let arr = list(
            values,
            vec![0, 1, 3, 4, 6, 8],
            Some(&[true, false, true, true, true]),
        );
        let arr = arr.sliced(1, 4);
        assert_ne!(arr.offsets().first(), &0);
        for idx in IDX {
            check(&arr, idx);
        }
    }

    #[test]
    fn test_list_gather_str_bool_null() {
        let values =
            Utf8ViewArray::from_slice([Some("a"), None, Some("a longer string value"), Some("")]);
        let arr = list(values.boxed(), vec![0, 1, 1, 3, 4], None);
        for idx in IDX {
            check(&arr, idx);
        }

        let values = BooleanArray::from_slice([true, false, true, true]).boxed();
        let arr = list(
            values,
            vec![0, 1, 1, 3, 4],
            Some(&[true, true, true, false]),
        );
        for idx in IDX {
            check(&arr, idx);
        }

        let arr = list(
            NullArray::new(ArrowDataType::Null, 4).boxed(),
            vec![0, 1, 1, 3, 4],
            None,
        );
        for idx in IDX {
            check(&arr, idx);
        }
    }

    #[test]
    fn test_list_gather_nested() {
        let leaves = Int64Array::from([Some(1), None, Some(3), Some(4), Some(5), Some(6)]).boxed();
        let inner = list(
            leaves,
            vec![0, 2, 2, 4, 5, 6],
            Some(&[true, true, false, true, true]),
        );
        let arr = list(
            inner.boxed(),
            vec![0, 2, 2, 4, 5],
            Some(&[true, false, true, true]),
        );
        for idx in IDX {
            check(&arr, idx);
        }

        let fields = vec![
            Field::new("a".into(), ArrowDataType::Int64, true),
            Field::new("b".into(), ArrowDataType::Boolean, true),
        ];
        let st = StructArray::new(
            ArrowDataType::Struct(fields),
            5,
            vec![
                Int64Array::from_slice([1, 2, 3, 4, 5]).boxed(),
                BooleanArray::from([Some(true), None, Some(false), Some(true), None]).boxed(),
            ],
            Some(Bitmap::from_iter([true, true, false, true, true])),
        );
        let arr = list(st.boxed(), vec![0, 2, 2, 4, 5], None);
        for idx in IDX {
            check(&arr, idx);
        }

        let fsl = FixedSizeListArray::new(
            FixedSizeListArray::default_datatype(ArrowDataType::Int64, 2),
            5,
            Int64Array::from_slice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).boxed(),
            Some(Bitmap::from_iter([true, false, true, true, true])),
        );
        let arr = list(
            fsl.boxed(),
            vec![0, 2, 2, 4, 5],
            Some(&[true, true, true, false]),
        );
        for idx in IDX {
            check(&arr, idx);
        }
    }

    #[test]
    fn test_list_gather_fallback() {
        let values = Utf8Array::<i64>::from_slice(["a", "b", "c", "d"]).boxed();
        assert!(!super::super::has_native_take(values.dtype()));
        let arr = list(values, vec![0, 2, 2, 3, 4], None);
        for idx in IDX {
            check(&arr, idx);
        }
    }

    #[test]
    fn test_list_gather_fsl_child_is_generic() {
        // Array children keep their layout, including non-default fields.
        let field = Field::new("x".into(), ArrowDataType::Int64, false);
        let fsl = FixedSizeListArray::new(
            ArrowDataType::FixedSizeList(Box::new(field), 2),
            3,
            Int64Array::from_slice([0, 1, 2, 3, 4, 5]).boxed(),
            None,
        );
        assert!(!super::super::has_native_take(fsl.dtype()));
        let arr = list(fsl.boxed(), vec![0, 2, 2, 3, 3], None);
        for idx in IDX {
            unsafe { super::take_unchecked(&arr, &indices(idx)) };
        }
    }

    #[test]
    fn test_list_gather_wide_rows() {
        // Rows of at least `ROW_COPY_MIN_LEN` take the row-copy path.
        let w = super::ROW_COPY_MIN_LEN as i64;
        let leaves = (0..6 * w).map(|i| (i % 7 != 0).then_some(i));
        let values = Int64Array::from_iter(leaves).boxed();
        let arr = list(
            values,
            vec![0, w, 2 * w, 2 * w, 4 * w, 6 * w],
            Some(&[true, false, true, true, true]),
        );
        for arr in [arr.clone(), arr.sliced(1, 4)] {
            for idx in IDX {
                check(&arr, idx);
            }
        }
        let bools = BooleanArray::from_slice((0..4 * w).map(|i| i % 3 == 0).collect::<Vec<_>>());
        let arr = list(bools.boxed(), vec![0, w, 2 * w, 3 * w, 4 * w], None);
        for idx in IDX {
            check(&arr, idx);
        }
    }

    #[test]
    fn test_list_gather_wide_view_rows() {
        // Wide rows of variable-width values concatenate row slices.
        let w = super::SLICE_CONCAT_MIN_LEN;
        let strs: Vec<Option<String>> = (0..3 * w)
            .map(|i| (i % 5 != 0).then(|| format!("value number {i}")))
            .collect();
        let values = Utf8ViewArray::from_slice(&strs).boxed();
        let w = w as i64;
        let arr = list(
            values,
            vec![0, w, w, 2 * w, 3 * w],
            Some(&[true, true, false, true]),
        );
        for idx in IDX {
            check(&arr, idx);
        }
    }

    #[test]
    fn test_list_gather_all_null() {
        let values = Int64Array::from_slice([1, 2, 3, 4]).boxed();
        let arr = list(
            values,
            vec![0, 2, 2, 3, 4],
            Some(&[false, false, false, false]),
        );
        for idx in IDX {
            check(&arr, idx);
        }
    }
}
