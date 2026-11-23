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
use polars_arrow::datatypes::{ArrowDataType, IdxArr, PhysicalType};
use polars_arrow::offset::{Offset, OffsetsBuffer};
use polars_utils::IdxSize;

use super::bitmap::take_bitmap_nulls_unchecked;

/// Mean row length from which copying rows beats an element gather (fixed-width children).
const ROW_COPY_MIN_LEN: usize = 32;

/// Primitive or boolean leaves, possibly under `FixedSizeList` levels: rows are contiguous.
fn is_fixed_width(dtype: &ArrowDataType) -> bool {
    match dtype {
        ArrowDataType::FixedSizeList(field, _) => is_fixed_width(field.dtype()),
        dt => matches!(
            dt.to_physical_type(),
            PhysicalType::Primitive(_) | PhysicalType::Boolean
        ),
    }
}

/// `take` implementation for ListArrays
pub(super) unsafe fn take_unchecked<I: Offset>(
    values: &ListArray<I>,
    indices: &IdxArr,
) -> ListArray<I>
where
    ListArray<I>: StaticArray + ArrayFromIterDtype<std::option::Option<Box<dyn array::Array>>>,
{
    let child = values.values();
    if !super::has_native_take(child.dtype()) {
        return super::take_unchecked_impl_generic(values, indices, &ListArray::new_null);
    }
    let offsets = values.offsets().as_slice();
    let src_len = offsets[offsets.len() - 1].to_usize() - offsets[0].to_usize();
    // Child indices must fit in `IdxSize` for the element gather.
    let row_copy = (is_fixed_width(child.dtype()) && src_len >= ROW_COPY_MIN_LEN * values.len())
        || child.len() > IdxSize::MAX as usize;
    take_native(values, indices, row_copy)
}

/// Gathers whole rows with a builder if `row_copy`, else child elements with one index gather.
/// Both keep the dtype of `values`.
unsafe fn take_native<I: Offset>(
    values: &ListArray<I>,
    indices: &IdxArr,
    row_copy: bool,
) -> ListArray<I> {
    if values.null_count() == values.len() || indices.null_count() == indices.len() {
        return ListArray::new_null(values.dtype().clone(), indices.len());
    }
    let child = values.values();
    let offsets = values.offsets().as_slice();
    let src_len = offsets[offsets.len() - 1].to_usize() - offsets[0].to_usize();

    let validity = values
        .validity()
        .filter(|v| v.unset_bits() > 0)
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
    let new_child = if row_copy {
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
        Array, BinaryArray, BooleanArray, Int64Array, ListArray, NullArray, PrimitiveArray,
        StructArray, Utf8ViewArray,
    };
    use polars_arrow::bitmap::Bitmap;
    use polars_arrow::datatypes::{ArrowDataType, Field, TimeUnit};
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
        if super::super::has_native_take(arr.values().dtype()) {
            // Both native paths, whichever one the thresholds pick.
            for row_copy in [false, true] {
                let out = unsafe { super::take_native(arr, &idx, row_copy) };
                assert_eq!(out, generic);
                assert_eq!(out.dtype(), arr.dtype());
                assert_eq!(out.offsets(), native.offsets());
            }
        }
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
    fn test_list_gather_children() {
        // Each native child type, sliced lists and children, a null row spanning values.
        let st = StructArray::new(
            ArrowDataType::Struct(vec![
                Field::new("a".into(), ArrowDataType::Int64, true),
                Field::new("b".into(), ArrowDataType::Boolean, true),
            ]),
            5,
            vec![
                Int64Array::from_slice([1, 2, 3, 4, 5]).boxed(),
                BooleanArray::from([Some(true), None, Some(false), Some(true), None]).boxed(),
            ],
            Some(Bitmap::from_iter([true, true, false, true, true])),
        );
        let leaves = Int64Array::from([Some(1), None, Some(3), Some(4), Some(5), Some(6)]).boxed();
        let inner = list(
            leaves,
            vec![0, 2, 2, 4, 5, 6],
            Some(&[true, true, false, true, true]),
        );
        let children = [
            Int64Array::from([Some(1), None, Some(3), Some(4), None, Some(6), Some(7)])
                .sliced(2, 5)
                .boxed(),
            BooleanArray::from_slice([true, false, true, true, false, true, false])
                .sliced(2, 5)
                .boxed(),
            NullArray::new(ArrowDataType::Null, 5).boxed(),
            BinaryArray::<i64>::from_slice([b"a".as_slice(), b"bb", b"", b"dddd", b"e"]).boxed(),
            Utf8ViewArray::from_slice([
                Some("x"),
                None,
                Some("a much longer string value"),
                Some(""),
                Some("y"),
                None,
                Some("z"),
            ])
            .sliced(2, 5)
            .boxed(),
            st.boxed(),
            inner.boxed(),
        ];
        for child in children {
            let arr = list(
                child,
                vec![0, 1, 3, 4, 4, 5],
                Some(&[true, true, false, true, true]),
            );
            for arr in [arr.clone(), arr.sliced(1, 4)] {
                for idx in IDX {
                    check(&arr, idx);
                }
            }
        }
    }

    #[test]
    fn test_list_gather_nested_keeps_dtype() {
        // Wide inner rows with a non-default field and a logical leaf type. Gathering only
        // the null inner row 0 once rebuilt the inner dtype and panicked.
        let w = 4 * super::ROW_COPY_MIN_LEN;
        let ts = ArrowDataType::Timestamp(TimeUnit::Millisecond, None);
        let leaves = Int64Array::from_vec((0..2 * w as i64).collect()).to(ts.clone());
        let st = StructArray::new(
            ArrowDataType::Struct(vec![Field::new("a".into(), ts, true)]),
            2 * w,
            vec![leaves.clone().boxed()],
            None,
        );
        let strs: Vec<Option<String>> = (0..2 * w).map(|i| Some(format!("value {i:08}"))).collect();
        let strs = Utf8ViewArray::from_slice(&strs);
        for child in [leaves.boxed(), st.boxed(), strs.boxed()] {
            let field = Field::new("element".into(), child.dtype().clone(), false);
            let w = w as i64;
            let inner = ListArray::<i64>::new(
                ArrowDataType::LargeList(Box::new(field)),
                vec![0, 0, w, w, 2 * w].try_into().unwrap(),
                child,
                Some(Bitmap::from_iter([false, true, true, true])),
            );
            let arr = list(inner.boxed(), vec![0, 1, 1, 3, 4], None);
            for idx in IDX {
                check(&arr, idx);
            }
        }
    }
}
