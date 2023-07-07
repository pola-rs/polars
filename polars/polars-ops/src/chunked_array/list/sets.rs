use std::fmt::{Display, Formatter};
use std::hash::Hash;

use arrow::array::{
    BinaryArray, ListArray, MutableArray, MutableBinaryArray, MutablePrimitiveArray,
    PrimitiveArray, Utf8Array,
};
use arrow::bitmap::Bitmap;
use arrow::offset::OffsetsBuffer;
use arrow::types::NativeType;
use polars_arrow::utils::combine_validities_and;
use polars_core::prelude::*;
use polars_core::utils::align_chunks_binary;
use polars_core::with_match_physical_integer_type;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

trait MaterializeValues<K> {
    // extends the iterator to the values and returns the current offset
    fn extend_buf<I: Iterator<Item = K>>(&mut self, values: I) -> usize;
}

impl<T> MaterializeValues<Option<T>> for MutablePrimitiveArray<T>
where
    T: NativeType,
{
    fn extend_buf<I: Iterator<Item = Option<T>>>(&mut self, values: I) -> usize {
        self.extend(values);
        self.len()
    }
}

impl<'a> MaterializeValues<Option<&'a [u8]>> for MutableBinaryArray<i64> {
    fn extend_buf<I: Iterator<Item = Option<&'a [u8]>>>(&mut self, values: I) -> usize {
        self.extend(values);
        self.len()
    }
}

fn set_operation<K, I, R>(
    set: &mut PlIndexSet<K>,
    set2: &mut PlIndexSet<K>,
    a: I,
    b: I,
    out: &mut R,
    set_op: SetOperation,
) -> usize
where
    K: Eq + Hash + Copy,
    I: IntoIterator<Item = K>,
    R: MaterializeValues<K>,
{
    set.clear();
    let a = a.into_iter();
    let b = b.into_iter();

    match set_op {
        SetOperation::Intersection => {
            set2.clear();
            set.extend(a);
            set2.extend(b);
            out.extend_buf(set.intersection(set2).copied())
        }
        SetOperation::Union => {
            set.extend(a);
            set.extend(b);
            out.extend_buf(set.drain(..))
        }
        SetOperation::Difference => {
            set.extend(a);
            for v in b {
                set.remove(&v);
            }
            out.extend_buf(set.drain(..))
        }
        SetOperation::SymmetricDifference => {
            set2.clear();
            // We could speed this up, but implementing ourselves, but we need to have a clonable
            // iterator as we need 2 passes
            set.extend(a);
            set2.extend(b);
            out.extend_buf(set.symmetric_difference(set2).copied())
        }
    }
}

fn copied_opt<T: Copy>(v: Option<&T>) -> Option<T> {
    v.copied()
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SetOperation {
    Intersection,
    Union,
    Difference,
    SymmetricDifference,
}

impl Display for SetOperation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            SetOperation::Intersection => "intersection",
            SetOperation::Union => "union",
            SetOperation::Difference => "difference",
            SetOperation::SymmetricDifference => "symmetric_difference",
        };
        write!(f, "{s}")
    }
}

fn primitive<T>(
    a: &PrimitiveArray<T>,
    b: &PrimitiveArray<T>,
    offsets_a: &[i64],
    offsets_b: &[i64],
    set_op: SetOperation,
    validity: Option<Bitmap>,
) -> ListArray<i64>
where
    T: NativeType + Hash + Copy + Eq,
{
    assert_eq!(offsets_a.len(), offsets_b.len());

    let mut set = Default::default();
    let mut set2 = Default::default();

    let mut values_out = MutablePrimitiveArray::with_capacity(std::cmp::max(
        *offsets_a.last().unwrap(),
        *offsets_b.last().unwrap(),
    ) as usize);
    let mut offsets = Vec::with_capacity(std::cmp::max(offsets_a.len(), offsets_b.len()));
    offsets.push(0i64);

    for i in 1..offsets_a.len() {
        unsafe {
            let start_a = *offsets_a.get_unchecked(i - 1) as usize;
            let end_a = *offsets_a.get_unchecked(i) as usize;

            let start_b = *offsets_b.get_unchecked(i - 1) as usize;
            let end_b = *offsets_b.get_unchecked(i) as usize;

            // going via skip iterator instead of slice doesn't heap alloc nor trigger a bitcount
            let a_iter = a
                .into_iter()
                .skip(start_a)
                .take(end_a - start_a)
                .map(copied_opt);
            let b_iter = b
                .into_iter()
                .skip(start_b)
                .take(end_b - start_b)
                .map(copied_opt);

            let offset =
                set_operation(&mut set, &mut set2, a_iter, b_iter, &mut values_out, set_op);
            offsets.push(offset as i64);
        }
    }
    let offsets = unsafe { OffsetsBuffer::new_unchecked(offsets.into()) };
    let dtype = ListArray::<i64>::default_datatype(values_out.data_type().clone());

    let values: PrimitiveArray<T> = values_out.into();
    ListArray::new(dtype, offsets, values.boxed(), validity)
}

fn binary(
    a: &BinaryArray<i64>,
    b: &BinaryArray<i64>,
    offsets_a: &[i64],
    offsets_b: &[i64],
    set_op: SetOperation,
    validity: Option<Bitmap>,
    as_utf8: bool,
) -> ListArray<i64> {
    assert_eq!(offsets_a.len(), offsets_b.len());

    let mut set = Default::default();
    let mut set2 = Default::default();

    let mut values_out = MutableBinaryArray::with_capacity(std::cmp::max(
        *offsets_a.last().unwrap(),
        *offsets_b.last().unwrap(),
    ) as usize);
    let mut offsets = Vec::with_capacity(std::cmp::max(offsets_a.len(), offsets_b.len()));
    offsets.push(0i64);

    for i in 1..offsets_a.len() {
        unsafe {
            let start_a = *offsets_a.get_unchecked(i - 1) as usize;
            let end_a = *offsets_a.get_unchecked(i) as usize;

            let start_b = *offsets_b.get_unchecked(i - 1) as usize;
            let end_b = *offsets_b.get_unchecked(i) as usize;

            // going via skip iterator instead of slice doesn't heap alloc nor trigger a bitcount
            let a_iter = a.into_iter().skip(start_a).take(end_a - start_a);
            let b_iter = b.into_iter().skip(start_b).take(end_b - start_b);

            let offset =
                set_operation(&mut set, &mut set2, a_iter, b_iter, &mut values_out, set_op);
            offsets.push(offset as i64);
        }
    }
    let offsets = unsafe { OffsetsBuffer::new_unchecked(offsets.into()) };
    let values: BinaryArray<i64> = values_out.into();

    if as_utf8 {
        let values = unsafe {
            Utf8Array::<i64>::new_unchecked(
                ArrowDataType::LargeUtf8,
                values.offsets().clone(),
                values.values().clone(),
                values.validity().cloned(),
            )
        };
        let dtype = ListArray::<i64>::default_datatype(values.data_type().clone());
        ListArray::new(dtype, offsets, values.boxed(), validity)
    } else {
        let dtype = ListArray::<i64>::default_datatype(values.data_type().clone());
        ListArray::new(dtype, offsets, values.boxed(), validity)
    }
}

fn utf8_to_binary(arr: &Utf8Array<i64>) -> BinaryArray<i64> {
    BinaryArray::<i64>::new(
        ArrowDataType::LargeBinary,
        arr.offsets().clone(),
        arr.values().clone(),
        arr.validity().cloned(),
    )
}

fn array_set_operation(
    a: &ListArray<i64>,
    b: &ListArray<i64>,
    set_op: SetOperation,
) -> ListArray<i64> {
    let offsets_a = a.offsets().as_slice();
    let offsets_b = b.offsets().as_slice();

    let values_a = a.values();
    let values_b = b.values();
    assert_eq!(values_a.data_type(), values_b.data_type());

    let dtype = values_b.data_type();
    let validity = combine_validities_and(a.validity(), b.validity());

    match dtype {
        ArrowDataType::LargeUtf8 => {
            let a = values_a.as_any().downcast_ref::<Utf8Array<i64>>().unwrap();
            let b = values_b.as_any().downcast_ref::<Utf8Array<i64>>().unwrap();

            let a = utf8_to_binary(a);
            let b = utf8_to_binary(b);
            binary(&a, &b, offsets_a, offsets_b, set_op, validity, true)
        }
        ArrowDataType::LargeBinary => {
            let a = values_a
                .as_any()
                .downcast_ref::<BinaryArray<i64>>()
                .unwrap();
            let b = values_b
                .as_any()
                .downcast_ref::<BinaryArray<i64>>()
                .unwrap();
            binary(a, b, offsets_a, offsets_b, set_op, validity, false)
        }
        ArrowDataType::Boolean => {
            todo!("boolean type not yet supported in list union operations")
        }
        _ => {
            with_match_physical_integer_type!(dtype.into(), |$T| {
                let a = values_a.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
                let b = values_b.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();

                primitive(&a, &b, offsets_a, offsets_b, set_op, validity)
            })
        }
    }
}

pub fn list_set_operation(a: &ListChunked, b: &ListChunked, set_op: SetOperation) -> ListChunked {
    let (a, b) = align_chunks_binary(a, b);

    // no downcasting needed as lists
    // already have logical types
    let chunks = a
        .downcast_iter()
        .zip(b.downcast_iter())
        .map(|(a, b)| array_set_operation(a, b, set_op).boxed())
        .collect::<Vec<_>>();

    // safety: dtypes are correct
    unsafe { a.with_chunks(chunks) }
}
