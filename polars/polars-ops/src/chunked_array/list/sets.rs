use std::fmt::{Display, Formatter};
use std::hash::Hash;

use arrow::array::{ListArray, MutableArray, MutablePrimitiveArray, PrimitiveArray};
use arrow::bitmap::Bitmap;
use arrow::offset::OffsetsBuffer;
use arrow::types::NativeType;
use polars_arrow::utils::combine_validities_and;
use polars_core::prelude::*;
use polars_core::utils::align_chunks_binary;
use polars_core::with_match_physical_integer_type;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

struct Intersection<'a, T, I: Iterator<Item = T>> {
    // iterator of the first set
    iter: I,
    // the second set
    other: &'a PlHashSet<T>,
}

impl<'a, T, I> Iterator for Intersection<'a, T, I>
where
    T: Eq + Hash,
    I: Iterator<Item = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        loop {
            let elt = self.iter.next()?;
            if self.other.contains(&elt) {
                return Some(elt);
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper)
    }
}

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

fn set_operation<K, I, R>(
    set: &mut PlHashSet<K>,
    a: I,
    b: I,
    out: &mut R,
    set_type: SetOperation,
) -> usize
where
    K: Eq + Hash + Copy,
    I: IntoIterator<Item = K>,
    R: MaterializeValues<K>,
{
    let a = a.into_iter();
    let b = b.into_iter();

    match set_type {
        SetOperation::Intersection => {
            let (smaller, larger) = if a.size_hint().0 <= b.size_hint().0 {
                (a, b)
            } else {
                (b, a)
            };
            set.extend(smaller);
            let iter = Intersection {
                iter: larger,
                other: set,
            };
            out.extend_buf(iter)
        }
        SetOperation::Union => {
            set.extend(a);
            set.extend(b);
            out.extend_buf(set.drain())
        }
        SetOperation::Difference => {
            set.extend(a);
            for v in b {
                set.remove(&v);
            }
            out.extend_buf(set.drain())
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
}

impl Display for SetOperation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            SetOperation::Intersection => "intersection",
            SetOperation::Union => "union",
            SetOperation::Difference => "difference",
        };
        write!(f, "{s}")
    }
}

fn primitive<T>(
    a: &PrimitiveArray<T>,
    b: &PrimitiveArray<T>,
    offsets_a: &[i64],
    offsets_b: &[i64],
    set_type: SetOperation,
    validity: Option<Bitmap>,
) -> ListArray<i64>
where
    T: NativeType + Hash + Copy + Eq,
{
    assert_eq!(offsets_a.len(), offsets_b.len());

    let mut set = Default::default();

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

            let offset = set_operation(&mut set, a_iter, b_iter, &mut values_out, set_type);
            offsets.push(offset as i64);
        }
    }
    let offsets = unsafe { OffsetsBuffer::new_unchecked(offsets.into()) };
    let dtype = ListArray::<i64>::default_datatype(values_out.data_type().clone());

    let values: PrimitiveArray<T> = values_out.into();
    ListArray::new(dtype, offsets, values.boxed(), validity)
}

fn array_set_operation(
    a: &ListArray<i64>,
    b: &ListArray<i64>,
    set_type: SetOperation,
) -> ListArray<i64> {
    let offsets_a = a.offsets().as_slice();
    let offsets_b = b.offsets().as_slice();

    let values_a = a.values();
    let values_b = b.values();
    assert_eq!(values_a.data_type(), values_b.data_type());

    let dtype = values_b.data_type();
    let validity = combine_validities_and(a.validity(), b.validity());

    with_match_physical_integer_type!(dtype.into(), |$T| {
        let a = values_a.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
        let b = values_b.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();

        primitive(&a, &b, offsets_a, offsets_b, set_type, validity)
    })
}

pub fn list_set_operation(a: &ListChunked, b: &ListChunked, set_type: SetOperation) -> ListChunked {
    let (a, b) = align_chunks_binary(a, b);

    // no downcasting needed as lists
    // already have logical types
    let chunks = a
        .downcast_iter()
        .zip(b.downcast_iter())
        .map(|(a, b)| array_set_operation(a, b, set_type).boxed())
        .collect::<Vec<_>>();

    // safety: dtypes are correct
    unsafe { a.with_chunks(chunks) }
}
