use std::hash::Hash;

use arrow::array::{ListArray, MutableArray, MutablePrimitiveArray, PrimitiveArray};
use arrow::bitmap::Bitmap;
use arrow::offset::OffsetsBuffer;
use arrow::types::NativeType;
use polars_core::prelude::*;

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
    set_a: &mut PlHashSet<K>,
    set_b: &mut PlHashSet<K>,
    a: I,
    b: I,
    out: &mut R,
    set_type: SetType,
) -> usize
where
    K: Eq + Hash + Copy,
    I: IntoIterator<Item = K>,
    R: MaterializeValues<K>,
{
    set_a.clear();
    set_b.clear();
    set_a.extend(a);
    set_b.extend(b);

    match set_type {
        SetType::Intersection => {
            let iter = set_a.intersection(set_b).copied();
            out.extend_buf(iter)
        }
        SetType::Union => {
            let iter = set_a.union(set_b).copied();
            out.extend_buf(iter)
        }
        SetType::Difference => {
            let iter = set_a.difference(set_b).copied();
            out.extend_buf(iter)
        }
    }
}

fn copied_opt<T: Copy>(v: Option<&T>) -> Option<T> {
    v.copied()
}

#[derive(Copy, Clone)]
enum SetType {
    Intersection,
    Union,
    Difference,
}

fn primitive<T>(
    a: &PrimitiveArray<T>,
    b: &PrimitiveArray<T>,
    offsets_a: &[i64],
    offsets_b: &[i64],
    set_type: SetType,
    validity: Option<Bitmap>,
) -> ListArray<i64>
where
    T: NativeType + Hash + Copy + Eq,
{
    assert_eq!(offsets_a.len(), offsets_b.len());

    let mut set_a = Default::default();
    let mut set_b = Default::default();

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

            let offset = set_operation(
                &mut set_a,
                &mut set_b,
                a_iter,
                b_iter,
                &mut values_out,
                set_type,
            );
            offsets.push(offset as i64);
        }
    }
    let offsets = unsafe { OffsetsBuffer::new_unchecked(offsets.into()) };
    let dtype = ListArray::<i64>::default_datatype(values_out.data_type().clone());

    let values: PrimitiveArray<T> = values_out.into();
    ListArray::new(dtype, offsets, values.boxed(), validity)
}

fn array_to_primitive(a: &ListArray<i64>, b: &ListArray<i64>, set_type: SetType) {
    let offset_a = a.offsets().as_slice();
    let offset_b = b.offsets().as_slice();
}
