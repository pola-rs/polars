use std::fmt::{Display, Formatter};
use std::hash::Hash;

use arrow::types::NativeType;
use polars_array::bitmap::combine_validities_and;
use polars_array::builder::StaticArrayBuilder;
use polars_array::{
    PlArrayType, PlBinaryViewArray, PlBinaryViewArrayBuilder, PlBitmap, PlListArray,
    PlPrimitiveArray, PlPrimitiveArrayBuilder, PlUtf8ViewArray,
};
use polars_buffer::Buffer;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_type;
use polars_utils::total_ord::{ToTotalOrd, TotalEq, TotalHash, TotalOrdWrap};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

trait MaterializeValues<K> {
    // extends the iterator to the values and returns the current offset
    fn extend_buf<I: Iterator<Item = K>>(&mut self, values: I) -> usize;
}

impl<T> MaterializeValues<Option<T>> for PlPrimitiveArrayBuilder<T>
where
    T: NativeType,
{
    fn extend_buf<I: Iterator<Item = Option<T>>>(&mut self, values: I) -> usize {
        for value in values {
            self.push(value);
        }
        StaticArrayBuilder::len(self)
    }
}

impl<T> MaterializeValues<TotalOrdWrap<Option<T>>> for PlPrimitiveArrayBuilder<T>
where
    T: NativeType,
{
    fn extend_buf<I: Iterator<Item = TotalOrdWrap<Option<T>>>>(&mut self, values: I) -> usize {
        self.extend_buf(values.map(|x| x.0))
    }
}

impl<'a> MaterializeValues<Option<&'a [u8]>> for PlBinaryViewArrayBuilder {
    fn extend_buf<I: Iterator<Item = Option<&'a [u8]>>>(&mut self, values: I) -> usize {
        for value in values {
            self.push(value);
        }
        StaticArrayBuilder::len(self)
    }
}

#[allow(clippy::too_many_arguments)]
fn set_operation<I, J, K, R>(
    set: &mut PlIndexSet<K>,
    set2: &mut PlIndexSet<K>,
    a: &mut I,
    b: &mut J,
    out: &mut R,
    set_op: SetOperation,
    broadcast_rhs: bool,
) -> usize
where
    K: Eq + Hash + Copy,
    I: Iterator<Item = K>,
    J: Iterator<Item = K>,
    R: MaterializeValues<K>,
{
    set.clear();

    match set_op {
        SetOperation::Intersection => {
            set.extend(a);
            // If broadcast `set2` should already be filled.
            if !broadcast_rhs {
                set2.clear();
                set2.extend(b);
            }
            out.extend_buf(set.intersection(set2).copied())
        },
        SetOperation::Union => {
            set.extend(a);
            set.extend(b);
            out.extend_buf(set.drain(..))
        },
        SetOperation::Difference => {
            set.extend(a);
            for v in b {
                set.swap_remove(&v);
            }
            out.extend_buf(set.drain(..))
        },
        SetOperation::SymmetricDifference => {
            // If broadcast `set2` should already be filled.
            if !broadcast_rhs {
                set2.clear();
                set2.extend(b);
            }
            // We could speed this up, but implementing ourselves, but we need to have a cloneable
            // iterator as we need 2 passes
            set.extend(a);
            out.extend_buf(set.symmetric_difference(set2).copied())
        },
    }
}

/// The element as the key it is looked up by, which is what makes a float's `NaN` compare equal to
/// itself.
///
/// The elements arrive by value rather than by reference: a `PlPrimitiveArray` reads a chunk that
/// repeats one value without a slot per element to point at.
fn wrapper_opt<T: Copy + TotalEq + TotalHash>(
    v: Option<T>,
) -> <Option<T> as ToTotalOrd>::TotalOrdItem {
    v.to_total_ord()
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[strum(serialize_all = "snake_case")]
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
    a: &PlPrimitiveArray<T>,
    b: &PlPrimitiveArray<T>,
    offsets_a: &[u64],
    offsets_b: &[u64],
    set_op: SetOperation,
    validity: Option<PlBitmap>,
) -> PolarsResult<PlListArray>
where
    T: NativeType + TotalHash + TotalEq + Copy + ToTotalOrd,
    <Option<T> as ToTotalOrd>::TotalOrdItem: Hash + Eq + Copy,
{
    let broadcast_lhs = offsets_a.len() == 2;
    let broadcast_rhs = offsets_b.len() == 2;

    let mut set = Default::default();
    let mut set2: PlIndexSet<<Option<T> as ToTotalOrd>::TotalOrdItem> = Default::default();

    let mut values_out = PlPrimitiveArrayBuilder::with_capacity(std::cmp::max(
        *offsets_a.last().unwrap(),
        *offsets_b.last().unwrap(),
    ) as usize);
    let mut offsets = Vec::with_capacity(std::cmp::max(offsets_a.len(), offsets_b.len()));
    offsets.push(0u64);

    let offsets_slice = if offsets_a.len() > offsets_b.len() {
        offsets_a
    } else {
        offsets_b
    };
    let first_a = offsets_a[0];
    let second_a = offsets_a[1];
    let first_b = offsets_b[0];
    let second_b = offsets_b[1];
    if broadcast_rhs {
        set2.extend(
            b.into_iter()
                .skip(first_b as usize)
                .take(second_b as usize - first_b as usize)
                .map(wrapper_opt),
        );
    }

    let mut iter_a = a.into_iter().skip(first_a as usize);
    let mut iter_b = b.into_iter().skip(first_b as usize);

    for i in 1..offsets_slice.len() {
        // If we go OOB we take the first element as we are then broadcasting.
        let start_a = *offsets_a.get(i - 1).unwrap_or(&first_a) as usize;
        let end_a = *offsets_a.get(i).unwrap_or(&second_a) as usize;

        let start_b = *offsets_b.get(i - 1).unwrap_or(&first_b) as usize;
        let end_b = *offsets_b.get(i).unwrap_or(&second_b) as usize;

        let mut iter_a_broadcast = iter_a.clone();
        let mut iter_b_broadcast = iter_b.clone();

        // The branches are the same every loop.
        // We rely on branch prediction here.
        let mut iter_a = if broadcast_lhs {
            iter_a_broadcast
                .by_ref()
                .take(second_a as usize - first_a as usize)
                .map(wrapper_opt)
        } else {
            iter_a.by_ref().take(end_a - start_a).map(wrapper_opt)
        };
        let mut iter_b = if broadcast_rhs {
            iter_b_broadcast
                .by_ref()
                .take(second_b as usize - first_b as usize)
                .map(wrapper_opt)
        } else {
            iter_b.by_ref().take(end_b - start_b).map(wrapper_opt)
        };

        let offset = set_operation(
            &mut set,
            &mut set2,
            &mut iter_a,
            &mut iter_b,
            &mut values_out,
            set_op,
            broadcast_rhs,
        );

        assert!(iter_a.next().is_none());
        if !broadcast_rhs || matches!(set_op, SetOperation::Union | SetOperation::Difference) {
            assert!(iter_b.next().is_none());
        };

        offsets.push(offset as u64);
    }
    let length = offsets.len() - 1;
    Ok(PlListArray::new(
        Box::new(values_out.freeze()),
        Buffer::from(offsets),
        length,
        validity,
    ))
}

fn binary(
    a: &PlBinaryViewArray,
    b: &PlBinaryViewArray,
    offsets_a: &[u64],
    offsets_b: &[u64],
    set_op: SetOperation,
    validity: Option<PlBitmap>,
    as_utf8: bool,
) -> PolarsResult<PlListArray> {
    let broadcast_lhs = offsets_a.len() == 2;
    let broadcast_rhs = offsets_b.len() == 2;
    let mut set: PlIndexSet<Option<&[u8]>> = Default::default();
    let mut set2: PlIndexSet<Option<&[u8]>> = Default::default();

    let mut values_out = PlBinaryViewArrayBuilder::with_capacity(std::cmp::max(
        *offsets_a.last().unwrap(),
        *offsets_b.last().unwrap(),
    ) as usize);
    let mut offsets = Vec::with_capacity(std::cmp::max(offsets_a.len(), offsets_b.len()));
    offsets.push(0u64);

    let offsets_slice = if offsets_a.len() > offsets_b.len() {
        offsets_a
    } else {
        offsets_b
    };
    let first_a = offsets_a[0];
    let second_a = offsets_a[1];
    let first_b = offsets_b[0];
    let second_b = offsets_b[1];

    if broadcast_rhs {
        // set2.extend(b_iter)
        set2.extend(
            b.into_iter()
                .skip(first_b as usize)
                .take(second_b as usize - first_b as usize),
        );
    }

    let mut iter_a = a.into_iter().skip(first_a as usize);
    let mut iter_b = b.into_iter().skip(first_b as usize);

    for i in 1..offsets_slice.len() {
        // If we go OOB we take the first element as we are then broadcasting.
        let start_a = *offsets_a.get(i - 1).unwrap_or(&first_a) as usize;
        let end_a = *offsets_a.get(i).unwrap_or(&second_a) as usize;

        let start_b = *offsets_b.get(i - 1).unwrap_or(&first_b) as usize;
        let end_b = *offsets_b.get(i).unwrap_or(&second_b) as usize;

        let mut iter_a_broadcast = iter_a.clone();
        let mut iter_b_broadcast = iter_b.clone();

        // The branches are the same every loop.
        // We rely on branch prediction here.
        let mut iter_a = if broadcast_lhs {
            iter_a_broadcast
                .by_ref()
                .take(second_a as usize - first_a as usize)
        } else {
            iter_a.by_ref().take(end_a - start_a)
        };
        let mut iter_b = if broadcast_rhs {
            iter_b_broadcast
                .by_ref()
                .take(second_b as usize - first_b as usize)
        } else {
            iter_b.by_ref().take(end_b - start_b)
        };

        let offset = set_operation(
            &mut set,
            &mut set2,
            &mut iter_a,
            &mut iter_b,
            &mut values_out,
            set_op,
            broadcast_rhs,
        );

        assert!(iter_a.next().is_none());
        if !broadcast_rhs || matches!(set_op, SetOperation::Union | SetOperation::Difference) {
            assert!(iter_b.next().is_none());
        };

        offsets.push(offset as u64);
    }
    let length = offsets.len() - 1;
    let values = values_out.freeze();

    // The values are read out of two arrays of the same type, so what went in is what comes back.
    let values: Box<dyn PlArray> = if as_utf8 {
        Box::new(unsafe { PlUtf8ViewArray::from_binview_unchecked(values) })
    } else {
        Box::new(values)
    };

    Ok(PlListArray::new(
        values,
        Buffer::from(offsets),
        length,
        validity,
    ))
}

fn array_set_operation(
    a: &PlListArray,
    b: &PlListArray,
    set_op: SetOperation,
    inner_dtype: &DataType,
) -> PolarsResult<PlListArray> {
    // The kernels below read the offsets as a slice and walk the values one element at a time, so
    // a chunk whose offsets repeat is written out first.
    // TODO(polars-array-scalar): a scalar chunk stands for one list, which could be reduced once
    // rather than written out.
    let a = a.to_flat();
    let b = b.to_flat();

    // `Flat` is what says these hold one range per element: the array's own `flat_offsets` asks a
    // predicate that a list of a *single* element answers `Scalar` to, its two offsets being both
    // one range and the range of its one element.
    let offsets_a = a.offsets().as_slice();
    let offsets_b = b.offsets().as_slice();

    let values_a = a.values();
    let values_b = b.values();
    assert_eq!(values_a.array_type(), values_b.array_type());

    let validity = combine_validities_and(a.as_array().validity(), b.as_array().validity());

    match inner_dtype {
        // The set is taken over the bytes either way; what comes back out is the strings they
        // were, which is what `as_utf8` says.
        DataType::String => binary(
            downcast::<PlUtf8ViewArray>(values_a).as_binview(),
            downcast::<PlUtf8ViewArray>(values_b).as_binview(),
            offsets_a,
            offsets_b,
            set_op,
            validity,
            true,
        ),
        DataType::Binary => binary(
            downcast(values_a),
            downcast(values_b),
            offsets_a,
            offsets_b,
            set_op,
            validity,
            false,
        ),
        DataType::Boolean => {
            polars_bail!(InvalidOperation: "boolean type not yet supported in list 'set' operations")
        },
        dtype => {
            with_match_physical_numeric_type!(dtype, |$T| {
                primitive(
                    downcast::<PlPrimitiveArray<$T>>(values_a),
                    downcast::<PlPrimitiveArray<$T>>(values_b),
                    offsets_a,
                    offsets_b,
                    set_op,
                    validity,
                )
            })
        },
    }
}

/// The array behind a chunk whose type is already known.
fn downcast<A: 'static>(array: &dyn PlArray) -> &A {
    array.as_any().downcast_ref::<A>().unwrap()
}

pub fn list_set_operation(
    a: &ListChunked,
    b: &ListChunked,
    set_op: SetOperation,
) -> PolarsResult<ListChunked> {
    polars_ensure!(a.len() == b.len() || b.len() == 1 || a.len() == 1, ShapeMismatch: "column lengths don't match");
    polars_ensure!(a.dtype() == b.dtype(), InvalidOperation: "cannot do 'set' operation on dtypes: {} and {}", a.dtype(), b.dtype());
    let mut a = a.clone();
    let mut b = b.clone();
    if a.len() != b.len() {
        a.rechunk_mut();
        b.rechunk_mut();
    }

    // We will OOB in the kernel otherwise.
    a.prune_empty_chunks();
    b.prune_empty_chunks();

    // A chunk carries no data type of its own, so which kernel the values want is asked of the
    // column rather than read off the array.
    let inner_dtype = a.inner_dtype().to_physical();

    // we use the unsafe variant because we want to keep the nested logical types type.
    unsafe {
        arity::try_binary_unchecked_same_type(
            &a,
            &b,
            |a, b| {
                array_set_operation(a, b, set_op, &inner_dtype)
                    .map(|arr| Box::new(arr) as Box<dyn PlArray>)
            },
            false,
            false,
        )
    }
}
