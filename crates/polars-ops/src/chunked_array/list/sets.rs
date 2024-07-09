use std::fmt::{Display, Formatter};
use std::hash::Hash;

use arrow::array::{
    Array, BinaryViewArray, ListArray, MutableArray, MutablePlBinary, MutablePrimitiveArray,
    PrimitiveArray, Utf8ViewArray,
};
use arrow::bitmap::Bitmap;
use arrow::compute::utils::combine_validities_and;
use arrow::offset::OffsetsBuffer;
use arrow::types::NativeType;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_type;
use polars_utils::total_ord::{ToTotalOrd, TotalEq, TotalHash, TotalOrdWrap};
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

impl<T> MaterializeValues<TotalOrdWrap<Option<T>>> for MutablePrimitiveArray<T>
where
    T: NativeType,
{
    fn extend_buf<I: Iterator<Item = TotalOrdWrap<Option<T>>>>(&mut self, values: I) -> usize {
        self.extend(values.map(|x| x.0));
        self.len()
    }
}

impl<'a> MaterializeValues<Option<&'a [u8]>> for MutablePlBinary {
    fn extend_buf<I: Iterator<Item = Option<&'a [u8]>>>(&mut self, values: I) -> usize {
        self.extend(values);
        self.len()
    }
}

fn set_operation<K, I, J, R>(
    set: &mut PlIndexSet<K>,
    set2: &mut PlIndexSet<K>,
    a: I,
    b: J,
    out: &mut R,
    set_op: SetOperation,
    broadcast_rhs: bool,
) -> usize
where
    K: Eq + Hash + Copy,
    I: IntoIterator<Item = K>,
    J: IntoIterator<Item = K>,
    R: MaterializeValues<K>,
{
    set.clear();
    let a = a.into_iter();
    let b = b.into_iter();

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
            // We could speed this up, but implementing ourselves, but we need to have a clonable
            // iterator as we need 2 passes
            set.extend(a);
            out.extend_buf(set.symmetric_difference(set2).copied())
        },
    }
}

fn copied_wrapper_opt<T: Copy + ToTotalOrd>(
    v: Option<&T>,
) -> <Option<T> as ToTotalOrd>::TotalOrdItem {
    v.copied().to_total_ord()
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
) -> PolarsResult<ListArray<i64>>
where
    T: NativeType + TotalHash + TotalEq + Copy + ToTotalOrd,
    <Option<T> as ToTotalOrd>::TotalOrdItem: Hash + Eq + Copy,
{
    let broadcast_lhs = offsets_a.len() == 2;
    let broadcast_rhs = offsets_b.len() == 2;

    let mut set = Default::default();
    let mut set2: PlIndexSet<<Option<T> as ToTotalOrd>::TotalOrdItem> = Default::default();

    let mut values_out = MutablePrimitiveArray::with_capacity(std::cmp::max(
        *offsets_a.last().unwrap(),
        *offsets_b.last().unwrap(),
    ) as usize);
    let mut offsets = Vec::with_capacity(std::cmp::max(offsets_a.len(), offsets_b.len()));
    offsets.push(0i64);

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
                .map(copied_wrapper_opt),
        );
    }
    for i in 1..offsets_slice.len() {
        // If we go OOB we take the first element as we are then broadcasting.
        let start_a = *offsets_a.get(i - 1).unwrap_or(&first_a) as usize;
        let end_a = *offsets_a.get(i).unwrap_or(&second_a) as usize;

        let start_b = *offsets_b.get(i - 1).unwrap_or(&first_b) as usize;
        let end_b = *offsets_b.get(i).unwrap_or(&second_b) as usize;

        // The branches are the same every loop.
        // We rely on branch prediction here.
        let offset = if broadcast_rhs {
            // going via skip iterator instead of slice doesn't heap alloc nor trigger a bitcount
            let a_iter = a
                .into_iter()
                .skip(start_a)
                .take(end_a - start_a)
                .map(copied_wrapper_opt);
            let b_iter = b
                .into_iter()
                .skip(first_b as usize)
                .take(second_b as usize - first_b as usize)
                .map(copied_wrapper_opt);
            set_operation(
                &mut set,
                &mut set2,
                a_iter,
                b_iter,
                &mut values_out,
                set_op,
                true,
            )
        } else if broadcast_lhs {
            let a_iter = a
                .into_iter()
                .skip(first_a as usize)
                .take(second_a as usize - first_a as usize)
                .map(copied_wrapper_opt);

            let b_iter = b
                .into_iter()
                .skip(start_b)
                .take(end_b - start_b)
                .map(copied_wrapper_opt);

            set_operation(
                &mut set,
                &mut set2,
                a_iter,
                b_iter,
                &mut values_out,
                set_op,
                false,
            )
        } else {
            // going via skip iterator instead of slice doesn't heap alloc nor trigger a bitcount
            let a_iter = a
                .into_iter()
                .skip(start_a)
                .take(end_a - start_a)
                .map(copied_wrapper_opt);

            let b_iter = b
                .into_iter()
                .skip(start_b)
                .take(end_b - start_b)
                .map(copied_wrapper_opt);
            set_operation(
                &mut set,
                &mut set2,
                a_iter,
                b_iter,
                &mut values_out,
                set_op,
                false,
            )
        };

        offsets.push(offset as i64);
    }
    let offsets = unsafe { OffsetsBuffer::new_unchecked(offsets.into()) };
    let dtype = ListArray::<i64>::default_datatype(values_out.data_type().clone());

    let values: PrimitiveArray<T> = values_out.into();
    Ok(ListArray::new(dtype, offsets, values.boxed(), validity))
}

fn binary(
    a: &BinaryViewArray,
    b: &BinaryViewArray,
    offsets_a: &[i64],
    offsets_b: &[i64],
    set_op: SetOperation,
    validity: Option<Bitmap>,
    as_utf8: bool,
) -> PolarsResult<ListArray<i64>> {
    let broadcast_lhs = offsets_a.len() == 2;
    let broadcast_rhs = offsets_b.len() == 2;
    let mut set = Default::default();
    let mut set2: PlIndexSet<Option<&[u8]>> = Default::default();

    let mut values_out = MutablePlBinary::with_capacity(std::cmp::max(
        *offsets_a.last().unwrap(),
        *offsets_b.last().unwrap(),
    ) as usize);
    let mut offsets = Vec::with_capacity(std::cmp::max(offsets_a.len(), offsets_b.len()));
    offsets.push(0i64);

    if broadcast_rhs {
        set2.extend(b);
    }
    let offsets_slice = if offsets_a.len() > offsets_b.len() {
        offsets_a
    } else {
        offsets_b
    };
    let first_a = offsets_a[0];
    let second_a = offsets_a[1];
    let first_b = offsets_b[0];
    let second_b = offsets_b[1];
    for i in 1..offsets_slice.len() {
        // If we go OOB we take the first element as we are then broadcasting.
        let start_a = *offsets_a.get(i - 1).unwrap_or(&first_a) as usize;
        let end_a = *offsets_a.get(i).unwrap_or(&second_a) as usize;

        let start_b = *offsets_b.get(i - 1).unwrap_or(&first_b) as usize;
        let end_b = *offsets_b.get(i).unwrap_or(&second_b) as usize;

        // The branches are the same every loop.
        // We rely on branch prediction here.
        let offset = if broadcast_rhs {
            // going via skip iterator instead of slice doesn't heap alloc nor trigger a bitcount
            let a_iter = a.into_iter().skip(start_a).take(end_a - start_a);
            let b_iter = b.into_iter();
            set_operation(
                &mut set,
                &mut set2,
                a_iter,
                b_iter,
                &mut values_out,
                set_op,
                true,
            )
        } else if broadcast_lhs {
            let a_iter = a.into_iter();
            let b_iter = b.into_iter().skip(start_b).take(end_b - start_b);
            set_operation(
                &mut set,
                &mut set2,
                a_iter,
                b_iter,
                &mut values_out,
                set_op,
                false,
            )
        } else {
            // going via skip iterator instead of slice doesn't heap alloc nor trigger a bitcount
            let a_iter = a.into_iter().skip(start_a).take(end_a - start_a);
            let b_iter = b.into_iter().skip(start_b).take(end_b - start_b);
            set_operation(
                &mut set,
                &mut set2,
                a_iter,
                b_iter,
                &mut values_out,
                set_op,
                false,
            )
        };
        offsets.push(offset as i64);
    }
    let offsets = unsafe { OffsetsBuffer::new_unchecked(offsets.into()) };
    let values = values_out.freeze();

    if as_utf8 {
        let values = unsafe { values.to_utf8view_unchecked() };
        let dtype = ListArray::<i64>::default_datatype(values.data_type().clone());
        Ok(ListArray::new(dtype, offsets, values.boxed(), validity))
    } else {
        let dtype = ListArray::<i64>::default_datatype(values.data_type().clone());
        Ok(ListArray::new(dtype, offsets, values.boxed(), validity))
    }
}

fn array_set_operation(
    a: &ListArray<i64>,
    b: &ListArray<i64>,
    set_op: SetOperation,
) -> PolarsResult<ListArray<i64>> {
    let offsets_a = a.offsets().as_slice();
    let offsets_b = b.offsets().as_slice();

    let values_a = a.values();
    let values_b = b.values();
    assert_eq!(values_a.data_type(), values_b.data_type());

    let dtype = values_b.data_type();
    let validity = combine_validities_and(a.validity(), b.validity());

    match dtype {
        ArrowDataType::Utf8View => {
            let a = values_a
                .as_any()
                .downcast_ref::<Utf8ViewArray>()
                .unwrap()
                .to_binview();
            let b = values_b
                .as_any()
                .downcast_ref::<Utf8ViewArray>()
                .unwrap()
                .to_binview();

            binary(&a, &b, offsets_a, offsets_b, set_op, validity, true)
        },
        ArrowDataType::BinaryView => {
            let a = values_a.as_any().downcast_ref::<BinaryViewArray>().unwrap();
            let b = values_b.as_any().downcast_ref::<BinaryViewArray>().unwrap();
            binary(a, b, offsets_a, offsets_b, set_op, validity, false)
        },
        ArrowDataType::Boolean => {
            polars_bail!(InvalidOperation: "boolean type not yet supported in list 'set' operations")
        },
        _ => {
            with_match_physical_numeric_type!(dtype.into(), |$T| {
                let a = values_a.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
                let b = values_b.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();

                primitive(&a, &b, offsets_a, offsets_b, set_op, validity)
            })
        },
    }
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
        a = a.rechunk();
        b = b.rechunk();
    }

    // We will OOB in the kernel otherwise.
    a.prune_empty_chunks();
    b.prune_empty_chunks();

    // Make categoricals compatible
    #[cfg(feature = "dtype-categorical")]
    if let (DataType::Categorical(_, _), DataType::Categorical(_, _)) =
        (&a.inner_dtype(), &b.inner_dtype())
    {
        (a, b) = make_list_categoricals_compatible(a, b)?;
    }

    // we use the unsafe variant because we want to keep the nested logical types type.
    unsafe {
        arity::try_binary_unchecked_same_type(
            &a,
            &b,
            |a, b| array_set_operation(a, b, set_op).map(|arr| arr.boxed()),
            false,
            false,
        )
    }
}
