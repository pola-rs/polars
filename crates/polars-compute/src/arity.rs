use arrow::array::PrimitiveArray;
use arrow::compute::utils::combine_validities_and;
use arrow::types::NativeType;

/// Applies a function to all the values (regardless of nullability).
///
/// May re-use the memory of the array if possible.
pub fn prim_unary_values<I, O, F>(mut arr: PrimitiveArray<I>, op: F) -> PrimitiveArray<O>
where
    I: NativeType,
    O: NativeType,
    F: Fn(I) -> O,
{
    // Reuse memory if possible.
    if std::mem::size_of::<I>() == std::mem::size_of::<O>()
        && std::mem::align_of::<I>() == std::mem::align_of::<O>()
    {
        if let Some(values) = arr.get_mut_values() {
            for v in values {
                // SAFETY: checked same size & alignment, NativeType is always Pod.
                *v = unsafe { std::mem::transmute_copy::<O, I>(&(op(*v))) };
            }
            return arr.transmute::<O>();
        }
    }

    let ret = arr.values_iter().map(|x| op(*x)).collect();
    PrimitiveArray::from_vec(ret).with_validity(arr.take_validity())
}

/// Apply a binary function to all the values (regardless of nullability)
/// in (lhs, rhs). Combines the validities with a bitand.
///
/// May re-use the memory of one of its arguments if possible.
pub fn prim_binary_values<L, R, O, F>(
    mut lhs: PrimitiveArray<L>,
    mut rhs: PrimitiveArray<R>,
    op: F,
) -> PrimitiveArray<O>
where
    L: NativeType,
    R: NativeType,
    O: NativeType,
    F: Fn(L, R) -> O,
{
    assert_eq!(lhs.len(), rhs.len());

    let validity = combine_validities_and(lhs.validity(), rhs.validity());

    // Reuse memory if possible.
    if std::mem::size_of::<L>() == std::mem::size_of::<O>()
        && std::mem::align_of::<L>() == std::mem::align_of::<O>()
    {
        if let Some(lv) = lhs.get_mut_values() {
            let pairs = lv.iter_mut().zip(rhs.values_iter());
            for (l, r) in pairs {
                // SAFETY: checked same size & alignment, NativeType is always Pod.
                *l = unsafe { std::mem::transmute_copy::<O, L>(&(op(*l, *r))) };
            }
            return lhs.transmute::<O>().with_validity(validity);
        }
    }
    if std::mem::size_of::<R>() == std::mem::size_of::<O>()
        && std::mem::align_of::<R>() == std::mem::align_of::<O>()
    {
        if let Some(rv) = rhs.get_mut_values() {
            let pairs = lhs.values_iter().zip(rv.iter_mut());
            for (l, r) in pairs {
                // SAFETY: checked same size & alignment, NativeType is always Pod.
                *r = unsafe { std::mem::transmute_copy::<O, R>(&(op(*l, *r))) };
            }
            return rhs.transmute::<O>().with_validity(validity);
        }
    }

    let pairs = lhs.values_iter().zip(rhs.values_iter());
    let ret = pairs.map(|(l, r)| op(*l, *r)).collect();
    PrimitiveArray::from_vec(ret).with_validity(validity)
}
