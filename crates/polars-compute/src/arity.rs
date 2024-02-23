use arrow::array::PrimitiveArray;
use arrow::compute::utils::combine_validities_and;
use arrow::types::NativeType;

/// To reduce codegen we use these helpers where the input and output arrays
/// may overlap. These are marked to never be inlined, this way only a single
/// unrolled kernel gets generated, even if we call it in multiple ways.
///
/// # Safety
///  - arr must point to a readable slice of length len.
///  - out must point to a writable slice of length len.
#[inline(never)]
unsafe fn ptr_apply_unary_kernel<I: Copy, O, F: Fn(I) -> O>(
    arr: *const I,
    out: *mut O,
    len: usize,
    op: F,
) {
    for i in 0..len {
        let ret = op(arr.add(i).read());
        out.add(i).write(ret);
    }
}

/// # Safety
///  - left must point to a readable slice of length len.
///  - right must point to a readable slice of length len.
///  - out must point to a writable slice of length len.
#[inline(never)]
unsafe fn ptr_apply_binary_kernel<L: Copy, R: Copy, O, F: Fn(L, R) -> O>(
    left: *const L,
    right: *const R,
    out: *mut O,
    len: usize,
    op: F,
) {
    for i in 0..len {
        let ret = op(left.add(i).read(), right.add(i).read());
        out.add(i).write(ret);
    }
}

/// Applies a function to all the values (regardless of nullability).
///
/// May reuse the memory of the array if possible.
pub fn prim_unary_values<I, O, F>(mut arr: PrimitiveArray<I>, op: F) -> PrimitiveArray<O>
where
    I: NativeType,
    O: NativeType,
    F: Fn(I) -> O,
{
    let len = arr.len();

    // Reuse memory if possible.
    if std::mem::size_of::<I>() == std::mem::size_of::<O>()
        && std::mem::align_of::<I>() == std::mem::align_of::<O>()
    {
        if let Some(values) = arr.get_mut_values() {
            let ptr = values.as_mut_ptr();
            // SAFETY: checked same size & alignment I/O, NativeType is always Pod.
            unsafe { ptr_apply_unary_kernel(ptr, ptr as *mut O, len, op) }
            return arr.transmute::<O>();
        }
    }

    let mut out = Vec::with_capacity(len);
    unsafe {
        // SAFETY: checked pointers point to slices of length len.
        ptr_apply_unary_kernel(arr.values().as_ptr(), out.as_mut_ptr(), len, op);
        out.set_len(len);
    }
    PrimitiveArray::from_vec(out).with_validity(arr.take_validity())
}

/// Apply a binary function to all the values (regardless of nullability)
/// in (lhs, rhs). Combines the validities with a bitand.
///
/// May reuse the memory of one of its arguments if possible.
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
    let len = lhs.len();

    let validity = combine_validities_and(lhs.validity(), rhs.validity());

    // Reuse memory if possible.
    if std::mem::size_of::<L>() == std::mem::size_of::<O>()
        && std::mem::align_of::<L>() == std::mem::align_of::<O>()
    {
        if let Some(lv) = lhs.get_mut_values() {
            let lp = lv.as_mut_ptr();
            let rp = rhs.values().as_ptr();
            unsafe {
                // SAFETY: checked same size & alignment L/O, NativeType is always Pod.
                ptr_apply_binary_kernel(lp, rp, lp as *mut O, len, op);
            }
            return lhs.transmute::<O>().with_validity(validity);
        }
    }
    if std::mem::size_of::<R>() == std::mem::size_of::<O>()
        && std::mem::align_of::<R>() == std::mem::align_of::<O>()
    {
        if let Some(rv) = rhs.get_mut_values() {
            let lp = lhs.values().as_ptr();
            let rp = rv.as_mut_ptr();
            unsafe {
                // SAFETY: checked same size & alignment R/O, NativeType is always Pod.
                ptr_apply_binary_kernel(lp, rp, rp as *mut O, len, op);
            }
            return rhs.transmute::<O>().with_validity(validity);
        }
    }

    let mut out = Vec::with_capacity(len);
    unsafe {
        // SAFETY: checked pointers point to slices of length len.
        let lp = lhs.values().as_ptr();
        let rp = rhs.values().as_ptr();
        ptr_apply_binary_kernel(lp, rp, out.as_mut_ptr(), len, op);
        out.set_len(len);
    }
    PrimitiveArray::from_vec(out).with_validity(validity)
}
