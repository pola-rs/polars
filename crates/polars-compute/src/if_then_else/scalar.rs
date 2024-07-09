use std::mem::MaybeUninit;

pub fn if_then_else_scalar_rest<T: Copy>(
    mask: u64,
    if_true: &[T],
    if_false: &[T],
    out: &mut [MaybeUninit<T>],
) {
    assert!(if_true.len() == out.len()); // Removes bounds checks in inner loop.
    let true_it = if_true.iter().copied();
    let false_it = if_false.iter().copied();
    for (i, (t, f)) in true_it.zip(false_it).enumerate() {
        let src = if (mask >> i) & 1 != 0 { t } else { f };
        out[i] = MaybeUninit::new(src);
    }
}

pub fn if_then_else_broadcast_false_scalar_rest<T: Copy>(
    mask: u64,
    if_true: &[T],
    if_false: T,
    out: &mut [MaybeUninit<T>],
) {
    assert!(if_true.len() == out.len()); // Removes bounds checks in inner loop.
    let true_it = if_true.iter().copied();
    for (i, t) in true_it.enumerate() {
        let src = if (mask >> i) & 1 != 0 { t } else { if_false };
        out[i] = MaybeUninit::new(src);
    }
}

pub fn if_then_else_broadcast_both_scalar_rest<T: Copy>(
    mask: u64,
    if_true: T,
    if_false: T,
    out: &mut [MaybeUninit<T>],
) {
    for (i, dst) in out.iter_mut().enumerate() {
        let src = if (mask >> i) & 1 != 0 {
            if_true
        } else {
            if_false
        };
        *dst = MaybeUninit::new(src);
    }
}

pub fn if_then_else_scalar_64<T: Copy>(
    mask: u64,
    if_true: &[T; 64],
    if_false: &[T; 64],
    out: &mut [MaybeUninit<T>; 64],
) {
    // This generated the best autovectorized code on ARM, and branchless everywhere.
    if_then_else_scalar_rest(mask, if_true, if_false, out)
}

pub fn if_then_else_broadcast_false_scalar_64<T: Copy>(
    mask: u64,
    if_true: &[T; 64],
    if_false: T,
    out: &mut [MaybeUninit<T>; 64],
) {
    // This generated the best autovectorized code on ARM, and branchless everywhere.
    if_then_else_broadcast_false_scalar_rest(mask, if_true, if_false, out)
}

pub fn if_then_else_broadcast_both_scalar_64<T: Copy>(
    mask: u64,
    if_true: T,
    if_false: T,
    out: &mut [MaybeUninit<T>; 64],
) {
    // This generated the best autovectorized code on ARM, and branchless everywhere.
    if_then_else_broadcast_both_scalar_rest(mask, if_true, if_false, out)
}
