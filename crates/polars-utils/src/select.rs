#[cfg(feature = "nightly")]
pub fn select_unpredictable<T>(cond: bool, true_val: T, false_val: T) -> T {
    core::hint::select_unpredictable(cond, true_val, false_val)
}

#[cfg(not(feature = "nightly"))]
pub fn select_unpredictable<T>(cond: bool, true_val: T, false_val: T) -> T {
    if cond { true_val } else { false_val }
}
