pub fn select_unpredictable<T>(cond: bool, true_val: T, false_val: T) -> T {
    #[cfg(feature = "nightly")]
    {
        cond.select_unpredictable(true_val, false_val)
    }

    #[cfg(not(feature = "nightly"))]
    {
        return if cond { true_val } else { false_val };
    }
}
