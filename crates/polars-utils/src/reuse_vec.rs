/// Re-uses the memory for a vec while clearing it. Allows casting the type of
/// the vec at the same time. The stdlib specializes collect() to re-use the
/// memory.
pub fn reuse_vec<T, U>(v: Vec<T>) -> Vec<U> {
    const {
        assert!(core::mem::size_of::<T>() == core::mem::size_of::<U>());
        assert!(core::mem::align_of::<T>() == core::mem::align_of::<U>());
    }
    v.into_iter().filter_map(|_| None).collect()
}
