/// # Safety
/// This may break aliasing rules, make sure you are the only owner.
#[allow(clippy::mut_from_ref)]
pub unsafe fn to_mutable_slice<T: Copy>(s: &[T]) -> &mut [T] {
    let ptr = s.as_ptr() as *mut T;
    let len = s.len();
    std::slice::from_raw_parts_mut(ptr, len)
}
