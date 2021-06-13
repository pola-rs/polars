pub trait IsValid {
    /// # Safety
    /// no bound checks
    unsafe fn is_valid_unchecked(&self, i: usize) -> bool;

    /// # Safety
    /// no bound checks
    unsafe fn is_null_unchecked(&self, i: usize) -> bool;
}
