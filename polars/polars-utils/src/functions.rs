// Faster than collecting from a flattened iterator.
pub fn flatten<T: Clone, R: AsRef<[T]>>(bufs: &[R], len: Option<usize>) -> Vec<T> {
    let len = len.unwrap_or_else(|| bufs.iter().map(|b| b.as_ref().len()).sum());

    let mut out = Vec::with_capacity(len);
    for b in bufs {
        out.extend_from_slice(b.as_ref());
    }
    out
}


#[inline]
pub unsafe fn debug_unwrap<T>(item: Option<T>) -> T {
    {
        #[cfg(debug_assertions)]
        {
            // check if the type is correct
            item.unwrap()
        }
        #[cfg(not(debug_assertions))]
        unsafe {
            item.unwrap_unchecked()
        }
    }
}