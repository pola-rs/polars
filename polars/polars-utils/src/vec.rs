pub trait IntoRawParts<T> {
    fn into_raw_parts(self) -> (*mut T, usize, usize);

    // doesn't take ownership
    fn raw_parts(&self) -> (*mut T, usize, usize);
}

impl<T> IntoRawParts<T> for Vec<T> {
    fn into_raw_parts(self) -> (*mut T, usize, usize) {
        let mut me = std::mem::ManuallyDrop::new(self);
        (me.as_mut_ptr(), me.len(), me.capacity())
    }

    fn raw_parts(&self) -> (*mut T, usize, usize) {
        (self.as_ptr() as *mut T, self.len(), self.capacity())
    }
}
