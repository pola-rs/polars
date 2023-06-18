use num_traits::Zero;

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

// A resize that may overwrite the stack ptr
// and thus can realloc
pub trait ResizeFaster<T: Copy> {
    fn resize_and_fill(&mut self, new_len: usize, value: T);
}

impl<T: Copy + Zero + PartialEq> ResizeFaster<T> for Vec<T> {
    fn resize_and_fill(&mut self, new_len: usize, value: T) {
        if self.capacity() == 0 || value == Zero::zero() || new_len > self.len() {
            // it is faster to allocate zeroed
            // so if the capacity is 0, we alloc (value might be 0)
            *self = vec![value; new_len]
        } else {
            self.truncate(new_len);
            for v in self {
                *v = value
            }
        }
    }
}
