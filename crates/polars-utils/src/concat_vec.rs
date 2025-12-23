pub trait ConcatVec<T>: private::Sealed {
    /// `concat()` for `Vec<Vec<T>>` that avoids clones if `self` is length-1.
    fn concat_vec(self) -> Vec<T>;
}

impl<T> private::Sealed for Vec<Vec<T>> {}

impl<T> ConcatVec<T> for Vec<Vec<T>>
where
    T: Copy,
{
    fn concat_vec(mut self) -> Vec<T> {
        if self.len() == 1 {
            self.pop().unwrap()
        } else {
            self.concat()
        }
    }
}

mod private {
    pub trait Sealed {}
}
