use std::error::Error;

pub trait FallibleIterator<E: Error>: Iterator {
    fn get_result(&mut self) -> Result<(), E>;
}

pub trait FromFallibleIterator<A, E: Error>: Sized {
    fn from_fallible_iter<F: FallibleIterator<E, Item = A>>(iter: F) -> Result<Self, E>;
}

impl<A, T: FromIterator<A>, E: Error> FromFallibleIterator<A, E> for T {
    fn from_fallible_iter<F: FallibleIterator<E, Item = A>>(mut iter: F) -> Result<T, E> {
        let out = T::from_iter(&mut iter);
        iter.get_result()?;
        Ok(out)
    }
}
