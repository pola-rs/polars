use std::iter::FromIterator;

use crate::types::NativeType;

use super::{MutablePrimitiveArray, PrimitiveArray};

impl<T: NativeType, P: AsRef<[Option<T>]>> From<P> for PrimitiveArray<T> {
    fn from(slice: P) -> Self {
        MutablePrimitiveArray::<T>::from(slice).into()
    }
}

impl<T: NativeType, Ptr: std::borrow::Borrow<Option<T>>> FromIterator<Ptr> for PrimitiveArray<T> {
    fn from_iter<I: IntoIterator<Item = Ptr>>(iter: I) -> Self {
        MutablePrimitiveArray::<T>::from_iter(iter).into()
    }
}
