use super::{BooleanArray, MutableBooleanArray};

impl<P: AsRef<[Option<bool>]>> From<P> for BooleanArray {
    fn from(slice: P) -> Self {
        MutableBooleanArray::from(slice).into()
    }
}

impl<Ptr: std::borrow::Borrow<Option<bool>>> FromIterator<Ptr> for BooleanArray {
    fn from_iter<I: IntoIterator<Item = Ptr>>(iter: I) -> Self {
        MutableBooleanArray::from_iter(iter).into()
    }
}
