use std::ops::Deref;

pub enum CowBox<'a, T: ?Sized> {
    Borrowed(&'a T),
    Owned(Box<T>),
}

impl<'a, T: ?Sized> Deref for CowBox<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            CowBox::Borrowed(v) => v,
            CowBox::Owned(v) => v.as_ref(),
        }
    }
}
