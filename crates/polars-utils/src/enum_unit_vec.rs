use either::Either;

/// List of items where a single item is stored on the stack.
/// Similar to UnitVec, but without size / alignment limitations.
#[derive(Debug, Clone)]
pub struct EnumUnitVec<T>(Either<[T; 1], Vec<T>>);

impl<T> EnumUnitVec<T> {
    pub const fn new() -> Self {
        Self(Either::Right(Vec::new()))
    }

    pub const fn new_single(value: T) -> Self {
        Self(Either::Left([value]))
    }
}

impl<T> Default for EnumUnitVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> std::ops::Deref for EnumUnitVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        AsRef::as_ref(&self.0)
    }
}

impl<T> std::ops::DerefMut for EnumUnitVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        AsMut::as_mut(&mut self.0)
    }
}

impl<T> From<Vec<T>> for EnumUnitVec<T> {
    fn from(value: Vec<T>) -> Self {
        Self(Either::Right(value))
    }
}

impl<T> IntoIterator for EnumUnitVec<T> {
    type IntoIter = Either<<[T; 1] as IntoIterator>::IntoIter, <Vec<T> as IntoIterator>::IntoIter>;
    type Item = T;

    fn into_iter(self) -> Self::IntoIter {
        match self.0 {
            Either::Left(v) => Either::Left(v.into_iter()),
            Either::Right(v) => Either::Right(v.into_iter()),
        }
    }
}

impl<T> FromIterator<T> for EnumUnitVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut iter = iter.into_iter();

        let Some(first) = iter.next() else {
            return Self::new();
        };

        let Some(second) = iter.next() else {
            return Self::new_single(first);
        };

        let mut vec = Vec::with_capacity(iter.size_hint().0 + 2);
        vec.push(first);
        vec.push(second);
        vec.extend(iter);
        Self::from(vec)
    }
}
