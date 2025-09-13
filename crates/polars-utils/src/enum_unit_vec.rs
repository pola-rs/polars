use either::Either;

/// List of items where a single item is stored on the stack.
/// Similar to UnitVec, but without size / alignment limitations.
#[derive(Debug, Clone, PartialEq, Hash)]
pub struct EUnitVec<T>(Either<[T; 1], Vec<T>>);

impl<T> EUnitVec<T> {
    pub const fn new() -> Self {
        Self(Either::Right(Vec::new()))
    }

    pub const fn new_single(value: T) -> Self {
        Self(Either::Left([value]))
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        match &self.0 {
            Either::Left(_) => 1,
            Either::Right(vec) => vec.capacity().max(1),
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        match &mut self.0 {
            Either::Left(_) => *self = Self::new(),
            Either::Right(vec) => vec.clear(),
        }
    }

    #[inline]
    pub fn push(&mut self, value: T) {
        match &mut self.0 {
            Either::Left(_) => {
                let first = self.single_to_vec(2);

                let vec = self.as_mut_vec().unwrap();
                vec.push(first);
                vec.push(value);
            },
            Either::Right(vec) => {
                if vec.capacity() == 0 {
                    self.0 = Either::Left([value])
                } else {
                    vec.push(value)
                }
            },
        }
    }

    pub fn reserve(&mut self, additional: usize) {
        match &mut self.0 {
            Either::Left(_) => {
                let new_len = self.len().checked_add(additional).unwrap();
                if new_len > 1 {
                    let first = self.single_to_vec(new_len);
                    self.as_mut_vec().unwrap().push(first);
                }
            },
            Either::Right(vec) => vec.reserve(additional),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        if capacity <= 1 {
            Self::new()
        } else {
            Self(Either::Right(Vec::with_capacity(capacity)))
        }
    }

    /// Replaces an inner repr of `Left([T])` with `Right(Vec::with_capacity(capacity))`, and returns
    /// `T`.
    ///
    /// # Panics
    /// Panics if `self` does not contain `Left([T])`.
    #[inline]
    fn single_to_vec(&mut self, capacity: usize) -> T {
        let Either::Left([first]) =
            std::mem::replace(&mut self.0, Either::Right(Vec::with_capacity(capacity)))
        else {
            panic!()
        };

        first
    }

    #[inline]
    fn as_mut_vec(&mut self) -> Option<&mut Vec<T>> {
        if let Either::Right(vec) = &mut self.0 {
            Some(vec)
        } else {
            None
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.as_ref()
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.as_mut()
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        match &mut self.0 {
            Either::Left(_) => Some(self.single_to_vec(0)),
            Either::Right(vec) => vec.pop(),
        }
    }
}

impl<T> Default for EUnitVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> std::ops::Deref for EUnitVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        AsRef::as_ref(&self.0)
    }
}

impl<T> std::ops::DerefMut for EUnitVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        AsMut::as_mut(&mut self.0)
    }
}

impl<T> From<Vec<T>> for EUnitVec<T> {
    fn from(value: Vec<T>) -> Self {
        Self(Either::Right(value))
    }
}

impl<T, const N: usize> From<[T; N]> for EUnitVec<T> {
    fn from(value: [T; N]) -> Self {
        value.into_iter().collect()
    }
}

impl<T> IntoIterator for EUnitVec<T> {
    type IntoIter = Either<<[T; 1] as IntoIterator>::IntoIter, <Vec<T> as IntoIterator>::IntoIter>;
    type Item = T;

    fn into_iter(self) -> Self::IntoIter {
        match self.0 {
            Either::Left(v) => Either::Left(v.into_iter()),
            Either::Right(v) => Either::Right(v.into_iter()),
        }
    }
}

impl<T> FromIterator<T> for EUnitVec<T> {
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

impl<T> Extend<T> for EUnitVec<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let mut iter = iter.into_iter();
        self.reserve(iter.size_hint().0);

        loop {
            if self.as_mut_vec().is_some() {
                break;
            }

            let Some(item) = iter.next() else {
                return;
            };

            self.push(item);
        }

        self.as_mut_vec().unwrap().extend(iter)
    }
}

#[macro_export]
macro_rules! eunitvec {
    () => {{
        $crate::enum_unit_vec::EUnitVec::new()
    }};
    ($elem:expr; $n:expr) => {{
        let mut new = $crate::enum_unit_vec::EUnitVec::new();
        for _ in 0..$n {
            new.push($elem)
        }
        new
    }};
    ($elem:expr) => {{
        $crate::enum_unit_vec::EUnitVec::new_single($elem)
    }};
    ($($x:expr),+ $(,)?) => {{
        vec![$($x),+].into()
    }};
}

mod tests {

    #[test]
    fn test_enum_unitvec_clone() {
        {
            let v = eunitvec![1usize];
            assert_eq!(v, v.clone());
        }

        for n in [
            26903816120209729usize,
            42566276440897687,
            44435161834424652,
            49390731489933083,
            51201454727649242,
            83861672190814841,
            92169290527847622,
            92476373900398436,
            95488551309275459,
            97499984126814549,
        ] {
            let v = eunitvec![n];
            assert_eq!(v, v.clone());
        }
    }

    #[test]
    fn test_enum_unitvec_repeat_n() {
        assert_eq!(eunitvec![5; 3].as_slice(), &[5, 5, 5])
    }
}
