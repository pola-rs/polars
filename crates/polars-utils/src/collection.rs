use std::marker::PhantomData;
use std::ops::{Deref, DerefMut, Index, IndexMut};

pub trait Collection<T: ?Sized> {
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize;
    fn get(&self, idx: usize) -> Option<&T>;
    fn get_mut(&mut self, idx: usize) -> Option<&mut T>;
}

/// Wrapper that implements indexing.
pub struct CollectionWrap<T, C: Collection<T>> {
    inner: C,
    phantom: PhantomData<T>,
}

impl<T, C: Collection<T>> CollectionWrap<T, C> {
    pub fn new(inner: C) -> Self {
        Self {
            inner,
            phantom: PhantomData,
        }
    }

    pub fn iter(&self) -> CollectionIter<'_, T, C> {
        CollectionIter {
            idx: 0,
            collection: &self.inner,
            phantom: PhantomData,
        }
    }

    pub fn for_each_mut<F>(&mut self, mut f: F)
    where
        F: for<'b> FnMut(&'b mut T),
    {
        (0..self.len()).for_each(move |i| f(self.get_mut(i).unwrap()))
    }

    pub fn map_mut<'a, B, F>(&'a mut self, mut f: F) -> impl Iterator<Item = B>
    where
        F: for<'b> FnMut(&'b mut T) -> B + 'a,
    {
        (0..self.len()).map(move |i| f(self.get_mut(i).unwrap()))
    }

    pub fn into_inner(self) -> C {
        self.inner
    }
}

impl<T, C: Collection<T>> Deref for CollectionWrap<T, C> {
    type Target = C;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T, C: Collection<T>> DerefMut for CollectionWrap<T, C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<T, C: Collection<T>> Index<usize> for CollectionWrap<T, C> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl<T, C: Collection<T>> IndexMut<usize> for CollectionWrap<T, C> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).unwrap()
    }
}

impl<T: Clone, C: Collection<T>, const N: usize> TryFrom<CollectionWrap<T, C>> for [T; N] {
    type Error = ();

    fn try_from(value: CollectionWrap<T, C>) -> Result<Self, Self::Error> {
        if value.len() != N {
            return Err(());
        }

        Ok(std::array::from_fn(|i| value.get(i).unwrap().clone()))
    }
}

impl<T, C: Collection<T>> From<C> for CollectionWrap<T, C> {
    fn from(value: C) -> Self {
        Self {
            inner: value,
            phantom: PhantomData,
        }
    }
}

impl<T> Collection<T> for &mut dyn Collection<T> {
    fn len(&self) -> usize {
        (**self).len()
    }

    fn get(&self, idx: usize) -> Option<&T> {
        (**self).get(idx)
    }

    fn get_mut(&mut self, idx: usize) -> Option<&mut T> {
        (**self).get_mut(idx)
    }
}

pub struct CollectionIter<'a, T: 'a, C: Collection<T>> {
    idx: usize,
    collection: &'a C,
    phantom: PhantomData<T>,
}

impl<'a, T, C: Collection<T>> Iterator for CollectionIter<'a, T, C> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.collection.get(self.idx);

        if item.is_some() {
            self.idx += 1;
        }

        item
    }
}

impl<T> Collection<T> for [T] {
    fn len(&self) -> usize {
        <[T]>::len(self)
    }

    fn get(&self, idx: usize) -> Option<&T> {
        <[T]>::get(self, idx)
    }

    fn get_mut(&mut self, idx: usize) -> Option<&mut T> {
        <[T]>::get_mut(self, idx)
    }
}

impl<T> Collection<T> for &mut [T] {
    fn len(&self) -> usize {
        <[T]>::len(self)
    }

    fn get(&self, idx: usize) -> Option<&T> {
        <[T]>::get(self, idx)
    }

    fn get_mut(&mut self, idx: usize) -> Option<&mut T> {
        <[T]>::get_mut(self, idx)
    }
}
