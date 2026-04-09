use std::marker::PhantomData;
use std::ops::{Deref, DerefMut, Index, IndexMut};

pub trait Collection<T> {
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize;
    fn get(&self, idx: usize) -> Option<&T>;
    fn get_mut(&mut self, idx: usize) -> Option<&mut T>;

    /// We can't implement `iter_mut()` so expose this as an alternative.
    fn map_mut<'a, F, O>(&'a mut self, mut f: F) -> impl Iterator<Item = O> + 'a
    where
        F: for<'b> FnMut(&'b mut T) -> O + 'a,
    {
        (0..self.len()).map(move |i| f(self.get_mut(i).unwrap()))
    }
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
        CollectionIter { idx: 0, wrap: self }
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

pub struct CollectionIter<'a, T, C: Collection<T>> {
    idx: usize,
    wrap: &'a CollectionWrap<T, C>,
}

impl<'a, T, C: Collection<T>> Iterator for CollectionIter<'a, T, C> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx == self.wrap.len() {
            return None;
        }

        self.idx += 1;

        self.wrap.get(self.idx - 1)
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

pub struct MappedCollection<'src, Src: ?Sized, MapRef, MapRefMut, T, U> {
    src: &'src mut Src,
    map: MapRef,
    map_mut: MapRefMut,
    phantom: PhantomData<(T, U)>,
}

impl<'src, Src: ?Sized, MapRef, MapRefMut, T, U> Collection<U>
    for MappedCollection<'src, Src, MapRef, MapRefMut, T, U>
where
    Src: Collection<T>,
    MapRef: for<'a> Fn(&'a T) -> &'a U,
    MapRefMut: for<'a> Fn(&'a mut T) -> &'a mut U,
{
    fn len(&self) -> usize {
        self.src.len()
    }

    fn get(&self, idx: usize) -> Option<&U> {
        self.src.get(idx).map(|t| (self.map)(t))
    }

    fn get_mut(&mut self, idx: usize) -> Option<&mut U> {
        self.src.get_mut(idx).map(|t| (self.map_mut)(t))
    }
}

impl<'src, Src: ?Sized, MapRef, MapRefMut, T, U>
    MappedCollection<'src, Src, MapRef, MapRefMut, T, U>
where
    Src: Collection<T>,
    MapRef: for<'a> Fn(&'a T) -> &'a U,
    MapRefMut: for<'a> Fn(&'a mut T) -> &'a mut U,
{
    pub fn new(src: &'src mut Src, map: MapRef, map_mut: MapRefMut) -> Self {
        Self {
            src,
            map,
            map_mut,
            phantom: PhantomData,
        }
    }
}
