use crate::prelude::*;
use fnv::{FnvBuildHasher, FnvHasher};
use std::collections::HashSet;
use std::hash::{BuildHasherDefault, Hash};

pub trait Unique {
    fn unique(&self) -> Self;
}

fn fill_set<A>(
    a: impl Iterator<Item = A>,
    capacity: usize,
) -> HashSet<A, BuildHasherDefault<FnvHasher>>
where
    A: Hash + Eq,
{
    let mut set = HashSet::with_capacity_and_hasher(capacity, FnvBuildHasher::default());

    for val in a {
        set.insert(val);
    }

    set
}

impl<T> Unique for ChunkedArray<T>
where
    T: PolarNumericType,
    T::Native: Hash + Eq,
    ChunkedArray<T>: ChunkOps,
{
    fn unique(&self) -> Self {
        let set = match self.cont_slice() {
            Ok(slice) => fill_set(slice.iter().map(|v| Some(*v)), self.len()),
            Err(_) => fill_set(self.into_iter(), self.len()),
        };

        let builder = PrimitiveChunkedBuilder::new(self.name(), set.len());
        builder.new_from_iter(set.iter().copied())
    }
}

impl Unique for Utf8Chunked {
    fn unique(&self) -> Self {
        let set = fill_set(self.into_iter(), self.len());
        let mut builder = Utf8ChunkedBuilder::new(self.name(), set.len());
        self.into_iter()
            .for_each(|val| builder.append_value(val).expect("could not append"));
        builder.finish()
    }
}

impl Unique for BooleanChunked {
    fn unique(&self) -> Self {
        // can be None, Some(true), Some(false)
        let mut unique = Vec::with_capacity(3);
        for v in self {
            if unique.len() == 3 {
                break;
            }
            if !unique.contains(&v) {
                unique.push(v)
            }
        }
        ChunkedArray::new_from_opt_slice(self.name(), &unique)
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use itertools::Itertools;

    #[test]
    fn unique() {
        let ca = ChunkedArray::<Int32Type>::new_from_slice("a", &[1, 2, 3, 2, 1]);
        assert_eq!(
            ca.unique().into_iter().collect_vec(),
            vec![Some(1), Some(2), Some(3)]
        );
        let ca = BooleanChunked::new_from_slice("a", &[true, false, true]);
        assert_eq!(
            ca.unique().into_iter().collect_vec(),
            vec![Some(true), Some(false)]
        );
    }
}
