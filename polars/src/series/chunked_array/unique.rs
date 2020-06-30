use crate::prelude::*;
use arrow::datatypes::ArrowPrimitiveType;
use fnv::{FnvBuildHasher, FnvHashSet, FnvHasher};
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

        let mut builder = PrimitiveChunkedBuilder::new(self.name(), set.len());
        builder.new_from_iter(set.iter().copied())
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use crate::series::chunked_array::unique::Unique;

    #[test]
    fn unique() {
        let ca = ChunkedArray::<Int32Type>::new_from_slice("a", &[1, 2, 3, 2, 1]);
        println!("{:?}", ca.unique());
    }
}
