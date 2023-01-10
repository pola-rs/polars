#[cfg(feature = "unique_counts")]
use std::hash::Hash;

#[cfg(feature = "unique_counts")]
use crate::frame::groupby::hashing::HASHMAP_INIT_SIZE;
use crate::prelude::*;
#[cfg(feature = "unique_counts")]
use crate::utils::NoNull;

#[cfg(feature = "unique_counts")]
fn unique_counts<I, J>(items: I) -> IdxCa
where
    I: Iterator<Item = J>,
    J: Hash + Eq,
{
    let mut map = PlIndexMap::with_capacity_and_hasher(HASHMAP_INIT_SIZE, Default::default());
    for item in items {
        map.entry(item)
            .and_modify(|cnt| {
                *cnt += 1;
            })
            .or_insert(1 as IdxSize);
    }
    let out: NoNull<IdxCa> = map.into_values().collect();
    out.into_inner()
}

impl Series {
    /// Returns a count of the unique values in the order of appearance.
    #[cfg(feature = "unique_counts")]
    pub fn unique_counts(&self) -> IdxCa {
        if self.dtype().to_physical().is_numeric() {
            if self.bit_repr_is_large() {
                let ca = self.bit_repr_large();
                unique_counts(ca.into_iter())
            } else {
                let ca = self.bit_repr_small();
                unique_counts(ca.into_iter())
            }
        } else {
            match self.dtype() {
                DataType::Utf8 => unique_counts(self.utf8().unwrap().into_iter()),
                dt => {
                    panic!("'unique_counts' not implemented for {dt} data types")
                }
            }
        }
    }
}
