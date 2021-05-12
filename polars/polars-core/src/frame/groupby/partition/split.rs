use crate::frame::groupby::{groupby_threaded, GroupedMap};
use crate::prelude::*;
use crate::utils::split_ca;
use crate::vector_hasher::create_hash_and_keys_threaded_vectorized;
use crate::POOL;
use ahash::RandomState;
use hashbrown::hash_map::RawEntryMut;
use hashbrown::HashMap;
use rayon::prelude::*;
use std::hash::Hash;

pub trait ToBitRepr {
    fn to_bit_repr(self) -> u64;
}

impl ToBitRepr for f64 {
    fn to_bit_repr(self) -> u64 {
        self.to_bits()
    }
}

impl ToBitRepr for f32 {
    fn to_bit_repr(self) -> u64 {
        self.to_bits() as u64
    }
}

impl ToBitRepr for u32 {
    fn to_bit_repr(self) -> u64 {
        self as u64
    }
}

impl ToBitRepr for u64 {
    fn to_bit_repr(self) -> u64 {
        self
    }
}

impl ToBitRepr for i32 {
    fn to_bit_repr(self) -> u64 {
        let a: u32 = unsafe { std::mem::transmute(self) };
        a.to_bit_repr()
    }
}

impl ToBitRepr for i64 {
    fn to_bit_repr(self) -> u64 {
        unsafe { std::mem::transmute(self) }
    }
}

pub trait IntoGroupMap<T> {
    fn group_maps(&self) -> Vec<GroupedMap<Option<u64>>>;
}

impl<T> IntoGroupMap<T::Native> for ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: Eq + Hash + Send + Copy + ToBitRepr,
{
    fn group_maps(&self) -> Vec<GroupedMap<Option<u64>>> {
        let group_size_hint = if let Some(m) = &self.categorical_map {
            self.len() / m.len()
        } else {
            0
        };
        {
            let n_threads = num_cpus::get();
            let splitted = split_ca(self, n_threads).unwrap();

            if self.chunks.len() == 1 {
                let iters = splitted
                    .iter()
                    .map(|ca| {
                        ca.downcast_iter()
                            .map(|arr| arr.into_iter().map(|opt_v| opt_v.map(|v| v.to_bit_repr())))
                    })
                    .flatten()
                    .collect();
                groupby_threaded(iters, group_size_hint)
            } else {
                let iters = splitted
                    .iter()
                    .map(|ca| ca.into_iter().map(|opt_v| opt_v.map(|v| v.to_bit_repr())))
                    .collect();
                groupby_threaded(iters, group_size_hint)
            }
        }
    }
}


