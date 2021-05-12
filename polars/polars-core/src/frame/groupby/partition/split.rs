use crate::frame::groupby::{groupby_threaded, GroupedMap};
use crate::prelude::*;
use crate::utils::split_ca;
use std::hash::Hash;

pub trait ToBitRepr {
    fn into_bit_repr(self) -> u64;
}

impl ToBitRepr for f64 {
    #[inline]
    fn into_bit_repr(self) -> u64 {
        self.to_bits()
    }
}

impl ToBitRepr for f32 {
    #[inline]
    fn into_bit_repr(self) -> u64 {
        self.to_bits() as u64
    }
}

impl ToBitRepr for u32 {
    #[inline]
    fn into_bit_repr(self) -> u64 {
        self as u64
    }
}

impl ToBitRepr for u64 {
    #[inline]
    fn into_bit_repr(self) -> u64 {
        self
    }
}

impl ToBitRepr for i32 {
    #[inline]
    fn into_bit_repr(self) -> u64 {
        let a: u32 = unsafe { std::mem::transmute(self) };
        a.into_bit_repr()
    }
}

impl ToBitRepr for i64 {
    #[inline]
    fn into_bit_repr(self) -> u64 {
        unsafe { std::mem::transmute(self) }
    }
}

impl ToBitRepr for i16 {
    #[inline]
    fn into_bit_repr(self) -> u64 {
        let a: u16 = unsafe { std::mem::transmute(self) };
        a.into_bit_repr()
    }
}

impl ToBitRepr for u16 {
    #[inline]
    fn into_bit_repr(self) -> u64 {
        self as u64
    }
}

impl ToBitRepr for i8 {
    #[inline]
    fn into_bit_repr(self) -> u64 {
        let a: u8 = unsafe { std::mem::transmute(self) };
        a.into_bit_repr()
    }
}

impl ToBitRepr for u8 {
    #[inline]
    fn into_bit_repr(self) -> u64 {
        self as u64
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
                        ca.downcast_iter().map(|arr| {
                            arr.into_iter()
                                .map(|opt_v| opt_v.map(|v| v.into_bit_repr()))
                        })
                    })
                    .flatten()
                    .collect();
                groupby_threaded(iters, group_size_hint)
            } else {
                let iters = splitted
                    .iter()
                    .map(|ca| ca.into_iter().map(|opt_v| opt_v.map(|v| v.into_bit_repr())))
                    .collect();
                groupby_threaded(iters, group_size_hint)
            }
        }
    }
}
