use std::hash::Hash;

use polars_arrow::bitmap::{Bitmap, BitmapBuilder, MutableBitmap};
use polars_core::prelude::*;
use polars_utils::float16::pf16;
use polars_utils::total_ord::{canonical_f16, canonical_f32, canonical_f64};

use super::{SMALL_MAX, finish_chunk, probe_words};

/// Integers use a bitset when their value range fits in this many bits,
const BITSET_MIN_BITS: u64 = 1 << 16;
/// or in this many bits per haystack value, whichever limit is larger.
const BITSET_BITS_PER_VALUE: u64 = 128;

/// Native value of a numeric column mapped to a key that hashes and compares by total equality.
pub(super) trait IsInNative: Copy + Send + Sync + 'static {
    type Key: Copy + Eq + Hash + Send + Sync + 'static;

    fn to_key(self) -> Self::Key;

    /// Smallest key and the number of distinct values from that key up to the largest one.
    /// Only integers support this.
    fn key_range(keys: &[Self::Key]) -> Option<(Self::Key, u64)>;

    /// Distance of `key` to `min`. Keys below `min` map to distances beyond any bitset range.
    fn key_offset(key: Self::Key, min: Self::Key) -> u64;
}

macro_rules! impl_is_in_native_int {
    ($($t:ty => $wide:ty),*) => {$(
        impl IsInNative for $t {
            type Key = $t;

            #[inline(always)]
            fn to_key(self) -> $t {
                self
            }

            fn key_range(keys: &[$t]) -> Option<($t, u64)> {
                let (min, max) = keys
                    .iter()
                    .fold(None, |acc: Option<($t, $t)>, &k| match acc {
                        None => Some((k, k)),
                        Some((lo, hi)) => Some((lo.min(k), hi.max(k))),
                    })?;
                let range = (max as $wide).wrapping_sub(min as $wide).checked_add(1)?;
                Some((min, u64::try_from(range).ok()?))
            }

            #[inline(always)]
            fn key_offset(key: $t, min: $t) -> u64 {
                (key as $wide).wrapping_sub(min as $wide).min(u64::MAX as $wide) as u64
            }
        }
    )*};
}

impl_is_in_native_int!(
    u8 => u64, u16 => u64, u32 => u64, u64 => u64, u128 => u128,
    i8 => u64, i16 => u64, i32 => u64, i64 => u64, i128 => u128
);

macro_rules! impl_is_in_native_float {
    ($($t:ty => $key:ty, $canonical:ident, $bits:expr),*) => {$(
        impl IsInNative for $t {
            type Key = $key;

            #[inline(always)]
            fn to_key(self) -> $key {
                $bits($canonical(self))
            }

            fn key_range(_keys: &[$key]) -> Option<($key, u64)> {
                None
            }

            fn key_offset(_key: $key, _min: $key) -> u64 {
                unreachable!()
            }
        }
    )*};
}

impl_is_in_native_float!(
    pf16 => u16, canonical_f16, |x: pf16| x.0.to_bits(),
    f32 => u32, canonical_f32, f32::to_bits,
    f64 => u64, canonical_f64, f64::to_bits
);

/// The fixed size lets the compiler unroll and vectorize the scan.
fn probe_small<N: IsInNative, const K: usize>(
    keys: &[N::Key],
    values: &[N],
    out: &mut BitmapBuilder,
) {
    let keys: [N::Key; K] = keys.try_into().unwrap();
    probe_words(values, out, |v| {
        let key = v.to_key();
        let mut found = false;
        for h in keys {
            found |= h == key;
        }
        found
    })
}

pub(super) enum PrimitiveLookup<T: PolarsNumericType>
where
    T::Native: IsInNative,
{
    Small(Vec<<T::Native as IsInNative>::Key>),
    Bitset {
        min: <T::Native as IsInNative>::Key,
        bits: Bitmap,
    },
    Hash(PlHashSet<<T::Native as IsInNative>::Key>),
}

impl<T: PolarsNumericType> PrimitiveLookup<T>
where
    T::Native: IsInNative,
{
    pub(super) fn new(ca: &ChunkedArray<T>) -> Self {
        let keys: Vec<_> = ca
            .downcast_iter()
            .flat_map(|arr| arr.non_null_values_iter())
            .map(IsInNative::to_key)
            .collect();

        if keys.len() <= SMALL_MAX {
            return Self::Small(keys);
        }

        if let Some((min, range)) = T::Native::key_range(&keys)
            && range <= BITSET_MIN_BITS.max(keys.len() as u64 * BITSET_BITS_PER_VALUE)
            && let Ok(range) = usize::try_from(range)
        {
            let mut bits = MutableBitmap::from_len_zeroed(range);
            for key in keys {
                bits.set(T::Native::key_offset(key, min) as usize, true);
            }
            return Self::Bitset {
                min,
                bits: bits.freeze(),
            };
        }

        Self::Hash(keys.into_iter().collect())
    }

    fn probe_values(&self, values: &[T::Native], out: &mut BitmapBuilder) {
        match self {
            Self::Small(keys) => match keys.len() {
                0 => out.extend_constant(values.len(), false),
                1 => probe_small::<T::Native, 1>(keys, values, out),
                2 => probe_small::<T::Native, 2>(keys, values, out),
                3 => probe_small::<T::Native, 3>(keys, values, out),
                4 => probe_small::<T::Native, 4>(keys, values, out),
                5 => probe_small::<T::Native, 5>(keys, values, out),
                6 => probe_small::<T::Native, 6>(keys, values, out),
                7 => probe_small::<T::Native, 7>(keys, values, out),
                8 => probe_small::<T::Native, 8>(keys, values, out),
                _ => unreachable!(),
            },
            Self::Bitset { min, bits } => probe_words(values, out, |v| {
                let off = T::Native::key_offset(v.to_key(), *min);
                let in_range = off < bits.len() as u64;
                let off = if in_range { off as usize } else { 0 };
                // SAFETY: off < bits.len().
                in_range & unsafe { bits.get_bit_unchecked(off) }
            }),
            Self::Hash(set) => probe_words(values, out, |v| set.contains(&v.to_key())),
        }
    }

    pub(super) fn probe(
        &self,
        ca: &ChunkedArray<T>,
        nulls_equal: bool,
        has_null: bool,
    ) -> BooleanChunked {
        let chunks = ca.downcast_iter().map(|arr| {
            let mut out = BitmapBuilder::with_capacity(arr.len());
            self.probe_values(arr.values().as_slice(), &mut out);
            finish_chunk(out.freeze(), arr.validity(), nulls_equal, has_null)
        });
        BooleanChunked::from_chunk_iter(ca.name().clone(), chunks)
    }
}

/// Type-erased [`PrimitiveLookup`].
pub(super) trait PrimitiveProbe: Send + Sync {
    fn probe_series(&self, needle: &Series, nulls_equal: bool, has_null: bool) -> BooleanChunked;
}

impl<T: PolarsNumericType> PrimitiveProbe for PrimitiveLookup<T>
where
    T::Native: IsInNative,
{
    fn probe_series(&self, needle: &Series, nulls_equal: bool, has_null: bool) -> BooleanChunked {
        self.probe(needle.as_ref().as_ref(), nulls_equal, has_null)
    }
}
