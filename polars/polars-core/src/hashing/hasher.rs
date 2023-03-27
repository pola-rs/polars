use polars_utils::HashSingle;
use xxhash_rust::xxh3::xxh3_64;

use super::*;

pub struct PlHasher {
    h: u64,
}

impl Hasher for PlHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.h
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let h = xxh3_64(bytes);
        self.h = h._fx_hash(self.h);
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.h = i._fx_hash(self.h)
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.h = i._fx_hash(self.h)
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.h = i._fx_hash(self.h)
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.h = i._fx_hash(self.h)
    }

    fn write_u128(&mut self, i: u128) {
        let bytes = i.to_ne_bytes();
        let lower = u64::from_ne_bytes(bytes[..8].try_into().unwrap());
        let upper = u64::from_ne_bytes(bytes[8..].try_into().unwrap());
        let k = lower._fx_hash(self.h);
        self.h = upper._fx_hash(k);
    }

    fn write_usize(&mut self, i: usize) {
        let v = u64::from_ne_bytes(i.to_ne_bytes());
        self.write_u64(v)
    }

    #[inline]
    fn write_i8(&mut self, i: i8) {
        self.h = i._fx_hash(self.h)
    }

    #[inline]
    fn write_i16(&mut self, i: i16) {
        self.h = i._fx_hash(self.h)
    }

    #[inline]
    fn write_i32(&mut self, i: i32) {
        self.h = i._fx_hash(self.h)
    }

    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.h = i._fx_hash(self.h)
    }

    fn write_i128(&mut self, i: i128) {
        let v = u128::from_ne_bytes(i.to_ne_bytes());
        self.write_u128(v)
    }
    fn write_isize(&mut self, i: isize) {
        let v = i64::from_ne_bytes(i.to_ne_bytes());
        self.write_i64(v)
    }
}

#[derive(Copy, Clone)]
pub struct PlHasherBuilder {
    seed: u64,
}

impl Default for PlHasherBuilder {
    fn default() -> Self {
        // TODO: use own logic for randomness
        let seed = ahash::RandomState::default().hash_single(FXHASH_K);
        Self { seed }
    }
}

impl PlHasherBuilder {
    pub fn with_seed(seed: u64) -> Self {
        Self { seed }
    }
}

impl BuildHasher for PlHasherBuilder {
    type Hasher = PlHasher;

    fn build_hasher(&self) -> Self::Hasher {
        PlHasher {
            h: FXHASH_K.wrapping_add(self.seed),
        }
    }
}
