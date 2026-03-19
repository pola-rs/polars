use super::*;

#[derive(Default)]
pub struct IdHasher {
    hash: u64,
}

impl Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.hash
    }

    fn write(&mut self, _bytes: &[u8]) {
        unreachable!("IdHasher should only be used for integer keys <= 64 bit precision")
    }

    fn write_u32(&mut self, i: u32) {
        self.hash = i as u64;
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.hash = i;
    }

    fn write_i32(&mut self, i: i32) {
        self.hash = i as u64;
    }

    fn write_i64(&mut self, i: i64) {
        self.hash = i as u64;
    }
}

pub type IdBuildHasher = BuildHasherDefault<IdHasher>;
