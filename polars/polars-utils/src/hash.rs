use std::hash::{BuildHasher, Hash};

pub trait HashSingle: BuildHasher {
    #[inline]
    fn hash_single<T: Hash>(&self, x: T) -> u64
    where
        Self: Sized,
    {
        #[cfg(feature = "nightly")]
        {
            self.hash_one(x)
        }
        #[cfg(not(feature = "nightly"))]
        {
            use std::hash::Hasher;
            let mut hasher = self.build_hasher();
            x.hash(&mut hasher);
            hasher.finish()
        }
    }
}

impl<T: BuildHasher> HashSingle for T {}
