use std::hash::BuildHasher;
use std::sync::atomic::{AtomicU64, Ordering};

use foldhash::SharedSeed;
use foldhash::quality::SeedableRandomState as FHSeedableState;

/// A value of ``u64::MAX`` indicates using a random hash seed, the default
/// behavior. Other values indicate a specific seed has been set.
static HASH_SEED: AtomicU64 = AtomicU64::new(u64::MAX);

/// Set a hash seed instead of using a random one. Ideally should only be called
/// once, at startup.
pub fn set_hash_seed(mut seed: u64) {
    // This is a marker value so it can't be used.
    if seed == u64::MAX {
        seed = u64::MAX - 1;
    }
    // Relaxed ordering is why this should only be called once, ideally at
    // startup via e.g. an environment variable, before any additional threads
    // are started.
    HASH_SEED.store(seed, Ordering::Relaxed);
}

/// A RandomState that uses the seed set with ``set_hash_seed()``, if any,
/// otherwise using a random seed.
#[derive(Clone, Debug)]
pub struct PlSeedableRandomStateQuality(FHSeedableState);

impl PlSeedableRandomStateQuality {
    #[inline]
    fn with_seed(seed: u64, shared_seed: &'static SharedSeed) -> Self {
        Self(FHSeedableState::with_seed(seed, shared_seed))
    }

    #[inline]
    pub fn fixed() -> Self {
        Self(FHSeedableState::fixed())
    }
}

impl BuildHasher for PlSeedableRandomStateQuality {
    type Hasher = <FHSeedableState as BuildHasher>::Hasher;

    #[inline]
    fn build_hasher(&self) -> Self::Hasher {
        self.0.build_hasher()
    }
}

impl Default for PlSeedableRandomStateQuality {
    /// If a seed is set use it, otherwise use a random seed as usual.
    #[inline]
    fn default() -> Self {
        let seed = HASH_SEED.load(Ordering::Relaxed);
        if seed == u64::MAX {
            Self(Default::default())
        } else {
            Self::with_seed(seed, SharedSeed::global_fixed())
        }
    }
}

// Backwards compat
pub type PlRandomState = PlSeedableRandomStateQuality;

pub type PlRandomStateQuality = foldhash::quality::RandomState;
pub type PlFixedStateQuality = foldhash::quality::FixedState;

pub type PlHashMap<K, V> = hashbrown::HashMap<K, V, PlSeedableRandomStateQuality>;
pub type PlHashSet<V> = hashbrown::HashSet<V, PlSeedableRandomStateQuality>;
pub type PlIndexMap<K, V> = indexmap::IndexMap<K, V, PlSeedableRandomStateQuality>;
pub type PlIndexSet<K> = indexmap::IndexSet<K, PlSeedableRandomStateQuality>;

pub trait SeedableFromU64SeedExt {
    fn seed_from_u64(seed: u64) -> Self;
}

impl SeedableFromU64SeedExt for PlSeedableRandomStateQuality {
    fn seed_from_u64(seed: u64) -> Self {
        PlSeedableRandomStateQuality::with_seed(seed, SharedSeed::global_fixed())
    }
}

pub trait InitHashMaps {
    type HashMap;

    fn new() -> Self::HashMap;

    fn with_capacity(capacity: usize) -> Self::HashMap;
}

impl<K, V> InitHashMaps for PlHashMap<K, V> {
    type HashMap = Self;

    fn new() -> Self::HashMap {
        Self::with_capacity_and_hasher(0, Default::default())
    }

    fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, Default::default())
    }
}

impl<K> InitHashMaps for PlHashSet<K> {
    type HashMap = Self;

    fn new() -> Self::HashMap {
        Self::with_capacity_and_hasher(0, Default::default())
    }

    fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, Default::default())
    }
}

impl<K> InitHashMaps for PlIndexSet<K> {
    type HashMap = Self;

    fn new() -> Self::HashMap {
        Self::with_capacity_and_hasher(0, Default::default())
    }

    fn with_capacity(capacity: usize) -> Self::HashMap {
        Self::with_capacity_and_hasher(capacity, Default::default())
    }
}

impl<K, V> InitHashMaps for PlIndexMap<K, V> {
    type HashMap = Self;

    fn new() -> Self::HashMap {
        Self::with_capacity_and_hasher(0, Default::default())
    }

    fn with_capacity(capacity: usize) -> Self::HashMap {
        Self::with_capacity_and_hasher(capacity, Default::default())
    }
}
