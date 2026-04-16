use foldhash::SharedSeed;

pub type PlRandomState = foldhash::quality::RandomState;
pub type PlSeedableRandomStateQuality = foldhash::quality::SeedableRandomState;
pub type PlRandomStateQuality = foldhash::quality::RandomState;
pub type PlFixedStateQuality = foldhash::quality::FixedState;

pub type PlHashMap<K, V> = hashbrown::HashMap<K, V, PlRandomState>;
pub type PlHashSet<V> = hashbrown::HashSet<V, PlRandomState>;
pub type PlIndexMap<K, V> = indexmap::IndexMap<K, V, PlRandomState>;
pub type PlIndexSet<K> = indexmap::IndexSet<K, PlRandomState>;

/// HashMap container with a getter that clears the HashMap.
#[derive(Default)]
pub struct ScratchHashMap<K, V>(PlHashMap<K, V>);

impl<K, V> ScratchHashMap<K, V> {
    /// Clear the HashMap and return a mutable reference to it.
    pub fn get(&mut self) -> &mut PlHashMap<K, V> {
        self.0.clear();
        &mut self.0
    }
}

/// HashSet container with a getter that clears the HashSet.
#[derive(Default)]
pub struct ScratchHashSet<K>(PlHashSet<K>);

impl<K> ScratchHashSet<K> {
    /// Clear the HashSet and return a mutable reference to it.
    pub fn get(&mut self) -> &mut PlHashSet<K> {
        self.0.clear();
        &mut self.0
    }
}

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
