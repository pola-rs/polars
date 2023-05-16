use ahash::RandomState;

pub type PlHashMap<K, V> = hashbrown::HashMap<K, V, RandomState>;
pub type PlHashSet<V> = hashbrown::HashSet<V, RandomState>;
