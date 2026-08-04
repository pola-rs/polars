use std::hash::{DefaultHasher, Hash, Hasher};
use std::marker::PhantomData;

use hashbrown::HashTable;
use polars_utils::arena::Node;

/// Compare and hash nodes at the top level, assumed to be O(1)
/// These operations should not descend into child/input nodes
pub trait ShallowNodeOps {
    fn shallow_hash<H: Hasher>(&self, node: Node, state: &mut H);
    fn shallow_eq(&self, a: Node, b: Node) -> bool;
}

#[repr(transparent)]
pub struct DeduplicationId<T>(u32, PhantomData<T>);

impl<T> DeduplicationId<T> {
    fn new(id: u32) -> Self {
        Self(id, PhantomData)
    }

    pub(crate) fn as_u32(self) -> u32 {
        self.0
    }
}

// We can not derive these because of the marker type T

impl<T> Clone for DeduplicationId<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for DeduplicationId<T> {}
impl<T> PartialEq for DeduplicationId<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<T> Eq for DeduplicationId<T> {}
impl<T> Hash for DeduplicationId<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}
impl<T> std::fmt::Debug for DeduplicationId<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DeduplicationId({})", self.0)
    }
}

struct DeduplicationEntry<T> {
    representative: Node,
    child_ids: Vec<DeduplicationId<T>>,
    id: DeduplicationId<T>,
}

pub struct Interner<T> {
    deduplication_map: HashTable<DeduplicationEntry<T>>,
}

impl<T> Default for Interner<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Interner<T> {
    pub fn new() -> Self {
        Self {
            deduplication_map: HashTable::new(),
        }
    }

    pub fn get_or_assign(
        &mut self,
        node: Node,
        child_ids: Vec<DeduplicationId<T>>,
        ops: &impl ShallowNodeOps,
    ) -> DeduplicationId<T> {
        let hash = combined_hash(node, &child_ids, ops);
        let next_id = DeduplicationId::new(1 + self.deduplication_map.len() as u32);
        self.deduplication_map
            .entry(
                hash,
                |other| ops.shallow_eq(node, other.representative) && child_ids == other.child_ids,
                |other| combined_hash(other.representative, &other.child_ids, ops),
            )
            .or_insert(DeduplicationEntry {
                representative: node,
                child_ids,
                id: next_id,
            })
            .get()
            .id
    }
}

fn combined_hash<T>(
    node: Node,
    child_ids: &[DeduplicationId<T>],
    ops: &impl ShallowNodeOps,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    ops.shallow_hash(node, &mut hasher);
    for child_id in child_ids {
        hasher.write_u32(child_id.as_u32());
    }
    hasher.finish()
}
