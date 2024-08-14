use std::borrow::Borrow;
use std::fmt::{self, Debug};
use std::hash::{BuildHasherDefault, Hash, Hasher};

use hashbrown::hash_map::RawEntryMut;
use hashbrown::HashMap;
use polars_error::{polars_bail, polars_err, PolarsResult};
use polars_utils::aliases::PlRandomState;

use super::DictionaryKey;
use crate::array::indexable::{AsIndexed, Indexable};
use crate::array::{Array, MutableArray};
use crate::datatypes::ArrowDataType;

/// Hasher for pre-hashed values; similar to `hash_hasher` but with native endianness.
///
/// We know that we'll only use it for `u64` values, so we can avoid endian conversion.
///
/// Invariant: hash of a u64 value is always equal to itself.
#[derive(Copy, Clone, Default)]
pub struct PassthroughHasher(u64);

impl Hasher for PassthroughHasher {
    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }

    fn write(&mut self, _: &[u8]) {
        unreachable!();
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

#[derive(Clone)]
pub struct Hashed<K> {
    hash: u64,
    key: K,
}

impl<K> Hash for Hashed<K> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash.hash(state)
    }
}

#[derive(Clone)]
pub struct ValueMap<K: DictionaryKey, M: MutableArray> {
    pub values: M,
    pub map: HashMap<Hashed<K>, (), BuildHasherDefault<PassthroughHasher>>, // NB: *only* use insert_hashed_nocheck() and no other hashmap API
    random_state: PlRandomState,
}

impl<K: DictionaryKey, M: MutableArray> ValueMap<K, M> {
    pub fn try_empty(values: M) -> PolarsResult<Self> {
        if !values.is_empty() {
            polars_bail!(ComputeError: "initializing value map with non-empty values array")
        }
        Ok(Self {
            values,
            map: HashMap::default(),
            random_state: PlRandomState::default(),
        })
    }

    pub fn from_values(values: M) -> PolarsResult<Self>
    where
        M: Indexable,
        M::Type: Eq + Hash,
    {
        let mut map = HashMap::<Hashed<K>, _, _>::with_capacity_and_hasher(
            values.len(),
            BuildHasherDefault::<PassthroughHasher>::default(),
        );
        let random_state = PlRandomState::default();
        for index in 0..values.len() {
            let key = K::try_from(index).map_err(|_| polars_err!(ComputeError: "overflow"))?;
            // SAFETY: we only iterate within bounds
            let value = unsafe { values.value_unchecked_at(index) };
            let hash = random_state.hash_one(value.borrow());

            let entry = map.raw_entry_mut().from_hash(hash, |item| {
                // SAFETY: invariant of the struct, it's always in bounds since we maintain it
                let stored_value = unsafe { values.value_unchecked_at(item.key.as_usize()) };
                stored_value.borrow() == value.borrow()
            });
            match entry {
                RawEntryMut::Occupied(_) => {
                    polars_bail!(InvalidOperation: "duplicate value in dictionary values array")
                },
                RawEntryMut::Vacant(entry) => {
                    // NB: don't use .insert() here!
                    entry.insert_hashed_nocheck(hash, Hashed { hash, key }, ());
                },
            }
        }
        Ok(Self {
            values,
            map,
            random_state,
        })
    }

    pub fn data_type(&self) -> &ArrowDataType {
        self.values.data_type()
    }

    pub fn into_values(self) -> M {
        self.values
    }

    pub fn take_into(&mut self) -> Box<dyn Array> {
        let arr = self.values.as_box();
        self.map.clear();
        arr
    }

    #[inline]
    pub fn values(&self) -> &M {
        &self.values
    }

    /// Try to insert a value and return its index (it may or may not get inserted).
    pub fn try_push_valid<V>(
        &mut self,
        value: V,
        mut push: impl FnMut(&mut M, V) -> PolarsResult<()>,
    ) -> PolarsResult<K>
    where
        M: Indexable,
        V: AsIndexed<M>,
        M::Type: Eq + Hash,
    {
        let hash = self.random_state.hash_one(value.as_indexed());
        let entry = self.map.raw_entry_mut().from_hash(hash, |item| {
            // SAFETY: we've already checked (the inverse) when we pushed it, so it should be ok?
            let index = unsafe { item.key.as_usize() };
            // SAFETY: invariant of the struct, it's always in bounds since we maintain it
            let stored_value = unsafe { self.values.value_unchecked_at(index) };
            stored_value.borrow() == value.as_indexed()
        });
        let out = match entry {
            RawEntryMut::Occupied(entry) => entry.key().key,
            RawEntryMut::Vacant(entry) => {
                let index = self.values.len();
                let key = K::try_from(index).map_err(|_| polars_err!(ComputeError: "overflow"))?;
                entry.insert_hashed_nocheck(hash, Hashed { hash, key }, ()); // NB: don't use .insert() here!
                push(&mut self.values, value)?;
                debug_assert_eq!(self.values.len(), index + 1);
                key
            },
        };
        Ok(out)
    }

    pub fn shrink_to_fit(&mut self) {
        self.values.shrink_to_fit();
    }
}

impl<K: DictionaryKey, M: MutableArray> Debug for ValueMap<K, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.values.fmt(f)
    }
}
