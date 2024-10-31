use std::borrow::Borrow;
use std::fmt::{self, Debug};
use std::hash::Hash;

use hashbrown::hash_table::Entry;
use hashbrown::HashTable;
use polars_error::{polars_bail, polars_err, PolarsResult};
use polars_utils::aliases::PlRandomState;

use super::DictionaryKey;
use crate::array::indexable::{AsIndexed, Indexable};
use crate::array::{Array, MutableArray};
use crate::datatypes::ArrowDataType;

#[derive(Clone)]
pub struct ValueMap<K: DictionaryKey, M: MutableArray> {
    pub values: M,
    pub map: HashTable<(u64, K)>,
    random_state: PlRandomState,
}

impl<K: DictionaryKey, M: MutableArray> ValueMap<K, M> {
    pub fn try_empty(values: M) -> PolarsResult<Self> {
        if !values.is_empty() {
            polars_bail!(ComputeError: "initializing value map with non-empty values array")
        }
        Ok(Self {
            values,
            map: HashTable::default(),
            random_state: PlRandomState::default(),
        })
    }

    pub fn from_values(values: M) -> PolarsResult<Self>
    where
        M: Indexable,
        M::Type: Eq + Hash,
    {
        let mut map: HashTable<(u64, K)> = HashTable::with_capacity(values.len());
        let random_state = PlRandomState::default();
        for index in 0..values.len() {
            let key = K::try_from(index).map_err(|_| polars_err!(ComputeError: "overflow"))?;
            // SAFETY: we only iterate within bounds
            let value = unsafe { values.value_unchecked_at(index) };
            let hash = random_state.hash_one(value.borrow());

            let entry = map.entry(
                hash,
                |(_h, key)| {
                    // SAFETY: invariant of the struct, it's always in bounds.
                    let stored_value = unsafe { values.value_unchecked_at(key.as_usize()) };
                    stored_value.borrow() == value.borrow()
                },
                |(h, _key)| *h,
            );
            match entry {
                Entry::Occupied(_) => {
                    polars_bail!(InvalidOperation: "duplicate value in dictionary values array")
                },
                Entry::Vacant(entry) => {
                    entry.insert((hash, key));
                },
            }
        }
        Ok(Self {
            values,
            map,
            random_state,
        })
    }

    pub fn dtype(&self) -> &ArrowDataType {
        self.values.dtype()
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
        let entry = self.map.entry(
            hash,
            |(_h, key)| {
                // SAFETY: invariant of the struct, it's always in bounds.
                let stored_value = unsafe { self.values.value_unchecked_at(key.as_usize()) };
                stored_value.borrow() == value.as_indexed()
            },
            |(h, _key)| *h,
        );
        let out = match entry {
            Entry::Occupied(entry) => entry.get().1,
            Entry::Vacant(entry) => {
                let index = self.values.len();
                let key = K::try_from(index).map_err(|_| polars_err!(ComputeError: "overflow"))?;
                entry.insert((hash, key));
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
