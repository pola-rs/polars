use std::fmt::{Debug, Formatter};
use std::hash::{BuildHasher, Hash, Hasher};

use arrow::array::*;
use polars_utils::aliases::PlRandomState;
#[cfg(any(feature = "serde-lazy", feature = "serde"))]
use serde::{Deserialize, Serialize};

use crate::datatypes::PlHashMap;
use crate::using_string_cache;

#[derive(Debug, Copy, Clone, PartialEq, Default)]
#[cfg_attr(
    any(feature = "serde-lazy", feature = "serde"),
    derive(Serialize, Deserialize)
)]
pub enum CategoricalOrdering {
    #[default]
    Physical,
    Lexical,
}

#[derive(Clone)]
pub enum RevMapping {
    /// Hashmap: maps the indexes from the global cache/categorical array to indexes in the local Utf8Array
    /// Utf8Array: caches the string values
    Global(PlHashMap<u32, u32>, Utf8ViewArray, u32),
    /// Utf8Array: caches the string values and a hash of all values for quick comparison
    Local(Utf8ViewArray, u128),
}

impl Debug for RevMapping {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            RevMapping::Global(_, _, _) => {
                write!(f, "global")
            },
            RevMapping::Local(_, _) => {
                write!(f, "local")
            },
        }
    }
}

impl Default for RevMapping {
    fn default() -> Self {
        let slice: &[Option<&str>] = &[];
        let cats = Utf8ViewArray::from_slice(slice);
        if using_string_cache() {
            let cache = &mut crate::STRING_CACHE.lock_map();
            let id = cache.uuid;
            RevMapping::Global(Default::default(), cats, id)
        } else {
            RevMapping::build_local(cats)
        }
    }
}

#[allow(clippy::len_without_is_empty)]
impl RevMapping {
    pub fn is_global(&self) -> bool {
        matches!(self, Self::Global(_, _, _))
    }

    pub fn is_local(&self) -> bool {
        matches!(self, Self::Local(_, _))
    }

    /// Get the categories in this [`RevMapping`]
    pub fn get_categories(&self) -> &Utf8ViewArray {
        match self {
            Self::Global(_, a, _) => a,
            Self::Local(a, _) => a,
        }
    }

    fn build_hash(categories: &Utf8ViewArray) -> u128 {
        // TODO! we must also validate the cases of duplicates!
        let mut hb = PlRandomState::with_seed(0).build_hasher();
        categories.values_iter().for_each(|val| {
            val.hash(&mut hb);
        });
        let hash = hb.finish();
        (hash as u128) << 64 | (categories.total_buffer_len() as u128)
    }

    pub fn build_local(categories: Utf8ViewArray) -> Self {
        let hash = Self::build_hash(&categories);
        Self::Local(categories, hash)
    }

    /// Get the length of the [`RevMapping`]
    pub fn len(&self) -> usize {
        self.get_categories().len()
    }

    /// [`Categorical`] to [`str`]
    ///
    /// [`Categorical`]: crate::datatypes::DataType::Categorical
    pub fn get(&self, idx: u32) -> &str {
        match self {
            Self::Global(map, a, _) => {
                let idx = *map.get(&idx).unwrap();
                a.value(idx as usize)
            },
            Self::Local(a, _) => a.value(idx as usize),
        }
    }

    pub fn get_optional(&self, idx: u32) -> Option<&str> {
        match self {
            Self::Global(map, a, _) => {
                let idx = *map.get(&idx)?;
                a.get(idx as usize)
            },
            Self::Local(a, _) => a.get(idx as usize),
        }
    }

    /// [`Categorical`] to [`str`]
    ///
    /// [`Categorical`]: crate::datatypes::DataType::Categorical
    ///
    /// # Safety
    /// This doesn't do any bound checking
    pub(crate) unsafe fn get_unchecked(&self, idx: u32) -> &str {
        match self {
            Self::Global(map, a, _) => {
                let idx = *map.get(&idx).unwrap();
                a.value_unchecked(idx as usize)
            },
            Self::Local(a, _) => a.value_unchecked(idx as usize),
        }
    }
    /// Check if the categoricals have a compatible mapping
    #[inline]
    pub fn same_src(&self, other: &Self) -> bool {
        match (self, other) {
            (RevMapping::Global(_, _, l), RevMapping::Global(_, _, r)) => *l == *r,
            (RevMapping::Local(_, l_hash), RevMapping::Local(_, r_hash)) => l_hash == r_hash,
            _ => false,
        }
    }

    /// [`str`] to [`Categorical`]
    ///
    ///
    /// [`Categorical`]: crate::datatypes::DataType::Categorical
    pub fn find(&self, value: &str) -> Option<u32> {
        match self {
            Self::Global(rev_map, a, id) => {
                // fast path is check
                if using_string_cache() {
                    let map = crate::STRING_CACHE.read_map();
                    if map.uuid == *id {
                        return map.get_cat(value);
                    }
                }
                rev_map
                    .iter()
                    // SAFETY:
                    // value is always within bounds
                    .find(|(_k, &v)| (unsafe { a.value_unchecked(v as usize) } == value))
                    .map(|(k, _v)| *k)
            },

            Self::Local(a, _) => {
                // SAFETY: within bounds
                unsafe { (0..a.len()).find(|idx| a.value_unchecked(*idx) == value) }
                    .map(|idx| idx as u32)
            },
        }
    }
}
