use std::fmt::{Debug, Formatter};

use ahash::RandomState;
use arrow::array::*;
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
    Global(PlHashMap<u32, u32>, Utf8Array<i64>, u32),
    /// Utf8Array: caches the string values and a hash of all values for quick comparison
    Local(Utf8Array<i64>, u128),
    /// Utf8Array: fixed user defined array of categories which caches the string values
    Enum(Utf8Array<i64>, u128),
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
            RevMapping::Enum(_, _) => {
                write!(f, "enum")
            },
        }
    }
}

impl Default for RevMapping {
    fn default() -> Self {
        let slice: &[Option<&str>] = &[];
        let cats = Utf8Array::<i64>::from(slice);
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

    #[inline]
    pub fn is_enum(&self) -> bool {
        matches!(self, Self::Enum(_, _))
    }

    /// Get the categories in this [`RevMapping`]
    pub fn get_categories(&self) -> &Utf8Array<i64> {
        match self {
            Self::Global(_, a, _) => a,
            Self::Local(a, _) | Self::Enum(a, _) => a,
        }
    }

    fn build_hash(categories: &Utf8Array<i64>) -> u128 {
        let hash_builder = RandomState::with_seed(0);
        let value_hash = hash_builder.hash_one(categories.values().as_slice());
        let offset_hash = hash_builder.hash_one(categories.offsets().as_slice());
        (value_hash as u128) << 64 | (offset_hash as u128)
    }

    pub fn build_enum(categories: Utf8Array<i64>) -> Self {
        let hash = Self::build_hash(&categories);
        Self::Enum(categories, hash)
    }

    pub fn build_local(categories: Utf8Array<i64>) -> Self {
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
            Self::Local(a, _) | Self::Enum(a, _) => a.value(idx as usize),
        }
    }

    pub fn get_optional(&self, idx: u32) -> Option<&str> {
        match self {
            Self::Global(map, a, _) => {
                let idx = *map.get(&idx)?;
                a.get(idx as usize)
            },
            Self::Local(a, _) | Self::Enum(a, _) => a.get(idx as usize),
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
            Self::Local(a, _) | Self::Enum(a, _) => a.value_unchecked(idx as usize),
        }
    }
    /// Check if the categoricals have a compatible mapping
    #[inline]
    pub fn same_src(&self, other: &Self) -> bool {
        match (self, other) {
            (RevMapping::Global(_, _, l), RevMapping::Global(_, _, r)) => *l == *r,
            (RevMapping::Local(_, l_hash), RevMapping::Local(_, r_hash)) => l_hash == r_hash,
            (RevMapping::Enum(_, l_hash), RevMapping::Enum(_, r_hash)) => l_hash == r_hash,
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
                    // Safety:
                    // value is always within bounds
                    .find(|(_k, &v)| (unsafe { a.value_unchecked(v as usize) } == value))
                    .map(|(k, _v)| *k)
            },

            Self::Local(a, _) | Self::Enum(a, _) => {
                // Safety: within bounds
                unsafe { (0..a.len()).find(|idx| a.value_unchecked(*idx) == value) }
                    .map(|idx| idx as u32)
            },
        }
    }
}
