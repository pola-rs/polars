use std::hash::{Hash, Hasher};

use arrow::array::*;
use arrow::legacy::trusted_len::TrustedLenPush;
use hashbrown::hash_map::{Entry, RawEntryMut};
use polars_utils::iter::EnumerateIdxTrait;

use crate::datatypes::PlHashMap;
use crate::hashing::_HASHMAP_INIT_SIZE;
use crate::prelude::*;
use crate::{using_string_cache, StringCache, POOL};

#[derive(Eq, Copy, Clone)]
pub struct StrHashLocal<'a> {
    str: &'a str,
    hash: u64,
}

impl<'a> Hash for StrHashLocal<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash)
    }
}

impl<'a> StrHashLocal<'a> {
    #[inline]
    pub(crate) fn new(s: &'a str, hash: u64) -> Self {
        Self { str: s, hash }
    }
}

impl<'a> PartialEq for StrHashLocal<'a> {
    fn eq(&self, other: &Self) -> bool {
        // can be collisions in the hashtable even though the hashes are equal
        // e.g. hashtable hash = hash % n_slots
        (self.hash == other.hash) && (self.str == other.str)
    }
}

pub struct CategoricalChunkedBuilder<'a> {
    cat_builder: UInt32Vec,
    name: String,
    ordering: CategoricalOrdering,
    reverse_mapping: MutableUtf8Array<i64>,
    // hashmap utilized by the local builder
    local_mapping: PlHashMap<StrHashLocal<'a>, u32>,
    // stored hashes from local builder
    hashes: Vec<u64>,
    fast_unique: bool,
}

impl CategoricalChunkedBuilder<'_> {
    pub fn new(name: &str, capacity: usize, ordering: CategoricalOrdering) -> Self {
        let reverse_mapping = MutableUtf8Array::<i64>::with_capacity(capacity / 10);

        Self {
            cat_builder: UInt32Vec::with_capacity(capacity),
            name: name.to_string(),
            ordering,
            reverse_mapping,
            local_mapping: PlHashMap::with_capacity_and_hasher(
                _HASHMAP_INIT_SIZE,
                StringCache::get_hash_builder(),
            ),
            hashes: vec![],
            fast_unique: true,
        }
    }
}
impl<'a> CategoricalChunkedBuilder<'a> {
    fn push_impl(&mut self, s: &'a str, store_hashes: bool) {
        let h = self.local_mapping.hasher().hash_one(s);
        let key = StrHashLocal::new(s, h);
        let mut idx = self.local_mapping.len() as u32;

        let entry = self
            .local_mapping
            .raw_entry_mut()
            .from_key_hashed_nocheck(h, &key);

        match entry {
            RawEntryMut::Occupied(entry) => idx = *entry.get(),
            RawEntryMut::Vacant(entry) => {
                if store_hashes {
                    self.hashes.push(h)
                }
                entry.insert_with_hasher(h, key, idx, |s| s.hash);
                self.reverse_mapping.push(Some(s));
            },
        };
        self.cat_builder.push(Some(idx));
    }

    // Prefill the rev_map with categories in a certain order
    pub fn prefill_rev_map(&mut self, categories: &'a Utf8Array<i64>) -> PolarsResult<()> {
        polars_ensure!(self.local_mapping.is_empty(), ComputeError: "prefill only at the start of building a Categorical");
        for (idx, s) in categories.values_iter().enumerate_idx() {
            let h = self.local_mapping.hasher().hash_one(s);
            let key = StrHashLocal::new(s, h);

            if let RawEntryMut::Vacant(entry) = self
                .local_mapping
                .raw_entry_mut()
                .from_key_hashed_nocheck(h, &key)
            {
                #[allow(clippy::unnecessary_cast)]
                entry.insert_with_hasher(h, key, idx as u32, |s| s.hash);
                self.reverse_mapping.push(Some(s));
            }
        }
        self.fast_unique = false;
        Ok(())
    }

    /// Check if this categorical already exists
    pub fn exits(&self, s: &str) -> bool {
        let h = self.local_mapping.hasher().hash_one(s);
        let key = StrHashLocal::new(s, h);
        self.local_mapping.contains_key(&key)
    }

    #[inline]
    pub fn append_value(&mut self, s: &'a str) {
        self.push_impl(s, false)
    }

    #[inline]
    pub fn append_null(&mut self) {
        self.cat_builder.push(None)
    }

    /// `store_hashes` is not needed by the local builder, only for the global builder under contention
    /// The hashes have the same order as the [`Utf8Array`] values.
    fn build_local_map<I>(&mut self, i: I, store_hashes: bool) -> Vec<u64>
    where
        I: IntoIterator<Item = Option<&'a str>>,
    {
        let mut iter = i.into_iter();
        if store_hashes {
            self.hashes = Vec::with_capacity(iter.size_hint().0 / 10 + self.reverse_mapping.len());
            for s in self.reverse_mapping.values_iter() {
                self.hashes.push(self.local_mapping.hasher().hash_one(s));
            }
        }

        for opt_s in &mut iter {
            match opt_s {
                Some(s) => self.push_impl(s, store_hashes),
                None => self.append_null(),
            }
        }

        if self.local_mapping.len() > u32::MAX as usize {
            panic!("not more than {} categories supported", u32::MAX)
        };
        // drop the hashmap
        std::mem::take(&mut self.local_mapping);
        std::mem::take(&mut self.hashes)
    }

    fn build_global_map_contention<I>(&mut self, i: I) -> RevMapping
    where
        I: IntoIterator<Item = Option<&'a str>>,
    {
        // first build the values: [`Utf8Array`]
        // we can use a local hashmap for that
        // `hashes.len()` is equal to to the number of unique values.
        let hashes = self.build_local_map(i, true);

        // locally we don't need a hashmap because we all categories are 1 integer apart
        // so the index is local, and the values is global
        let mut local_to_global: Vec<u32>;
        let id;

        // now we have to lock the global string cache.
        // we will create a mapping from our local categoricals to global categoricals
        // and a mapping from global categoricals to our local categoricals
        debug_assert_eq!(hashes.len(), self.reverse_mapping.len());
        local_to_global = Vec::with_capacity(self.reverse_mapping.len());
        let values: Utf8Array<i64> = std::mem::take(&mut self.reverse_mapping).into();

        // in a separate scope so that we drop the global cache as soon as we are finished
        {
            let cache = &mut crate::STRING_CACHE.lock_map();
            id = cache.uuid;

            for (s, h) in values.values_iter().zip(hashes) {
                let global_idx = cache.insert_from_hash(h, s);
                // safety:
                // we allocated enough
                unsafe { local_to_global.push_unchecked(global_idx) }
            }
            if cache.len() > u32::MAX as usize {
                panic!("not more than {} categories supported", u32::MAX)
            };
        }
        // we now know the exact size
        // no reallocs
        let mut global_to_local = PlHashMap::with_capacity(local_to_global.len());

        let update_cats = || {
            if !local_to_global.is_empty() {
                // when all categorical are null, `local_to_global` is empty and all cats physical values are 0.
                self.cat_builder.apply_values(|cats| {
                    for cat in cats {
                        debug_assert!((*cat as usize) < local_to_global.len());
                        *cat = *unsafe { local_to_global.get_unchecked(*cat as usize) };
                    }
                })
            };
        };

        POOL.join(
            || fill_global_to_local(&local_to_global, &mut global_to_local),
            update_cats,
        );

        RevMapping::Global(global_to_local, values, id)
    }

    pub fn drain_iter_and_finish<I>(mut self, i: I) -> CategoricalChunked
    where
        I: IntoIterator<Item = Option<&'a str>>,
    {
        // Appends all the values in a single lock of the global string cache reducing contention
        if using_string_cache() {
            let rev_map = self.build_global_map_contention(i);
            let arr: PrimitiveArray<u32> = std::mem::take(&mut self.cat_builder).into();
            // Safety: keys and values are in bounds
            let mut ca = unsafe {
                CategoricalChunked::from_cats_and_rev_map_unchecked(
                    UInt32Chunked::with_chunk(&self.name, arr),
                    Arc::new(rev_map),
                    self.ordering,
                )
            };
            ca.set_fast_unique(self.fast_unique);
            ca
        } else {
            let _ = self.build_local_map(i, false);
            self.finish()
        }
    }

    pub fn finish(self) -> CategoricalChunked {
        // Safety: keys and values are in bounds
        let mut ca = unsafe {
            CategoricalChunked::from_keys_and_values(
                &self.name,
                &self.cat_builder.into(),
                &self.reverse_mapping.into(),
                self.ordering,
            )
        };
        ca.set_fast_unique(self.fast_unique);
        ca
    }
}

fn fill_global_to_local(local_to_global: &[u32], global_to_local: &mut PlHashMap<u32, u32>) {
    let mut local_idx = 0;
    #[allow(clippy::explicit_counter_loop)]
    for global_idx in local_to_global {
        // we know the keys are unique so this is much faster
        global_to_local.insert_unique_unchecked(*global_idx, local_idx);
        local_idx += 1;
    }
}

impl CategoricalChunked {
    /// Create a [`CategoricalChunked`] from a categorical indices. The indices will
    /// probe the global string cache.
    pub(crate) fn from_global_indices(
        cats: UInt32Chunked,
        ordering: CategoricalOrdering,
    ) -> PolarsResult<CategoricalChunked> {
        let len = crate::STRING_CACHE.read_map().len() as u32;
        let oob = cats.into_iter().flatten().any(|cat| cat >= len);
        polars_ensure!(
            !oob,
            ComputeError:
            "cannot construct Categorical from these categories; at least one of them is out of bounds"
        );
        Ok(unsafe { Self::from_global_indices_unchecked(cats, ordering) })
    }

    /// Create a [`CategoricalChunked`] from a categorical indices. The indices will
    /// probe the global string cache.
    ///
    /// # Safety
    ///
    /// This does not do any bound checks
    pub unsafe fn from_global_indices_unchecked(
        cats: UInt32Chunked,
        ordering: CategoricalOrdering,
    ) -> CategoricalChunked {
        let cache = crate::STRING_CACHE.read_map();

        let cap = std::cmp::min(std::cmp::min(cats.len(), cache.len()), _HASHMAP_INIT_SIZE);
        let mut rev_map = PlHashMap::with_capacity(cap);
        let mut str_values = MutableUtf8Array::with_capacities(cap, cap * 24);

        for arr in cats.downcast_iter() {
            for cat in arr.into_iter().flatten().copied() {
                let offset = str_values.len() as u32;

                if let Entry::Vacant(entry) = rev_map.entry(cat) {
                    entry.insert(offset);
                    let str_val = cache.get_unchecked(cat);
                    str_values.push(Some(str_val))
                }
            }
        }

        let rev_map = RevMapping::Global(rev_map, str_values.into(), cache.uuid);

        CategoricalChunked::from_cats_and_rev_map_unchecked(cats, Arc::new(rev_map), ordering)
    }

    pub unsafe fn from_keys_and_values_global(
        name: &str,
        keys: impl IntoIterator<Item = Option<u32>> + Send,
        capacity: usize,
        values: &Utf8Array<i64>,
        ordering: CategoricalOrdering,
    ) -> Self {
        // locally we don't need a hashmap because we all categories are 1 integer apart
        // so the index is local, and the values is global
        let mut local_to_global: Vec<u32> = Vec::with_capacity(values.len());
        let id;

        // now we have to lock the global string cache.
        // we will create a mapping from our local categoricals to global categoricals
        // and a mapping from global categoricals to our local categoricals

        // in a separate scope so that we drop the global cache as soon as we are finished
        {
            let cache = &mut crate::STRING_CACHE.lock_map();
            id = cache.uuid;

            for s in values.values_iter() {
                let global_idx = cache.insert(s);

                // safety:
                // we allocated enough
                unsafe { local_to_global.push_unchecked(global_idx) }
            }
            if cache.len() > u32::MAX as usize {
                panic!("not more than {} categories supported", u32::MAX)
            };
        }
        // we now know the exact size
        // no reallocs
        let mut global_to_local = PlHashMap::with_capacity(local_to_global.len());
        let compute_cats = || {
            let mut result = UInt32Vec::with_capacity(capacity);

            for opt_value in keys.into_iter() {
                result.push(opt_value.map(|cat| {
                    debug_assert!((cat as usize) < local_to_global.len());
                    *unsafe { local_to_global.get_unchecked(cat as usize) }
                }));
            }
            result
        };

        let (_, cats) = POOL.join(
            || fill_global_to_local(&local_to_global, &mut global_to_local),
            compute_cats,
        );
        unsafe {
            CategoricalChunked::from_cats_and_rev_map_unchecked(
                UInt32Chunked::with_chunk(name, cats.into()),
                Arc::new(RevMapping::Global(global_to_local, values.clone(), id)),
                ordering,
            )
        }
    }

    pub(crate) unsafe fn from_keys_and_values_local(
        name: &str,
        keys: &PrimitiveArray<u32>,
        values: &Utf8Array<i64>,
        ordering: CategoricalOrdering,
    ) -> CategoricalChunked {
        CategoricalChunked::from_cats_and_rev_map_unchecked(
            UInt32Chunked::with_chunk(name, keys.clone()),
            Arc::new(RevMapping::build_local(values.clone())),
            ordering,
        )
    }

    /// # Safety
    /// The caller must ensure that index values in the `keys` are in within bounds of the `values` length.
    pub(crate) unsafe fn from_keys_and_values(
        name: &str,
        keys: &PrimitiveArray<u32>,
        values: &Utf8Array<i64>,
        ordering: CategoricalOrdering,
    ) -> Self {
        if !using_string_cache() {
            CategoricalChunked::from_keys_and_values_local(name, keys, values, ordering)
        } else {
            CategoricalChunked::from_keys_and_values_global(
                name,
                keys.into_iter().map(|c| c.copied()),
                keys.len(),
                values,
                ordering,
            )
        }
    }

    /// Create a [`CategoricalChunked`] from a fixed list of categories and a List of strings.
    /// This will error if a string is not in the fixed list of categories
    pub fn from_string_to_enum(
        values: &StringChunked,
        categories: &Utf8Array<i64>,
        ordering: CategoricalOrdering,
    ) -> PolarsResult<CategoricalChunked> {
        polars_ensure!(categories.null_count()  == 0, ComputeError: "categories can not contain null values");

        // Build a mapping string -> idx
        let mut map = PlHashMap::with_capacity(categories.len());
        for (idx, cat) in categories.values_iter().enumerate_idx() {
            #[allow(clippy::unnecessary_cast)]
            map.insert(cat, idx as u32);
        }
        // Find idx of every value in the map
        let ca_idx: UInt32Chunked = values
            .into_iter()
            .map(|opt_s: Option<&str>| {
                opt_s
                    .map(|s| {
                        map.get(s).copied().ok_or_else(|| {
                            polars_err!(not_in_enum, value = s, categories = categories)
                        })
                    })
                    .transpose()
            })
            .collect::<Result<UInt32Chunked, PolarsError>>()?;
        let rev_map = RevMapping::build_enum(categories.clone());
        unsafe {
            Ok(CategoricalChunked::from_cats_and_rev_map_unchecked(
                ca_idx,
                Arc::new(rev_map),
                ordering,
            ))
        }
    }
}

#[cfg(test)]
mod test {
    use crate::chunked_array::categorical::CategoricalChunkedBuilder;
    use crate::prelude::*;
    use crate::{disable_string_cache, enable_string_cache, SINGLE_LOCK};

    #[test]
    fn test_categorical_rev() -> PolarsResult<()> {
        let _lock = SINGLE_LOCK.lock();
        disable_string_cache();
        let slice = &[
            Some("foo"),
            None,
            Some("bar"),
            Some("foo"),
            Some("foo"),
            Some("bar"),
        ];
        let ca = StringChunked::new("a", slice);
        let out = ca.cast(&DataType::Categorical(None, Default::default()))?;
        let out = out.categorical().unwrap().clone();
        assert_eq!(out.get_rev_map().len(), 2);

        // test the global branch
        enable_string_cache();
        // empty global cache
        let out = ca.cast(&DataType::Categorical(None, Default::default()))?;
        let out = out.categorical().unwrap().clone();
        assert_eq!(out.get_rev_map().len(), 2);
        // full global cache
        let out = ca.cast(&DataType::Categorical(None, Default::default()))?;
        let out = out.categorical().unwrap().clone();
        assert_eq!(out.get_rev_map().len(), 2);

        // Check that we don't panic if we append two categorical arrays
        // build under the same string cache
        // https://github.com/pola-rs/polars/issues/1115
        let ca1 = StringChunked::new("a", slice)
            .cast(&DataType::Categorical(None, Default::default()))?;
        let mut ca1 = ca1.categorical().unwrap().clone();
        let ca2 = StringChunked::new("a", slice)
            .cast(&DataType::Categorical(None, Default::default()))?;
        let ca2 = ca2.categorical().unwrap();
        ca1.append(ca2).unwrap();

        Ok(())
    }

    #[test]
    fn test_categorical_builder() {
        use crate::{disable_string_cache, enable_string_cache};
        let _lock = crate::SINGLE_LOCK.lock();
        for use_string_cache in [false, true] {
            disable_string_cache();
            if use_string_cache {
                enable_string_cache();
            }

            // Use 2 builders to check if the global string cache
            // does not interfere with the index mapping
            let mut builder1 = CategoricalChunkedBuilder::new("foo", 10, Default::default());
            let mut builder2 = CategoricalChunkedBuilder::new("foo", 10, Default::default());
            let s = builder1
                .drain_iter_and_finish(vec![None, Some("hello"), Some("vietnam")])
                .into_series();
            assert_eq!(s.str_value(0).unwrap(), "null");
            assert_eq!(s.str_value(1).unwrap(), "hello");
            assert_eq!(s.str_value(2).unwrap(), "vietnam");

            let s = builder2.drain_iter_and_finish.into_series();
            assert_eq!(s.str_value(0).unwrap(), "hello");
            assert_eq!(s.str_value(1).unwrap(), "null");
            assert_eq!(s.str_value(2).unwrap(), "world");
        }
    }
}
