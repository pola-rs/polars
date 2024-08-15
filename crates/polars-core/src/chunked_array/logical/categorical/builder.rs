use arrow::array::*;
use arrow::legacy::trusted_len::TrustedLenPush;
use hashbrown::hash_map::Entry;
use polars_utils::itertools::Itertools;

use crate::hashing::_HASHMAP_INIT_SIZE;
use crate::prelude::*;
use crate::{using_string_cache, StringCache, POOL};

// Wrap u32 key to avoid incorrect usage of hashmap with custom lookup
#[repr(transparent)]
struct KeyWrapper(u32);

pub struct CategoricalChunkedBuilder {
    cat_builder: UInt32Vec,
    name: String,
    ordering: CategoricalOrdering,
    categories: MutablePlString,
    // hashmap utilized by the local builder
    local_mapping: PlHashMap<KeyWrapper, ()>,
}

impl CategoricalChunkedBuilder {
    pub fn new(name: &str, capacity: usize, ordering: CategoricalOrdering) -> Self {
        Self {
            cat_builder: UInt32Vec::with_capacity(capacity),
            name: name.to_string(),
            ordering,
            categories: MutablePlString::with_capacity(_HASHMAP_INIT_SIZE),
            local_mapping: PlHashMap::with_capacity_and_hasher(
                capacity / 10,
                StringCache::get_hash_builder(),
            ),
        }
    }

    fn get_cat_idx(&mut self, s: &str, h: u64) -> (u32, bool) {
        let len = self.local_mapping.len() as u32;

        // Custom hashing / equality functions for comparing the &str to the idx
        // SAFETY: index in hashmap are within bounds of categories
        let r = unsafe {
            self.local_mapping.raw_table_mut().find_or_find_insert_slot(
                h,
                |(k, _)| self.categories.value_unchecked(k.0 as usize) == s,
                |(k, _): &(KeyWrapper, ())| {
                    StringCache::get_hash_builder()
                        .hash_one(self.categories.value_unchecked(k.0 as usize))
                },
            )
        };

        match r {
            Ok(v) => {
                // SAFETY: Bucket is initialized
                (unsafe { v.as_ref().0 .0 }, false)
            },
            Err(e) => {
                self.categories.push(Some(s));
                // SAFETY: No mutations in hashmap since find_or_find_insert_slot call
                unsafe {
                    self.local_mapping
                        .raw_table_mut()
                        .insert_in_slot(h, e, (KeyWrapper(len), ()))
                };
                (len, true)
            },
        }
    }

    /// Registers a value to a categorical index without pushing it.
    /// Returns the index and if the value was new.
    #[inline]
    pub fn register_value(&mut self, s: &str) -> (u32, bool) {
        let h = self.local_mapping.hasher().hash_one(s);
        self.get_cat_idx(s, h)
    }

    #[inline]
    pub fn append_value(&mut self, s: &str) {
        let h = self.local_mapping.hasher().hash_one(s);
        let idx = self.get_cat_idx(s, h).0;
        self.cat_builder.push(Some(idx));
    }

    #[inline]
    pub fn append_null(&mut self) {
        self.cat_builder.push(None)
    }

    #[inline]
    pub fn append(&mut self, opt_s: Option<&str>) {
        match opt_s {
            None => self.append_null(),
            Some(s) => self.append_value(s),
        }
    }

    fn drain_iter<'a, I>(&mut self, i: I)
    where
        I: IntoIterator<Item = Option<&'a str>>,
    {
        for opt_s in i.into_iter() {
            self.append(opt_s);
        }
    }

    /// Fast path for global categorical which preserves hashes and saves an allocation by
    /// altering the keys in place.
    fn drain_iter_global_and_finish<'a, I>(&mut self, i: I) -> CategoricalChunked
    where
        I: IntoIterator<Item = Option<&'a str>>,
    {
        let iter = i.into_iter();
        // Save hashes for later when inserting into the global hashmap.
        let mut hashes = Vec::with_capacity(_HASHMAP_INIT_SIZE);
        for s in self.categories.values_iter() {
            hashes.push(self.local_mapping.hasher().hash_one(s));
        }

        for opt_s in iter {
            match opt_s {
                None => self.append_null(),
                Some(s) => {
                    let hash = self.local_mapping.hasher().hash_one(s);
                    let (cat_idx, new) = self.get_cat_idx(s, hash);
                    self.cat_builder.push(Some(cat_idx));
                    if new {
                        // We appended a value to the map.
                        hashes.push(hash);
                    }
                },
            }
        }

        let categories = std::mem::take(&mut self.categories).freeze();

        // We will create a mapping from our local categoricals to global categoricals
        // and a mapping from global categoricals to our local categoricals.
        let mut local_to_global: Vec<u32> = Vec::with_capacity(categories.len());
        let (id, local_to_global) = crate::STRING_CACHE.apply(|cache| {
            for (s, h) in categories.values_iter().zip(hashes) {
                // SAFETY: we allocated enough.
                unsafe { local_to_global.push_unchecked(cache.insert_from_hash(h, s)) }
            }
            local_to_global
        });

        // Change local indices inplace to their global counterparts.
        let update_cats = || {
            if !local_to_global.is_empty() {
                // when all categorical are null, `local_to_global` is empty and all cats physical values are 0.
                self.cat_builder.apply_values(|cats| {
                    for cat in cats {
                        debug_assert!((*cat as usize) < local_to_global.len());
                        *cat = *unsafe { local_to_global.get_unchecked(*cat as usize) };
                    }
                })
            }
        };

        let mut global_to_local = PlHashMap::with_capacity(local_to_global.len());
        POOL.join(
            || fill_global_to_local(&local_to_global, &mut global_to_local),
            update_cats,
        );

        let indices = std::mem::take(&mut self.cat_builder).into();
        let indices = UInt32Chunked::with_chunk(&self.name, indices);

        // SAFETY: indices are in bounds of new rev_map
        unsafe {
            CategoricalChunked::from_cats_and_rev_map_unchecked(
                indices,
                Arc::new(RevMapping::Global(global_to_local, categories, id)),
                false,
                self.ordering,
            )
        }
        .with_fast_unique(true)
    }

    pub fn drain_iter_and_finish<'a, I>(mut self, i: I) -> CategoricalChunked
    where
        I: IntoIterator<Item = Option<&'a str>>,
    {
        if using_string_cache() {
            self.drain_iter_global_and_finish(i)
        } else {
            self.drain_iter(i);
            self.finish()
        }
    }

    pub fn finish(self) -> CategoricalChunked {
        // SAFETY: keys and values are in bounds
        unsafe {
            CategoricalChunked::from_keys_and_values(
                &self.name,
                &self.cat_builder.into(),
                &self.categories.into(),
                self.ordering,
            )
        }
        .with_fast_unique(true)
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
    /// This does not do any bound checks
    pub unsafe fn from_global_indices_unchecked(
        cats: UInt32Chunked,
        ordering: CategoricalOrdering,
    ) -> CategoricalChunked {
        let cache = crate::STRING_CACHE.read_map();

        let cap = std::cmp::min(std::cmp::min(cats.len(), cache.len()), _HASHMAP_INIT_SIZE);
        let mut rev_map = PlHashMap::with_capacity(cap);
        let mut str_values = MutablePlString::with_capacity(cap);

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

        CategoricalChunked::from_cats_and_rev_map_unchecked(
            cats,
            Arc::new(rev_map),
            false,
            ordering,
        )
    }

    pub(crate) unsafe fn from_keys_and_values_global(
        name: &str,
        keys: impl IntoIterator<Item = Option<u32>> + Send,
        capacity: usize,
        values: &Utf8ViewArray,
        ordering: CategoricalOrdering,
    ) -> Self {
        // Vec<u32> where the index is local and the value is the global index
        let mut local_to_global: Vec<u32> = Vec::with_capacity(values.len());
        let (id, local_to_global) = crate::STRING_CACHE.apply(|cache| {
            // locally we don't need a hashmap because we all categories are 1 integer apart
            // so the index is local, and the values is global
            for s in values.values_iter() {
                // SAFETY: we allocated enough
                unsafe { local_to_global.push_unchecked(cache.insert(s)) }
            }
            local_to_global
        });

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

        let mut global_to_local = PlHashMap::with_capacity(local_to_global.len());
        let (_, cats) = POOL.join(
            || fill_global_to_local(&local_to_global, &mut global_to_local),
            compute_cats,
        );
        unsafe {
            CategoricalChunked::from_cats_and_rev_map_unchecked(
                UInt32Chunked::with_chunk(name, cats.into()),
                Arc::new(RevMapping::Global(global_to_local, values.clone(), id)),
                false,
                ordering,
            )
        }
    }

    pub(crate) unsafe fn from_keys_and_values_local(
        name: &str,
        keys: &PrimitiveArray<u32>,
        values: &Utf8ViewArray,
        ordering: CategoricalOrdering,
    ) -> CategoricalChunked {
        CategoricalChunked::from_cats_and_rev_map_unchecked(
            UInt32Chunked::with_chunk(name, keys.clone()),
            Arc::new(RevMapping::build_local(values.clone())),
            false,
            ordering,
        )
    }

    /// # Safety
    /// The caller must ensure that index values in the `keys` are in within bounds of the `values` length.
    pub(crate) unsafe fn from_keys_and_values(
        name: &str,
        keys: &PrimitiveArray<u32>,
        values: &Utf8ViewArray,
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
        categories: &Utf8ViewArray,
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
        let iter = values.downcast_iter().map(|arr| {
            arr.iter()
                .map(|opt_s: Option<&str>| opt_s.and_then(|s| map.get(s).copied()))
                .collect_arr()
        });
        let mut keys: UInt32Chunked = ChunkedArray::from_chunk_iter(values.name(), iter);
        keys.rename(values.name());
        let rev_map = RevMapping::build_local(categories.clone());
        unsafe {
            Ok(CategoricalChunked::from_cats_and_rev_map_unchecked(
                keys,
                Arc::new(rev_map),
                true,
                ordering,
            ))
        }
    }
}

#[cfg(test)]
mod test {
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
            let builder1 = CategoricalChunkedBuilder::new("foo", 10, Default::default());
            let builder2 = CategoricalChunkedBuilder::new("foo", 10, Default::default());
            let s = builder1
                .drain_iter_and_finish(vec![None, Some("hello"), Some("vietnam")])
                .into_series();
            assert_eq!(s.str_value(0).unwrap(), "null");
            assert_eq!(s.str_value(1).unwrap(), "hello");
            assert_eq!(s.str_value(2).unwrap(), "vietnam");

            let s = builder2
                .drain_iter_and_finish(vec![Some("hello"), None, Some("world")])
                .into_series();
            assert_eq!(s.str_value(0).unwrap(), "hello");
            assert_eq!(s.str_value(1).unwrap(), "null");
            assert_eq!(s.str_value(2).unwrap(), "world");
        }
    }
}
