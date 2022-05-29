use crate::prelude::*;
use crate::{datatypes::PlHashMap, use_string_cache};
use arrow::array::*;

pub enum RevMappingBuilder {
    /// Hashmap: maps the indexes from the global cache/categorical array to indexes in the local Utf8Array
    /// Utf8Array: caches the string values
    Global(PlHashMap<u32, u32>, MutableUtf8Array<i64>, u128),
    /// Utf8Array: caches the string values
    Local(MutableUtf8Array<i64>),
}

impl RevMappingBuilder {
    fn insert(&mut self, idx: u32, value: &str) {
        use RevMappingBuilder::*;
        match self {
            Local(builder) => builder.push(Some(value)),
            Global(map, builder, _) => {
                if !map.contains_key(&idx) {
                    builder.push(Some(value));
                    let new_idx = builder.len() as u32 - 1;
                    map.insert(idx, new_idx);
                }
            }
        };
    }

    fn finish(self) -> RevMapping {
        use RevMappingBuilder::*;
        match self {
            Local(b) => RevMapping::Local(b.into()),
            Global(mut map, b, uuid) => {
                map.shrink_to_fit();
                RevMapping::Global(map, b.into(), uuid)
            }
        }
    }
}

#[derive(Debug)]
pub enum RevMapping {
    /// Hashmap: maps the indexes from the global cache/categorical array to indexes in the local Utf8Array
    /// Utf8Array: caches the string values
    Global(PlHashMap<u32, u32>, Utf8Array<i64>, u128),
    /// Utf8Array: caches the string values
    Local(Utf8Array<i64>),
}

impl Default for RevMapping {
    fn default() -> Self {
        let slice: &[Option<&str>] = &[];
        RevMapping::Local(Utf8Array::<i64>::from(slice))
    }
}

#[allow(clippy::len_without_is_empty)]
impl RevMapping {
    /// Get the length of the [`RevMapping`]
    pub fn len(&self) -> usize {
        match self {
            Self::Global(_, a, _) => a.len(),
            Self::Local(a) => a.len(),
        }
    }

    /// Categorical to str
    pub fn get(&self, idx: u32) -> &str {
        match self {
            Self::Global(map, a, _) => {
                let idx = *map.get(&idx).unwrap();
                a.value(idx as usize)
            }
            Self::Local(a) => a.value(idx as usize),
        }
    }

    /// Categorical to str
    ///
    /// # Safety
    /// This doesn't do any bound checking
    pub(crate) unsafe fn get_unchecked(&self, idx: u32) -> &str {
        match self {
            Self::Global(map, a, _) => {
                let idx = *map.get(&idx).unwrap();
                a.value_unchecked(idx as usize)
            }
            Self::Local(a) => a.value_unchecked(idx as usize),
        }
    }
    /// Check if the categoricals are created under the same global string cache.
    pub fn same_src(&self, other: &Self) -> bool {
        match (self, other) {
            (RevMapping::Global(_, _, l), RevMapping::Global(_, _, r)) => *l == *r,
            (RevMapping::Local(l), RevMapping::Local(r)) => {
                std::ptr::eq(l as *const Utf8Array<_>, r as *const Utf8Array<_>)
            }
            _ => false,
        }
    }

    /// str to Categorical
    pub fn find(&self, value: &str) -> Option<u32> {
        match self {
            Self::Global(map, a, _) => {
                map.iter()
                    // Safety:
                    // value is always within bounds
                    .find(|(_k, &v)| (unsafe { a.value_unchecked(v as usize) } == value))
                    .map(|(k, _v)| *k)
            }
            Self::Local(a) => {
                // Safety: within bounds
                unsafe { (0..a.len()).find(|idx| a.value_unchecked(*idx) == value) }
                    .map(|idx| idx as u32)
            }
        }
    }
}

pub struct CategoricalChunkedBuilder {
    array_builder: UInt32Vec,
    name: String,
    reverse_mapping: RevMappingBuilder,
}

impl CategoricalChunkedBuilder {
    pub fn new(name: &str, capacity: usize) -> Self {
        let builder = MutableUtf8Array::<i64>::with_capacity(capacity / 10);
        let reverse_mapping = if use_string_cache() {
            let uuid = crate::STRING_CACHE.lock_map().uuid;
            RevMappingBuilder::Global(PlHashMap::default(), builder, uuid)
        } else {
            RevMappingBuilder::Local(builder)
        };

        Self {
            array_builder: UInt32Vec::with_capacity(capacity),
            name: name.to_string(),
            reverse_mapping,
        }
    }
}
impl CategoricalChunkedBuilder {
    /// Appends all the values in a single lock of the global string cache.
    pub fn drain_iter<'a, I>(&mut self, i: I)
    where
        I: IntoIterator<Item = Option<&'a str>>,
    {
        if use_string_cache() {
            let mut cache = crate::STRING_CACHE.lock_map();

            for opt_s in i {
                match opt_s {
                    Some(s) => {
                        let idx = match cache.map.get(s) {
                            Some(idx) => *idx,
                            None => {
                                let idx = cache.map.len() as u32;
                                cache.map.insert(s.to_string(), idx);
                                idx
                            }
                        };
                        // we still need to check if the idx is already stored in our map
                        self.reverse_mapping.insert(idx, s);
                        self.array_builder.push(Some(idx));
                    }
                    None => {
                        self.array_builder.push(None);
                    }
                }
            }
        } else {
            let mut mapping = PlHashMap::new();
            for opt_s in i {
                match opt_s {
                    Some(s) => {
                        let idx = match mapping.get(s) {
                            Some(idx) => *idx,
                            None => {
                                let idx = mapping.len() as u32;
                                self.reverse_mapping.insert(idx, s);
                                mapping.insert(s, idx);
                                idx
                            }
                        };
                        self.array_builder.push(Some(idx));
                    }
                    None => {
                        self.array_builder.push(None);
                    }
                }
            }

            if mapping.len() > u32::MAX as usize {
                panic!("not more than {} categories supported", u32::MAX)
            };
        }
    }

    pub fn finish(self) -> CategoricalChunked {
        CategoricalChunked::from_chunks_original(
            &self.name,
            vec![self.array_builder.into_arc()],
            self.reverse_mapping.finish(),
        )
    }
}

#[cfg(test)]
mod test {
    use crate::chunked_array::categorical::CategoricalChunkedBuilder;
    use crate::prelude::*;
    use crate::{reset_string_cache, toggle_string_cache, SINGLE_LOCK};

    #[test]
    fn test_categorical_rev() -> Result<()> {
        let _lock = SINGLE_LOCK.lock();
        reset_string_cache();
        let slice = &[
            Some("foo"),
            None,
            Some("bar"),
            Some("foo"),
            Some("foo"),
            Some("bar"),
        ];
        let ca = Utf8Chunked::new("a", slice);
        let out = ca.cast(&DataType::Categorical(None))?;
        let out = out.categorical().unwrap().clone();
        assert_eq!(out.get_rev_map().len(), 2);

        // test the global branch
        toggle_string_cache(true);
        // empty global cache
        let out = ca.cast(&DataType::Categorical(None))?;
        let out = out.categorical().unwrap().clone();
        assert_eq!(out.get_rev_map().len(), 2);
        // full global cache
        let out = ca.cast(&DataType::Categorical(None))?;
        let out = out.categorical().unwrap().clone();
        assert_eq!(out.get_rev_map().len(), 2);

        // Check that we don't panic if we append two categorical arrays
        // build under the same string cache
        // https://github.com/pola-rs/polars/issues/1115
        let ca1 = Utf8Chunked::new("a", slice).cast(&DataType::Categorical(None))?;
        let mut ca1 = ca1.categorical().unwrap().clone();
        let ca2 = Utf8Chunked::new("a", slice).cast(&DataType::Categorical(None))?;
        let ca2 = ca2.categorical().unwrap();
        ca1.append(ca2).unwrap();

        Ok(())
    }

    #[test]
    fn test_categorical_builder() {
        use crate::{reset_string_cache, toggle_string_cache};
        let _lock = crate::SINGLE_LOCK.lock();
        for b in &[false, true] {
            reset_string_cache();
            toggle_string_cache(*b);

            // Use 2 builders to check if the global string cache
            // does not interfere with the index mapping
            let mut builder1 = CategoricalChunkedBuilder::new("foo", 10);
            let mut builder2 = CategoricalChunkedBuilder::new("foo", 10);
            builder1.drain_iter(vec![None, Some("hello"), Some("vietnam")]);
            builder2.drain_iter(vec![Some("hello"), None, Some("world")].into_iter());

            let s = builder1.finish().into_series();
            assert_eq!(s.str_value(0), "null");
            assert_eq!(s.str_value(1), "hello");
            assert_eq!(s.str_value(2), "vietnam");

            let s = builder2.finish().into_series();
            assert_eq!(s.str_value(0), "hello");
            assert_eq!(s.str_value(1), "null");
            assert_eq!(s.str_value(2), "world");
        }
    }
}
