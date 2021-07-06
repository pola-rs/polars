use crate::prelude::*;
use crate::{
    datatypes::PlHashMap,
    use_string_cache,
    utils::arrow::array::{Array, ArrayBuilder},
};
use arrow::array::{LargeStringArray, LargeStringBuilder, PrimitiveBuilder};
use std::marker::PhantomData;

pub enum RevMappingBuilder {
    Global(PlHashMap<u32, u32>, LargeStringBuilder, u128),
    Local(LargeStringBuilder),
}

impl RevMappingBuilder {
    fn insert(&mut self, idx: u32, value: &str) {
        use RevMappingBuilder::*;
        match self {
            Local(builder) => builder.append_value(value).unwrap(),
            Global(map, builder, _) => {
                if !map.contains_key(&idx) {
                    builder.append_value(value).unwrap();
                    let new_idx = builder.len() as u32 - 1;
                    map.insert(idx, new_idx);
                }
            }
        };
    }

    fn finish(self) -> RevMapping {
        use RevMappingBuilder::*;
        match self {
            Local(mut b) => RevMapping::Local(b.finish()),
            Global(mut map, mut b, uuid) => {
                map.shrink_to_fit();
                RevMapping::Global(map, b.finish(), uuid)
            }
        }
    }
}

pub enum RevMapping {
    Global(PlHashMap<u32, u32>, LargeStringArray, u128),
    Local(LargeStringArray),
}

#[allow(clippy::len_without_is_empty)]
impl RevMapping {
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
    /// Check if the categoricals are created under the same global string cache.
    pub fn same_src(&self, other: &Self) -> bool {
        match (self, other) {
            (RevMapping::Global(_, _, l), RevMapping::Global(_, _, r)) => *l == *r,
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
    array_builder: PrimitiveBuilder<UInt32Type>,
    field: Field,
    reverse_mapping: RevMappingBuilder,
}

impl CategoricalChunkedBuilder {
    pub fn new(name: &str, capacity: usize) -> Self {
        let builder = LargeStringBuilder::new(capacity / 10);
        let reverse_mapping = if use_string_cache() {
            let uuid = crate::STRING_CACHE.lock_map().uuid;
            RevMappingBuilder::Global(PlHashMap::default(), builder, uuid)
        } else {
            RevMappingBuilder::Local(builder)
        };

        CategoricalChunkedBuilder {
            array_builder: PrimitiveBuilder::<UInt32Type>::new(capacity),
            field: Field::new(name, DataType::Categorical),
            reverse_mapping,
        }
    }
}
impl CategoricalChunkedBuilder {
    /// Appends all the values in a single lock of the global string cache.
    pub fn from_iter<'a, I>(&mut self, i: I)
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
                        self.array_builder.append_value(idx).unwrap();
                    }
                    None => {
                        self.array_builder.append_null().unwrap();
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
                        self.array_builder.append_value(idx).unwrap();
                    }
                    None => {
                        self.array_builder.append_null().unwrap();
                    }
                }
            }

            if mapping.len() > u32::MAX as usize {
                panic!("not more than {} categories supported", u32::MAX)
            };
        }
    }

    pub fn finish(mut self) -> ChunkedArray<CategoricalType> {
        let arr = Arc::new(self.array_builder.finish());
        ChunkedArray {
            field: Arc::new(self.field),
            chunks: vec![arr],
            phantom: PhantomData,
            categorical_map: Some(Arc::new(self.reverse_mapping.finish())),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use crate::{reset_string_cache, toggle_string_cache, SINGLE_LOCK};

    #[test]
    fn test_categorical_rev() -> Result<()> {
        let _lock = SINGLE_LOCK.lock();
        reset_string_cache();
        let ca = Utf8Chunked::new_from_opt_slice(
            "a",
            &[
                Some("foo"),
                None,
                Some("bar"),
                Some("foo"),
                Some("foo"),
                Some("bar"),
            ],
        );
        let out = ca.cast::<CategoricalType>()?;
        assert_eq!(out.categorical_map.unwrap().len(), 2);

        // test the global branch
        toggle_string_cache(true);
        // empty global cache
        let out = ca.cast::<CategoricalType>()?;
        assert_eq!(out.categorical_map.unwrap().len(), 2);
        // full global cache
        let out = ca.cast::<CategoricalType>()?;
        assert_eq!(out.categorical_map.unwrap().len(), 2);
        Ok(())
    }
}
