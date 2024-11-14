use hashbrown::hash_table::Entry;
use hashbrown::HashTable;

use super::*;

pub fn create_categorical_chunked_listbuilder(
    name: PlSmallStr,
    ordering: CategoricalOrdering,
    capacity: usize,
    values_capacity: usize,
    rev_map: Arc<RevMapping>,
) -> Box<dyn ListBuilderTrait> {
    match &*rev_map {
        RevMapping::Local(_, h) => Box::new(ListLocalCategoricalChunkedBuilder::new(
            name,
            ordering,
            capacity,
            values_capacity,
            *h,
        )),
        RevMapping::Global(_, _, _) => Box::new(ListGlobalCategoricalChunkedBuilder::new(
            name,
            ordering,
            capacity,
            values_capacity,
            rev_map,
        )),
    }
}

pub struct ListEnumCategoricalChunkedBuilder {
    inner: ListPrimitiveChunkedBuilder<UInt32Type>,
    ordering: CategoricalOrdering,
    rev_map: RevMapping,
}

impl ListEnumCategoricalChunkedBuilder {
    pub(super) fn new(
        name: PlSmallStr,
        ordering: CategoricalOrdering,
        capacity: usize,
        values_capacity: usize,
        rev_map: RevMapping,
    ) -> Self {
        Self {
            inner: ListPrimitiveChunkedBuilder::new(
                name,
                capacity,
                values_capacity,
                DataType::UInt32,
            ),
            ordering,
            rev_map,
        }
    }
}

impl ListBuilderTrait for ListEnumCategoricalChunkedBuilder {
    fn append_series(&mut self, s: &Series) -> PolarsResult<()> {
        let DataType::Enum(Some(rev_map), _) = s.dtype() else {
            polars_bail!(ComputeError: "expected enum type")
        };
        polars_ensure!(rev_map.same_src(&self.rev_map),ComputeError: "incompatible enum types");
        self.inner.append_series(s)
    }

    fn append_null(&mut self) {
        self.inner.append_null()
    }

    fn finish(&mut self) -> ListChunked {
        let inner_dtype = DataType::Enum(Some(Arc::new(self.rev_map.clone())), self.ordering);
        let mut ca = self.inner.finish();
        unsafe { ca.set_dtype(DataType::List(Box::new(inner_dtype))) }
        ca
    }
}

struct ListLocalCategoricalChunkedBuilder {
    inner: ListPrimitiveChunkedBuilder<UInt32Type>,
    idx_lookup: HashTable<u32>,
    ordering: CategoricalOrdering,
    categories: MutablePlString,
    categories_hash: u128,
}

impl ListLocalCategoricalChunkedBuilder {
    #[inline]
    pub fn get_hash_builder() -> PlRandomState {
        PlRandomState::with_seed(0)
    }

    pub(super) fn new(
        name: PlSmallStr,
        ordering: CategoricalOrdering,
        capacity: usize,
        values_capacity: usize,
        hash: u128,
    ) -> Self {
        Self {
            inner: ListPrimitiveChunkedBuilder::new(
                name,
                capacity,
                values_capacity,
                DataType::UInt32,
            ),
            idx_lookup: HashTable::with_capacity(capacity),
            ordering,
            categories: MutablePlString::with_capacity(capacity),
            categories_hash: hash,
        }
    }
}

impl ListBuilderTrait for ListLocalCategoricalChunkedBuilder {
    fn append_series(&mut self, s: &Series) -> PolarsResult<()> {
        let DataType::Categorical(Some(rev_map), _) = s.dtype() else {
            polars_bail!(ComputeError: "expected categorical type")
        };
        let RevMapping::Local(cats_right, new_hash) = &**rev_map else {
            polars_bail!(string_cache_mismatch)
        };
        let ca = s.categorical().unwrap();

        // Fast path rev_maps are compatible & lookup is initialized
        if self.categories_hash == *new_hash && !self.idx_lookup.is_empty() {
            return self.inner.append_series(s);
        }

        let hash_builder = ListLocalCategoricalChunkedBuilder::get_hash_builder();

        // Map the physical of the appended series to be compatible with the existing rev map
        let mut idx_mapping = PlHashMap::with_capacity(ca.len());

        for (idx, cat) in cats_right.values_iter().enumerate() {
            let hash_cat = hash_builder.hash_one(cat);
            let len = self.idx_lookup.len();

            // Custom hashing / equality functions for comparing the &str to the idx
            // SAFETY: index in hashmap are within bounds of categories
            unsafe {
                let r = self.idx_lookup.entry(
                    hash_cat,
                    |k| self.categories.value_unchecked(*k as usize) == cat,
                    |k| hash_builder.hash_one(self.categories.value_unchecked(*k as usize)),
                );

                match r {
                    Entry::Occupied(v) => {
                        // SAFETY: bucket is initialized.
                        idx_mapping.insert_unique_unchecked(idx as u32, *v.get());
                    },
                    Entry::Vacant(slot) => {
                        idx_mapping.insert_unique_unchecked(idx as u32, len as u32);
                        self.categories.push(Some(cat));
                        slot.insert(len as u32);
                    },
                }
            }
        }

        let op = |opt_v: Option<&u32>| opt_v.map(|v| *idx_mapping.get(v).unwrap());
        // SAFETY: length is correct as we do one-one mapping over ca.
        let iter = unsafe {
            ca.physical()
                .downcast_iter()
                .flat_map(|arr| arr.iter().map(op))
                .trust_my_length(ca.len())
        };
        self.inner.append_iter(iter);

        Ok(())
    }

    fn append_null(&mut self) {
        self.inner.append_null()
    }

    fn finish(&mut self) -> ListChunked {
        let categories: Utf8ViewArray = std::mem::take(&mut self.categories).into();
        let rev_map = RevMapping::build_local(categories);
        let inner_dtype = DataType::Categorical(Some(Arc::new(rev_map)), self.ordering);
        let mut ca = self.inner.finish();
        unsafe { ca.set_dtype(DataType::List(Box::new(inner_dtype))) }
        ca
    }
}

struct ListGlobalCategoricalChunkedBuilder {
    inner: ListPrimitiveChunkedBuilder<UInt32Type>,
    ordering: CategoricalOrdering,
    map_merger: GlobalRevMapMerger,
}

impl ListGlobalCategoricalChunkedBuilder {
    pub(super) fn new(
        name: PlSmallStr,
        ordering: CategoricalOrdering,
        capacity: usize,
        values_capacity: usize,
        rev_map: Arc<RevMapping>,
    ) -> Self {
        let inner =
            ListPrimitiveChunkedBuilder::new(name, capacity, values_capacity, DataType::UInt32);
        Self {
            inner,
            ordering,
            map_merger: GlobalRevMapMerger::new(rev_map),
        }
    }
}

impl ListBuilderTrait for ListGlobalCategoricalChunkedBuilder {
    fn append_series(&mut self, s: &Series) -> PolarsResult<()> {
        let DataType::Categorical(Some(rev_map), _) = s.dtype() else {
            polars_bail!(ComputeError: "expected categorical type")
        };
        self.map_merger.merge_map(rev_map)?;
        self.inner.append_series(s)
    }

    fn append_null(&mut self) {
        self.inner.append_null()
    }

    fn finish(&mut self) -> ListChunked {
        let rev_map = std::mem::take(&mut self.map_merger).finish();
        let inner_dtype = DataType::Categorical(Some(rev_map), self.ordering);
        let mut ca = self.inner.finish();
        unsafe { ca.set_dtype(DataType::List(Box::new(inner_dtype))) }
        ca
    }
}
