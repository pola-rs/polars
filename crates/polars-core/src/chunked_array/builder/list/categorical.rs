use hashbrown::hash_map::RawEntryMut;

use super::*;

pub fn create_categorical_chunked_listbuilder(
    name: &str,
    capacity: usize,
    values_capacity: usize,
    rev_map: Arc<RevMapping>,
) -> Box<dyn ListBuilderTrait> {
    match &*rev_map {
        RevMapping::Local(_) => Box::new(ListLocalCategoricalChunkedBuilder::new(
            name,
            capacity,
            values_capacity,
        )),
        RevMapping::Global(_, _, _) => Box::new(ListGlobalCategoricalChunkedBuilder::new(
            name,
            capacity,
            values_capacity,
            rev_map,
        )),
    }
}

struct ListLocalCategoricalChunkedBuilder {
    inner: ListPrimitiveChunkedBuilder<UInt32Type>,
    idx_lookup: PlHashMap<u32, ()>,
    categories: MutableUtf8Array<i64>,
}

impl ListLocalCategoricalChunkedBuilder {
    pub(super) fn new(name: &str, capacity: usize, values_capacity: usize) -> Self {
        Self {
            inner: ListPrimitiveChunkedBuilder::new(
                name,
                capacity,
                values_capacity,
                DataType::UInt32,
            ),
            idx_lookup: PlHashMap::with_capacity(capacity),
            categories: MutableUtf8Array::with_capacity(capacity),
        }
    }
}

impl ListBuilderTrait for ListLocalCategoricalChunkedBuilder {
    fn append_series(&mut self, s: &Series) -> PolarsResult<()> {
        let DataType::Categorical(Some(rev_map)) = s.dtype() else {
            polars_bail!(ComputeError: "expected categorical type")
        };
        let RevMapping::Local(cats_right) = &**rev_map else {
            polars_bail!(string_cache_mismatch)
        };
        let ca = s.categorical().unwrap();

        // Map the physical of the appended series to be compatible with the existing rev map
        let mut idx_mapping = PlHashMap::with_capacity(ca.len());
        for (idx, cat) in cats_right.values_iter().enumerate() {
            let hash_cat = self.idx_lookup.hasher().hash_one(cat);
            let len = self.idx_lookup.len();
            let entry = self.idx_lookup.raw_entry_mut().from_hash(hash_cat, |k| {
                // Safety: All keys in idx_lookup are in bounds
                unsafe { self.categories.value_unchecked(*k as usize) == cat }
            });

            match entry {
                // New Category
                RawEntryMut::Vacant(e) => {
                    idx_mapping.insert_unique_unchecked(idx as u32, len as u32);
                    e.insert(len as u32, ());
                    self.categories.push(Some(cat))
                },
                // Already Existing Category
                RawEntryMut::Occupied(e) => {
                    idx_mapping.insert_unique_unchecked(idx as u32, *e.key());
                },
            }
        }

        let new_physical = ca
            .physical()
            .into_iter()
            .map(move |i: Option<u32>| i.map(|i| *idx_mapping.get(&i).unwrap()));

        self.inner.append_iter(new_physical);

        Ok(())
    }

    fn append_null(&mut self) {
        self.inner.append_null()
    }

    fn finish(&mut self) -> ListChunked {
        let categories: Utf8Array<i64> = std::mem::take(&mut self.categories).into();
        let inner_dtype = DataType::Categorical(Some(Arc::new(RevMapping::Local(categories))));
        let mut ca = self.inner.finish();
        unsafe { ca.set_dtype(DataType::List(Box::new(inner_dtype))) }
        ca
    }
}

struct ListGlobalCategoricalChunkedBuilder {
    inner: ListPrimitiveChunkedBuilder<UInt32Type>,
    map_merger: GlobalRevMapMerger,
}

impl ListGlobalCategoricalChunkedBuilder {
    pub(super) fn new(
        name: &str,
        capacity: usize,
        values_capacity: usize,
        rev_map: Arc<RevMapping>,
    ) -> Self {
        let inner =
            ListPrimitiveChunkedBuilder::new(name, capacity, values_capacity, DataType::UInt32);
        Self {
            inner,
            map_merger: GlobalRevMapMerger::new(rev_map),
        }
    }
}

impl ListBuilderTrait for ListGlobalCategoricalChunkedBuilder {
    fn append_series(&mut self, s: &Series) -> PolarsResult<()> {
        let DataType::Categorical(Some(rev_map)) = s.dtype() else {
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
        let inner_dtype = DataType::Categorical(Some(rev_map));
        let mut ca = self.inner.finish();
        unsafe { ca.set_dtype(DataType::List(Box::new(inner_dtype))) }
        ca
    }
}
