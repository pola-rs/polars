use hashbrown::hash_map::RawEntryMut;
use polars_parquet::read::{ColumnChunkMetaData, RowGroupMetaData};
use polars_utils::aliases::{PlHashMap, PlHashSet};
use polars_utils::idx_vec::UnitVec;
use polars_utils::unitvec;

pub(super) struct ColumnToColumnChunkMD<'a> {
    partitions: PlHashMap<String, UnitVec<usize>>,
    pub metadata: &'a RowGroupMetaData,
}

impl<'a> ColumnToColumnChunkMD<'a> {
    pub(super) fn new(metadata: &'a RowGroupMetaData) -> Self {
        Self {
            partitions: Default::default(),
            metadata,
        }
    }

    pub(super) fn set_partitions(&mut self, field_names: Option<&PlHashSet<&str>>) {
        for (i, ccmd) in self.metadata.columns().iter().enumerate() {
            let name = &ccmd.descriptor().path_in_schema[0];
            if field_names
                .map(|field_names| field_names.contains(name.as_str()))
                .unwrap_or(true)
            {
                let entry = self.partitions.raw_entry_mut().from_key(name.as_str());

                match entry {
                    RawEntryMut::Vacant(slot) => {
                        slot.insert(name.to_string(), unitvec![i]);
                    },
                    RawEntryMut::Occupied(mut slot) => {
                        slot.get_mut().push(i);
                    },
                };
            }
        }
    }

    pub(super) fn get_partitions(&self, name: &str) -> UnitVec<&ColumnChunkMetaData> {
        debug_assert!(
            !self.partitions.is_empty(),
            "fields should be partitioned first"
        );
        let columns = self.metadata.columns();
        self.partitions
            .get(name)
            .map(|idx| idx.iter().map(|i| &columns[*i]).collect::<UnitVec<_>>())
            .unwrap_or_default()
    }
}
