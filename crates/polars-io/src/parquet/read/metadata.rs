use hashbrown::hash_map::RawEntryMut;
use polars_parquet::read::{ColumnChunkMetaData, RowGroupMetaData};
use polars_utils::aliases::{PlHashMap, PlHashSet};
use polars_utils::idx_vec::UnitVec;
use polars_utils::unitvec;

/// This is a utility struct that Partitions the `ColumnChunkMetaData` by `field.name == descriptor.path_in_schema[0]`
/// This is required to fix quadratic behavior in wide parquet files. See #18319.
pub struct PartitionedColumnChunkMD<'a> {
    partitions: Option<PlHashMap<String, UnitVec<usize>>>,
    metadata: &'a RowGroupMetaData,
}

impl<'a> PartitionedColumnChunkMD<'a> {
    pub fn new(metadata: &'a RowGroupMetaData) -> Self {
        Self {
            partitions: Default::default(),
            metadata,
        }
    }

    pub(super) fn num_rows(&self) -> usize {
        self.metadata.num_rows()
    }

    pub fn set_partitions(&mut self, field_names: Option<&PlHashSet<&str>>) {
        let mut partitions = PlHashMap::default();
        for (i, ccmd) in self.metadata.columns().iter().enumerate() {
            let name = &ccmd.descriptor().path_in_schema[0];
            if field_names
                .map(|field_names| field_names.contains(name.as_str()))
                .unwrap_or(true)
            {
                let entry = partitions.raw_entry_mut().from_key(name.as_str());

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
        self.partitions = Some(partitions)
    }

    pub fn get_partitions(&self, name: &str) -> Option<UnitVec<&ColumnChunkMetaData>> {
        let columns = self.metadata.columns();
        self.partitions
            .as_ref()
            .expect("fields should be partitioned first")
            .get(name)
            .map(|idx| idx.iter().map(|i| &columns[*i]).collect::<UnitVec<_>>())
    }
}
