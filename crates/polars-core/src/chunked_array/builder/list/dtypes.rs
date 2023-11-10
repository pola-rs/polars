use super::*;

pub(super) enum DtypeMerger {
    #[cfg(feature = "dtype-categorical")]
    Categorical(GlobalRevMapMerger),
    Other(Option<DataType>),
}

impl Default for DtypeMerger {
    fn default() -> Self {
        DtypeMerger::Other(None)
    }
}

impl DtypeMerger {
    pub(super) fn new(dtype: Option<DataType>) -> Self {
        match dtype {
            #[cfg(feature = "dtype-categorical")]
            Some(DataType::Categorical(Some(rev_map))) if rev_map.is_global() => {
                DtypeMerger::Categorical(GlobalRevMapMerger::new(rev_map))
            },
            _ => DtypeMerger::Other(dtype),
        }
    }

    #[inline]
    pub(super) fn update(&mut self, dtype: &DataType) -> PolarsResult<()> {
        match self {
            #[cfg(feature = "dtype-categorical")]
            DtypeMerger::Categorical(merger) => {
                let DataType::Categorical(Some(rev_map)) = dtype else {
                    polars_bail!(ComputeError: "expected categorical rev-map")
                };
                polars_ensure!(rev_map.is_global(), string_cache_mismatch);
                return merger.merge_map(rev_map);
            },
            DtypeMerger::Other(Some(set_dtype)) => {
                polars_ensure!(set_dtype == dtype, ComputeError: "dtypes don't match, got {}, expected: {}", dtype, set_dtype)
            },
            _ => {},
        }
        Ok(())
    }

    pub(super) fn materialize(self) -> Option<DataType> {
        match self {
            #[cfg(feature = "dtype-categorical")]
            DtypeMerger::Categorical(merger) => Some(DataType::Categorical(Some(merger.finish()))),
            DtypeMerger::Other(dtype) => dtype,
        }
    }
}
