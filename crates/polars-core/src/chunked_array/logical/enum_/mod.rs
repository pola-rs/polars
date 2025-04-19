use std::sync::Arc;

use arrow::array::UInt32Vec;
use arrow::bitmap::MutableBitmap;
use polars_error::{PolarsResult, polars_bail, polars_err};
use polars_utils::aliases::{InitHashMaps, PlHashMap};
use polars_utils::pl_str::PlSmallStr;

use super::{CategoricalChunked, CategoricalOrdering, DataType, Field, RevMapping, UInt32Chunked};

pub struct EnumChunkedBuilder {
    name: PlSmallStr,
    enum_builder: UInt32Vec,

    rev: Arc<RevMapping>,
    ordering: CategoricalOrdering,
    seen: MutableBitmap,

    // Mapping to amortize the costs of lookups.
    mapping: PlHashMap<PlSmallStr, u32>,
    strict: bool,
}

impl EnumChunkedBuilder {
    pub fn new(
        name: PlSmallStr,
        capacity: usize,
        rev: Arc<RevMapping>,
        ordering: CategoricalOrdering,
        strict: bool,
    ) -> Self {
        let seen = MutableBitmap::from_len_zeroed(rev.len());

        Self {
            name,
            enum_builder: UInt32Vec::with_capacity(capacity),

            rev,
            ordering,
            seen,

            mapping: PlHashMap::new(),
            strict,
        }
    }

    pub fn append_str(&mut self, v: &str) -> PolarsResult<&mut Self> {
        match self.mapping.get(v) {
            Some(v) => self.enum_builder.push(Some(*v)),
            None => {
                let Some(iv) = self.rev.find(v) else {
                    if self.strict {
                        polars_bail!(InvalidOperation: "cannot append '{v}' to enum without that variant");
                    } else {
                        self.enum_builder.push(None);
                        return Ok(self);
                    }
                };
                self.seen.set(iv as usize, true);
                self.mapping.insert(v.into(), iv);
                self.enum_builder.push(Some(iv));
            },
        }

        Ok(self)
    }

    pub fn append_null(&mut self) -> &mut Self {
        self.enum_builder.push(None);
        self
    }

    pub fn append_enum(&mut self, v: u32, rev: &RevMapping) -> PolarsResult<&mut Self> {
        if !self.rev.same_src(rev) {
            if self.strict {
                return Err(polars_err!(ComputeError: "incompatible enum types"));
            } else {
                self.enum_builder.push(None);
            }
        } else {
            self.seen.set(v as usize, true);
            self.enum_builder.push(Some(v));
        }

        Ok(self)
    }

    pub fn finish(self) -> CategoricalChunked {
        let arr = self.enum_builder.freeze();
        let null_count = arr.validity().map_or(0, |a| a.unset_bits());
        let length = arr.len();
        let ca = unsafe {
            UInt32Chunked::new_with_dims(
                Arc::new(Field::new(self.name, DataType::UInt32)),
                vec![Box::new(arr)],
                length,
                null_count,
            )
        };
        // Fast Unique <=> unique(rev) == unique(ca)
        let fast_unique = !ca.has_nulls() && self.seen.unset_bits() == 0;

        // SAFETY: keys and values are in bounds
        unsafe {
            CategoricalChunked::from_cats_and_rev_map_unchecked(ca, self.rev, true, self.ordering)
                .with_fast_unique(fast_unique)
        }
    }
}
