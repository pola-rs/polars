mod ops;
mod series;

use super::*;
use crate::chunked_array::categorical::*;
use crate::prelude::DataType::UInt32;
use crate::prelude::*;

#[derive(Clone)]
pub struct CategoricalChunked {
    logical: Logical<CategoricalType, UInt32Type>,
    /// 1st bit: original local categorical
    ///             meaning that n_unique is the same as the cat map length
    bit_settings: u8,
}

impl CategoricalChunked {
    pub(crate) fn len(&self) -> usize {
        self.logical.len()
    }

    /// Get a reference to the logical array (the categories).
    pub(crate) fn rechunk(&self) -> Self {
        let mut out = self.clone();
        out.logical.0 = out.logical.0.rechunk();
        out
    }

    /// Get a reference to the logical array (the categories).
    pub(crate) fn logical(&self) -> &UInt32Chunked {
        &self.logical
    }

    /// Get a reference to the logical array (the categories).
    pub(crate) fn logical_mut(&mut self) -> &mut UInt32Chunked {
        &mut self.logical
    }

    /// Build a categorical from an original RevMap. That means that the number of categories in the `RevMapping == self.unique().len()`.
    pub(crate) fn from_chunks_original(
        name: &str,
        chunks: Vec<ArrayRef>,
        rev_map: RevMapping,
    ) -> Self {
        let ca = UInt32Chunked::from_chunks(name, chunks);
        let mut logical = Logical::<UInt32Type, _>::new_logical::<CategoricalType>(ca);
        logical.2 = Some(DataType::Categorical(Some(Arc::new(rev_map))));
        let bit_settings = 1u8;
        Self {
            logical,
            bit_settings,
        }
    }

    pub(crate) fn from_cats_and_rev_map(idx: UInt32Chunked, rev_map: Arc<RevMapping>) -> Self {
        let mut logical = Logical::<UInt32Type, _>::new_logical::<CategoricalType>(idx);
        logical.2 = Some(DataType::Categorical(Some(rev_map)));
        Self {
            logical,
            bit_settings: 0,
        }
    }

    pub(crate) fn set_rev_map(&mut self, rev_map: Arc<RevMapping>, keep_fast_unique: bool) {
        self.logical.2 = Some(DataType::Categorical(Some(rev_map)));
        if !keep_fast_unique {
            self.set_fast_unique(false)
        }
    }

    pub(crate) fn can_fast_unique(&self) -> bool {
        self.bit_settings & 1 << 0 != 0 && self.logical.chunks.len() == 1
    }

    pub(crate) fn set_fast_unique(&mut self, can: bool) {
        if can {
            self.bit_settings |= 1u8 << 0;
        } else {
            self.bit_settings &= !(1u8 << 0);
        }
    }

    /// Get a reference to the mapping of categorical types to the string values.
    pub fn get_rev_map(&self) -> &Arc<RevMapping> {
        if let DataType::Categorical(Some(rev_map)) = &self.logical.2.as_ref().unwrap() {
            rev_map
        } else {
            panic!("implementation error")
        }
    }

    /// Create an `[Iterator]` that iterates over the `&str` values of the `[CategoricalChunked]`.
    pub fn iter_str(&self) -> CatIter<'_> {
        let iter = self.logical().into_iter();
        CatIter {
            rev: self.get_rev_map(),
            iter,
        }
    }
}

impl LogicalType for CategoricalChunked {
    fn dtype(&self) -> &DataType {
        self.logical.2.as_ref().unwrap()
    }

    fn get_any_value(&self, i: usize) -> AnyValue<'_> {
        match self.logical.0.get(i) {
            Some(i) => AnyValue::Categorical(i, &self.get_rev_map()),
            None => AnyValue::Null,
        }
    }

    fn cast(&self, dtype: &DataType) -> Result<Series> {
        match dtype {
            DataType::Utf8 => {
                let mapping = &**self.get_rev_map();

                let mut builder =
                    Utf8ChunkedBuilder::new(self.logical.name(), self.len(), self.len() * 5);

                let f = |idx: u32| mapping.get(idx);

                if !self.logical.has_validity() {
                    self.logical
                        .into_no_null_iter()
                        .for_each(|idx| builder.append_value(f(idx)));
                } else {
                    self.logical.into_iter().for_each(|opt_idx| {
                        builder.append_option(opt_idx.map(f));
                    });
                }

                let ca = builder.finish();
                Ok(ca.into_series())
            }
            DataType::UInt32 => {
                let ca =
                    UInt32Chunked::from_chunks(self.logical.name(), self.logical.chunks.clone());
                Ok(ca.into_series())
            }
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_) => Ok(self.clone().into_series()),
            _ => self.logical.cast(dtype),
        }
    }
}

pub struct CatIter<'a> {
    rev: &'a RevMapping,
    iter: Box<dyn PolarsIterator<Item = Option<u32>> + 'a>,
}

unsafe impl<'a> TrustedLen for CatIter<'a> {}

impl<'a> Iterator for CatIter<'a> {
    type Item = Option<&'a str>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|item| {
            item.map(|idx| {
                // Safety:
                // all categories are in bound
                unsafe { self.rev.get_unchecked(idx) }
            })
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a> ExactSizeIterator for CatIter<'a> {}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{reset_string_cache, toggle_string_cache, SINGLE_LOCK};
    use std::convert::TryFrom;

    #[test]
    fn test_categorical_round_trip() -> Result<()> {
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
        let ca = ca.cast(&DataType::Categorical)?;
        let ca = ca.categorical().unwrap();

        let arr: DictionaryArray<u32> = (ca).into();
        let s = Series::try_from(("foo", Arc::new(arr) as ArrayRef))?;
        assert_eq!(s.dtype(), &DataType::Categorical);
        assert_eq!(s.null_count(), 1);
        assert_eq!(s.len(), 6);

        Ok(())
    }

    #[test]
    fn test_append_categorical() {
        let _lock = SINGLE_LOCK.lock();
        reset_string_cache();
        toggle_string_cache(true);

        let mut s1 = Series::new("1", vec!["a", "b", "c"])
            .cast(&DataType::Categorical)
            .unwrap();
        let s2 = Series::new("2", vec!["a", "x", "y"])
            .cast(&DataType::Categorical)
            .unwrap();
        let appended = s1.append(&s2).unwrap();
        assert_eq!(appended.str_value(0), "a");
        assert_eq!(appended.str_value(1), "b");
        assert_eq!(appended.str_value(4), "x");
        assert_eq!(appended.str_value(5), "y");
    }

    #[test]
    fn test_fast_unique() {
        let s = Series::new("1", vec!["a", "b", "c"])
            .cast(&DataType::Categorical)
            .unwrap();

        assert_eq!(s.n_unique().unwrap(), 3);
        // make sure that it does not take the fast path after take/ slice
        let out = s.take(&([1, 2].as_ref()).into()).unwrap();
        assert_eq!(out.n_unique().unwrap(), 2);
        let out = s.slice(1, 2);
        assert_eq!(out.n_unique().unwrap(), 2);
    }
}
