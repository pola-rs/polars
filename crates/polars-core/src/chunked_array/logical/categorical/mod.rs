mod builder;
mod from;
mod merge;
mod ops;
pub mod string_cache;

use bitflags::bitflags;
pub use builder::*;
pub use merge::*;
use polars_utils::iter::EnumerateIdxTrait;
use polars_utils::sync::SyncPtr;

use super::*;
use crate::chunked_array::Settings;
use crate::prelude::*;
use crate::using_string_cache;

bitflags! {
    #[derive(Default, Clone)]
    struct BitSettings: u8 {
        const ORIGINAL = 0x01;
        const LEXICAL_ORDERING = 0x02;
    }
}

#[derive(Clone)]
pub struct CategoricalChunked {
    physical: Logical<CategoricalType, UInt32Type>,
    /// 1st bit: original local categorical
    ///             meaning that n_unique is the same as the cat map length
    /// 2nd bit: use lexical sorting
    bit_settings: BitSettings,
}

impl CategoricalChunked {
    pub(crate) fn field(&self) -> Field {
        let name = self.physical().name();
        Field::new(name, self.dtype().clone())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.physical.len()
    }

    #[inline]
    pub fn null_count(&self) -> usize {
        self.physical.null_count()
    }

    pub fn name(&self) -> &str {
        self.physical.name()
    }

    // TODO: Rename this
    /// Get a reference to the physical array (the categories).
    pub fn physical(&self) -> &UInt32Chunked {
        &self.physical
    }

    /// Get a mutable reference to the physical array (the categories).
    pub(crate) fn physical_mut(&mut self) -> &mut UInt32Chunked {
        &mut self.physical
    }

    /// Convert a categorical column to its local representation.
    pub fn to_local(&self) -> Self {
        let rev_map = self.get_rev_map();
        let (physical_map, categories) = match rev_map.as_ref() {
            RevMapping::Global(m, c, _) => (m, c),
            RevMapping::Local(_, _) => return self.clone(),
            RevMapping::Enum(a, h) => unsafe {
                return Self::from_cats_and_rev_map_unchecked(
                    self.physical().clone(),
                    RevMapping::Local(a.clone(), *h).into(),
                );
            },
        };

        let local_rev_map = RevMapping::build_local(categories.clone());
        // TODO: A fast path can possibly be implemented here:
        // if all physical map keys are equal to their values,
        // we can skip the apply and only update the rev_map
        let local_ca = self
            .physical()
            .apply(|opt_v| opt_v.map(|v| *physical_map.get(&v).unwrap()));

        let mut out =
            unsafe { Self::from_cats_and_rev_map_unchecked(local_ca, local_rev_map.into()) };
        out.set_fast_unique(self.can_fast_unique());
        out.set_lexical_ordering(self.uses_lexical_ordering());

        out
    }

    pub fn to_global(&self) -> PolarsResult<Self> {
        polars_ensure!(using_string_cache(), string_cache_mismatch);
        // Fast path
        let categories = match &**self.get_rev_map() {
            RevMapping::Global(_, _, _) => return Ok(self.clone()),
            RevMapping::Local(categories, _) => categories,
            RevMapping::Enum(categories, _) => categories,
        };
        let physical = self.physical();

        let mut builder = CategoricalChunkedBuilder::new(self.name(), physical.len());
        builder.global_map_from_local(physical, categories.clone());
        Ok(builder.finish())
    }

    pub fn to_enum(&self, categories: &Utf8Array<i64>, hash: u128) -> PolarsResult<Self> {
        // Fast paths
        match self.get_rev_map().as_ref() {
            RevMapping::Enum(_, cur_hash) if hash == *cur_hash => return Ok(self.clone()),
            RevMapping::Local(_, cur_hash) if hash == *cur_hash => {
                return unsafe {
                    Ok(CategoricalChunked::from_cats_and_rev_map_unchecked(
                        self.physical().clone(),
                        RevMapping::Enum(categories.clone(), hash).into(),
                    ))
                }
            },
            _ => (),
        };
        // Make a mapping from old idx to new idx
        let old_rev_map = self.get_rev_map();
        #[allow(clippy::unnecessary_cast)]
        let idx_map: PlHashMap<u32, u32> = categories
            .values_iter()
            .enumerate_idx()
            .filter_map(|(new_idx, s)| old_rev_map.find(s).map(|old_idx| (old_idx, new_idx as u32)))
            .collect();

        // Loop over the physicals and try get new idx
        let new_phys: UInt32Chunked = self
            .physical()
            .into_iter()
            .map(|opt_v: Option<u32>| opt_v.map(|v| idx_map.get(&v).copied().ok_or_else(|| polars_err!(OutOfBounds: "value {} is not present in Enum {:?}",old_rev_map.get(v),&categories))).transpose())
            .collect::<PolarsResult<_>>()?;

        Ok(
            // Safety: we created the physical from the enum categories
            unsafe {
                CategoricalChunked::from_cats_and_rev_map_unchecked(
                    new_phys,
                    RevMapping::Enum(categories.clone(), hash).into(),
                )
            },
        )
    }

    pub(crate) fn get_flags(&self) -> Settings {
        self.physical().get_flags()
    }

    /// Set flags for the Chunked Array
    pub(crate) fn set_flags(&mut self, flags: Settings) {
        self.physical_mut().set_flags(flags)
    }

    /// Build a categorical from an original RevMap. That means that the number of categories in the `RevMapping == self.unique().len()`.
    pub(crate) fn from_chunks_original(
        name: &str,
        chunk: PrimitiveArray<u32>,
        rev_map: RevMapping,
    ) -> Self {
        let ca = ChunkedArray::with_chunk(name, chunk);
        let mut logical = Logical::<UInt32Type, _>::new_logical::<CategoricalType>(ca);
        logical.2 = Some(DataType::Categorical(Some(Arc::new(rev_map))));

        let mut bit_settings = BitSettings::default();
        bit_settings.insert(BitSettings::ORIGINAL);
        Self {
            physical: logical,
            bit_settings,
        }
    }

    pub fn set_lexical_ordering(&mut self, toggle: bool) {
        if toggle {
            self.bit_settings.insert(BitSettings::LEXICAL_ORDERING);
        } else {
            self.bit_settings.remove(BitSettings::LEXICAL_ORDERING);
        }
    }

    /// Return whether or not the [`CategoricalChunked`] uses the lexical order
    /// of the string values when sorting.
    pub fn uses_lexical_ordering(&self) -> bool {
        self.bit_settings.contains(BitSettings::LEXICAL_ORDERING)
    }

    /// Create a [`CategoricalChunked`] from an array of `idx` and an existing [`RevMapping`]:  `rev_map`.
    ///
    /// # Safety
    /// Invariant in `v < rev_map.len() for v in idx` must hold.
    pub unsafe fn from_cats_and_rev_map_unchecked(
        idx: UInt32Chunked,
        rev_map: Arc<RevMapping>,
    ) -> Self {
        let mut logical = Logical::<UInt32Type, _>::new_logical::<CategoricalType>(idx);
        logical.2 = Some(DataType::Categorical(Some(rev_map)));
        Self {
            physical: logical,
            bit_settings: Default::default(),
        }
    }

    /// # Safety
    /// The existing index values must be in bounds of the new [`RevMapping`].
    pub(crate) unsafe fn set_rev_map(&mut self, rev_map: Arc<RevMapping>, keep_fast_unique: bool) {
        self.physical.2 = Some(DataType::Categorical(Some(rev_map)));
        if !keep_fast_unique {
            self.set_fast_unique(false)
        }
    }

    pub(crate) fn can_fast_unique(&self) -> bool {
        self.bit_settings.contains(BitSettings::ORIGINAL) && self.physical.chunks.len() == 1
    }

    pub(crate) fn set_fast_unique(&mut self, toggle: bool) {
        if toggle {
            self.bit_settings.insert(BitSettings::ORIGINAL);
        } else {
            self.bit_settings.remove(BitSettings::ORIGINAL);
        }
    }

    /// Get a reference to the mapping of categorical types to the string values.
    pub fn get_rev_map(&self) -> &Arc<RevMapping> {
        if let DataType::Categorical(Some(rev_map)) = &self.physical.2.as_ref().unwrap() {
            rev_map
        } else {
            panic!("implementation error")
        }
    }

    /// Create an `[Iterator]` that iterates over the `&str` values of the `[CategoricalChunked]`.
    pub fn iter_str(&self) -> CatIter<'_> {
        let iter = self.physical().into_iter();
        CatIter {
            rev: self.get_rev_map(),
            iter,
        }
    }
}

impl LogicalType for CategoricalChunked {
    fn dtype(&self) -> &DataType {
        self.physical.2.as_ref().unwrap()
    }

    fn get_any_value(&self, i: usize) -> PolarsResult<AnyValue<'_>> {
        polars_ensure!(i < self.len(), oob = i, self.len());
        Ok(unsafe { self.get_any_value_unchecked(i) })
    }

    unsafe fn get_any_value_unchecked(&self, i: usize) -> AnyValue<'_> {
        match self.physical.0.get_unchecked(i) {
            Some(i) => AnyValue::Categorical(i, self.get_rev_map(), SyncPtr::new_null()),
            None => AnyValue::Null,
        }
    }

    fn cast(&self, dtype: &DataType) -> PolarsResult<Series> {
        match dtype {
            DataType::Utf8 => {
                let mapping = &**self.get_rev_map();

                let mut builder =
                    Utf8ChunkedBuilder::new(self.physical.name(), self.len(), self.len() * 5);

                let f = |idx: u32| mapping.get(idx);

                if !self.physical.has_validity() {
                    self.physical
                        .into_no_null_iter()
                        .for_each(|idx| builder.append_value(f(idx)));
                } else {
                    self.physical.into_iter().for_each(|opt_idx| {
                        builder.append_option(opt_idx.map(f));
                    });
                }

                let ca = builder.finish();
                Ok(ca.into_series())
            },
            DataType::UInt32 => {
                let ca = unsafe {
                    UInt32Chunked::from_chunks(self.physical.name(), self.physical.chunks.clone())
                };
                Ok(ca.into_series())
            },
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(rev_map) => {
                // Casting to a Enum
                if let Some(rev_map) = rev_map {
                    if let RevMapping::Enum(categories, hash) = &**rev_map {
                        return Ok(self.to_enum(categories, *hash)?.into_series());
                    }
                }

                // Casting from an Enum to a local or global
                if matches!(&**self.get_rev_map(), RevMapping::Enum(_, _)) && rev_map.is_none() {
                    if using_string_cache() {
                        return Ok(self.to_global()?.into_series());
                    } else {
                        return Ok(self.to_local().into_series());
                    }
                }
                // Otherwise we do nothing
                Ok(self.clone().into_series())
            },
            _ => self.physical.cast(dtype),
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
    use std::convert::TryFrom;

    use super::*;
    use crate::{disable_string_cache, enable_string_cache, SINGLE_LOCK};

    #[test]
    fn test_categorical_round_trip() -> PolarsResult<()> {
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
        let ca = Utf8Chunked::new("a", slice);
        let ca = ca.cast(&DataType::Categorical(None))?;
        let ca = ca.categorical().unwrap();

        let arr: DictionaryArray<u32> = (ca).into();
        let s = Series::try_from(("foo", Box::new(arr) as ArrayRef))?;
        assert!(matches!(s.dtype(), &DataType::Categorical(_)));
        assert_eq!(s.null_count(), 1);
        assert_eq!(s.len(), 6);

        Ok(())
    }

    #[test]
    fn test_append_categorical() {
        let _lock = SINGLE_LOCK.lock();
        disable_string_cache();
        enable_string_cache();

        let mut s1 = Series::new("1", vec!["a", "b", "c"])
            .cast(&DataType::Categorical(None))
            .unwrap();
        let s2 = Series::new("2", vec!["a", "x", "y"])
            .cast(&DataType::Categorical(None))
            .unwrap();
        let appended = s1.append(&s2).unwrap();
        assert_eq!(appended.str_value(0).unwrap(), "a");
        assert_eq!(appended.str_value(1).unwrap(), "b");
        assert_eq!(appended.str_value(4).unwrap(), "x");
        assert_eq!(appended.str_value(5).unwrap(), "y");
    }

    #[test]
    fn test_fast_unique() {
        let _lock = SINGLE_LOCK.lock();
        let s = Series::new("1", vec!["a", "b", "c"])
            .cast(&DataType::Categorical(None))
            .unwrap();

        assert_eq!(s.n_unique().unwrap(), 3);
        // Make sure that it does not take the fast path after take/slice.
        let out = s.take(&IdxCa::new("", [1, 2])).unwrap();
        assert_eq!(out.n_unique().unwrap(), 2);
        let out = s.slice(1, 2);
        assert_eq!(out.n_unique().unwrap(), 2);
    }

    #[test]
    fn test_categorical_flow() -> PolarsResult<()> {
        let _lock = SINGLE_LOCK.lock();
        disable_string_cache();

        // tests several things that may lose the dtype information
        let s = Series::new("a", vec!["a", "b", "c"]).cast(&DataType::Categorical(None))?;

        assert_eq!(
            s.field().into_owned(),
            Field::new("a", DataType::Categorical(None))
        );
        assert!(matches!(
            s.get(0)?,
            AnyValue::Categorical(0, RevMapping::Local(_, _), _)
        ));

        let groups = s.group_tuples(false, true);
        let aggregated = unsafe { s.agg_list(&groups?) };
        match aggregated.get(0)? {
            AnyValue::List(s) => {
                assert!(matches!(s.dtype(), DataType::Categorical(_)));
                let str_s = s.cast(&DataType::Utf8).unwrap();
                assert_eq!(str_s.get(0)?, AnyValue::Utf8("a"));
                assert_eq!(s.len(), 1);
            },
            _ => panic!(),
        }
        let flat = aggregated.explode()?;
        let ca = flat.categorical().unwrap();
        let vals = ca.iter_str().map(|v| v.unwrap()).collect::<Vec<_>>();
        assert_eq!(vals, &["a", "b", "c"]);
        Ok(())
    }
}
