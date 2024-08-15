mod builder;
mod from;
mod merge;
mod ops;
pub mod revmap;
pub mod string_cache;

use bitflags::bitflags;
pub use builder::*;
pub use merge::*;
use polars_utils::itertools::Itertools;
use polars_utils::sync::SyncPtr;
pub use revmap::*;

use super::*;
use crate::chunked_array::cast::CastOptions;
use crate::chunked_array::metadata::MetadataFlags;
use crate::prelude::*;
use crate::series::IsSorted;
use crate::using_string_cache;

bitflags! {
    #[derive(Default, Clone)]
    struct BitSettings: u8 {
        const ORIGINAL = 0x01;
    }
}

#[derive(Clone)]
pub struct CategoricalChunked {
    physical: Logical<CategoricalType, UInt32Type>,
    /// 1st bit: original local categorical
    ///             meaning that n_unique is the same as the cat map length
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

    pub fn is_enum(&self) -> bool {
        matches!(self.dtype(), DataType::Enum(_, _))
    }

    /// Convert a categorical column to its local representation.
    pub fn to_local(&self) -> Self {
        let rev_map = self.get_rev_map();
        let (physical_map, categories) = match rev_map.as_ref() {
            RevMapping::Global(m, c, _) => (m, c),
            RevMapping::Local(_, _) if !self.is_enum() => return self.clone(),
            RevMapping::Local(_, _) => {
                // Change dtype from Enum to Categorical
                let mut local = self.clone();
                local.physical.2 = Some(DataType::Categorical(
                    Some(rev_map.clone()),
                    self.get_ordering(),
                ));
                return local;
            },
        };

        let local_rev_map = RevMapping::build_local(categories.clone());
        // TODO: A fast path can possibly be implemented here:
        // if all physical map keys are equal to their values,
        // we can skip the apply and only update the rev_map
        let local_ca = self
            .physical()
            .apply(|opt_v| opt_v.map(|v| *physical_map.get(&v).unwrap()));

        let mut out = unsafe {
            Self::from_cats_and_rev_map_unchecked(
                local_ca,
                local_rev_map.into(),
                false,
                self.get_ordering(),
            )
        };
        out.set_fast_unique(self._can_fast_unique());

        out
    }

    pub fn to_global(&self) -> PolarsResult<Self> {
        polars_ensure!(using_string_cache(), string_cache_mismatch);
        // Fast path
        let categories = match &**self.get_rev_map() {
            RevMapping::Global(_, _, _) => return Ok(self.clone()),
            RevMapping::Local(categories, _) => categories,
        };

        // SAFETY: keys and values are in bounds
        unsafe {
            Ok(CategoricalChunked::from_keys_and_values_global(
                self.name(),
                self.physical(),
                self.len(),
                categories,
                self.get_ordering(),
            ))
        }
    }

    // Convert to fixed enum. Values not in categories are mapped to None.
    pub fn to_enum(&self, categories: &Utf8ViewArray, hash: u128) -> Self {
        // Fast paths
        match self.get_rev_map().as_ref() {
            RevMapping::Local(_, cur_hash) if hash == *cur_hash => {
                return unsafe {
                    CategoricalChunked::from_cats_and_rev_map_unchecked(
                        self.physical().clone(),
                        self.get_rev_map().clone(),
                        true,
                        self.get_ordering(),
                    )
                };
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
            .map(|opt_v: Option<u32>| opt_v.and_then(|v| idx_map.get(&v).copied()))
            .collect();

        // SAFETY: we created the physical from the enum categories
        unsafe {
            CategoricalChunked::from_cats_and_rev_map_unchecked(
                new_phys,
                Arc::new(RevMapping::Local(categories.clone(), hash)),
                true,
                self.get_ordering(),
            )
        }
    }

    pub(crate) fn get_flags(&self) -> MetadataFlags {
        self.physical().get_flags()
    }

    /// Set flags for the Chunked Array
    pub(crate) fn set_flags(&mut self, mut flags: MetadataFlags) {
        // We should not set the sorted flag if we are sorting in lexical order
        if self.uses_lexical_ordering() {
            flags.set_sorted_flag(IsSorted::Not)
        }
        self.physical_mut().set_flags(flags)
    }

    /// Return whether or not the [`CategoricalChunked`] uses the lexical order
    /// of the string values when sorting.
    pub fn uses_lexical_ordering(&self) -> bool {
        self.get_ordering() == CategoricalOrdering::Lexical
    }

    pub(crate) fn get_ordering(&self) -> CategoricalOrdering {
        if let DataType::Categorical(_, ordering) | DataType::Enum(_, ordering) =
            &self.physical.2.as_ref().unwrap()
        {
            *ordering
        } else {
            panic!("implementation error")
        }
    }

    /// Create a [`CategoricalChunked`] from an array of `idx` and an existing [`RevMapping`]:  `rev_map`.
    ///
    /// # Safety
    /// Invariant in `v < rev_map.len() for v in idx` must hold.
    pub unsafe fn from_cats_and_rev_map_unchecked(
        idx: UInt32Chunked,
        rev_map: Arc<RevMapping>,
        is_enum: bool,
        ordering: CategoricalOrdering,
    ) -> Self {
        let mut logical = Logical::<UInt32Type, _>::new_logical::<CategoricalType>(idx);
        if is_enum {
            logical.2 = Some(DataType::Enum(Some(rev_map), ordering));
        } else {
            logical.2 = Some(DataType::Categorical(Some(rev_map), ordering));
        }
        Self {
            physical: logical,
            bit_settings: Default::default(),
        }
    }

    pub(crate) fn set_ordering(
        mut self,
        ordering: CategoricalOrdering,
        keep_fast_unique: bool,
    ) -> Self {
        self.physical.2 = match self.dtype() {
            DataType::Enum(_, _) => {
                Some(DataType::Enum(Some(self.get_rev_map().clone()), ordering))
            },
            DataType::Categorical(_, _) => Some(DataType::Categorical(
                Some(self.get_rev_map().clone()),
                ordering,
            )),
            _ => panic!("implementation error"),
        };

        if !keep_fast_unique {
            self.set_fast_unique(false)
        }
        self
    }

    /// # Safety
    /// The existing index values must be in bounds of the new [`RevMapping`].
    pub(crate) unsafe fn set_rev_map(&mut self, rev_map: Arc<RevMapping>, keep_fast_unique: bool) {
        self.physical.2 = match self.dtype() {
            DataType::Enum(_, _) => Some(DataType::Enum(Some(rev_map), self.get_ordering())),
            DataType::Categorical(_, _) => {
                Some(DataType::Categorical(Some(rev_map), self.get_ordering()))
            },
            _ => panic!("implementation error"),
        };

        if !keep_fast_unique {
            self.set_fast_unique(false)
        }
    }

    /// True if all categories are represented in this array. When this is the case, the unique
    /// values of the array are the categories.
    pub fn _can_fast_unique(&self) -> bool {
        self.bit_settings.contains(BitSettings::ORIGINAL)
            && self.physical.chunks.len() == 1
            && self.null_count() == 0
    }

    pub(crate) fn set_fast_unique(&mut self, toggle: bool) {
        if toggle {
            self.bit_settings.insert(BitSettings::ORIGINAL);
        } else {
            self.bit_settings.remove(BitSettings::ORIGINAL);
        }
    }

    pub(crate) fn with_fast_unique(mut self, toggle: bool) -> Self {
        self.set_fast_unique(toggle);
        self
    }

    pub fn _with_fast_unique(self, toggle: bool) -> Self {
        self.with_fast_unique(toggle)
    }

    /// Get a reference to the mapping of categorical types to the string values.
    pub fn get_rev_map(&self) -> &Arc<RevMapping> {
        if let DataType::Categorical(Some(rev_map), _) | DataType::Enum(Some(rev_map), _) =
            &self.physical.2.as_ref().unwrap()
        {
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
            Some(i) => match self.dtype() {
                DataType::Enum(_, _) => AnyValue::Enum(i, self.get_rev_map(), SyncPtr::new_null()),
                DataType::Categorical(_, _) => {
                    AnyValue::Categorical(i, self.get_rev_map(), SyncPtr::new_null())
                },
                _ => unimplemented!(),
            },
            None => AnyValue::Null,
        }
    }

    fn cast_with_options(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Series> {
        match dtype {
            DataType::String => {
                let mapping = &**self.get_rev_map();

                let mut builder = StringChunkedBuilder::new(self.physical.name(), self.len());

                let f = |idx: u32| mapping.get(idx);

                if !self.physical.has_nulls() {
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
            DataType::Enum(Some(rev_map), ordering) => {
                let RevMapping::Local(categories, hash) = &**rev_map else {
                    polars_bail!(ComputeError: "can not cast to enum with global mapping")
                };
                Ok(self
                    .to_enum(categories, *hash)
                    .set_ordering(*ordering, true)
                    .into_series()
                    .with_name(self.name()))
            },
            DataType::Enum(None, _) => {
                polars_bail!(ComputeError: "can not cast to enum without categories present")
            },
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(rev_map, ordering) => {
                // Casting from an Enum to a local or global
                if matches!(self.dtype(), DataType::Enum(_, _)) && rev_map.is_none() {
                    if using_string_cache() {
                        return Ok(self
                            .to_global()?
                            .set_ordering(*ordering, true)
                            .into_series());
                    } else {
                        return Ok(self.to_local().set_ordering(*ordering, true).into_series());
                    }
                }
                // Otherwise we do nothing
                Ok(self.clone().set_ordering(*ordering, true).into_series())
            },
            dt if dt.is_numeric() => {
                // Apply the cast to the categories and then index into the casted series
                let categories = StringChunked::with_chunk(
                    self.physical.name(),
                    self.get_rev_map().get_categories().clone(),
                );
                let casted_series = categories.cast_with_options(dtype, options)?;

                #[cfg(feature = "bigidx")]
                {
                    let s = self
                        .physical
                        .cast_with_options(&DataType::UInt64, options)?;
                    Ok(unsafe { casted_series.take_unchecked(s.u64()?) })
                }
                #[cfg(not(feature = "bigidx"))]
                {
                    // SAFETY: Invariant of categorical means indices are in bound
                    Ok(unsafe { casted_series.take_unchecked(&self.physical) })
                }
            },
            _ => self.physical.cast_with_options(dtype, options),
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
                // SAFETY:
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
        let ca = StringChunked::new("a", slice);
        let ca = ca.cast(&DataType::Categorical(None, Default::default()))?;
        let ca = ca.categorical().unwrap();

        let arr = ca.to_arrow(CompatLevel::newest(), false);
        let s = Series::try_from(("foo", arr))?;
        assert!(matches!(s.dtype(), &DataType::Categorical(_, _)));
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
            .cast(&DataType::Categorical(None, Default::default()))
            .unwrap();
        let s2 = Series::new("2", vec!["a", "x", "y"])
            .cast(&DataType::Categorical(None, Default::default()))
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
            .cast(&DataType::Categorical(None, Default::default()))
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
        let s = Series::new("a", vec!["a", "b", "c"])
            .cast(&DataType::Categorical(None, Default::default()))?;

        assert_eq!(
            s.field().into_owned(),
            Field::new("a", DataType::Categorical(None, Default::default()))
        );
        assert!(matches!(
            s.get(0)?,
            AnyValue::Categorical(0, RevMapping::Local(_, _), _)
        ));

        let groups = s.group_tuples(false, true);
        let aggregated = unsafe { s.agg_list(&groups?) };
        match aggregated.get(0)? {
            AnyValue::List(s) => {
                assert!(matches!(s.dtype(), DataType::Categorical(_, _)));
                let str_s = s.cast(&DataType::String).unwrap();
                assert_eq!(str_s.get(0)?, AnyValue::String("a"));
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
