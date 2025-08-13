use std::marker::PhantomData;

use arrow::bitmap::BitmapBuilder;
use num_traits::Zero;

use crate::chunked_array::cast::CastOptions;
use crate::chunked_array::flags::StatisticsFlags;
use crate::chunked_array::ops::ChunkFullNull;
use crate::prelude::*;
use crate::series::IsSorted;
use crate::utils::handle_casting_failures;

pub type CategoricalChunked<T> = Logical<T, <T as PolarsCategoricalType>::PolarsPhysical>;
pub type Categorical8Chunked = CategoricalChunked<Categorical8Type>;
pub type Categorical16Chunked = CategoricalChunked<Categorical16Type>;
pub type Categorical32Chunked = CategoricalChunked<Categorical32Type>;

pub trait CategoricalPhysicalDtypeExt {
    fn dtype(&self) -> DataType;
}

impl CategoricalPhysicalDtypeExt for CategoricalPhysical {
    fn dtype(&self) -> DataType {
        match self {
            Self::U8 => DataType::UInt8,
            Self::U16 => DataType::UInt16,
            Self::U32 => DataType::UInt32,
        }
    }
}

impl<T: PolarsCategoricalType> CategoricalChunked<T> {
    pub fn is_enum(&self) -> bool {
        matches!(self.dtype(), DataType::Enum(_, _))
    }

    pub(crate) fn get_flags(&self) -> StatisticsFlags {
        // If we use lexical ordering then physical sortedness does not imply
        // our sortedness.
        let mut flags = self.phys.get_flags();
        if self.uses_lexical_ordering() {
            flags.set_sorted(IsSorted::Not);
        }
        flags
    }

    /// Set flags for the ChunkedArray.
    pub(crate) fn set_flags(&mut self, mut flags: StatisticsFlags) {
        // We should not set the sorted flag if we are sorting in lexical order.
        if self.uses_lexical_ordering() {
            flags.set_sorted(IsSorted::Not)
        }
        self.physical_mut().set_flags(flags)
    }

    /// Return whether or not the [`CategoricalChunked`] uses the lexical order
    /// of the string values when sorting.
    pub fn uses_lexical_ordering(&self) -> bool {
        !self.is_enum()
    }

    pub fn full_null_with_dtype(name: PlSmallStr, length: usize, dtype: DataType) -> Self {
        let phys =
            ChunkedArray::<<T as PolarsCategoricalType>::PolarsPhysical>::full_null(name, length);
        unsafe { Self::from_cats_and_dtype_unchecked(phys, dtype) }
    }

    /// Create a [`CategoricalChunked`] from a physical array and dtype.
    ///
    /// Checks that all the category ids are valid, mapping invalid ones to nulls.
    pub fn from_cats_and_dtype(
        mut cat_ids: ChunkedArray<T::PolarsPhysical>,
        dtype: DataType,
    ) -> Self {
        let (DataType::Enum(_, mapping) | DataType::Categorical(_, mapping)) = &dtype else {
            panic!("from_cats_and_dtype called on non-categorical type")
        };
        assert!(dtype.cat_physical().ok() == Some(T::physical()));

        unsafe {
            let mut validity = BitmapBuilder::new();
            for arr in cat_ids.downcast_iter_mut() {
                validity.reserve(arr.len());
                if arr.has_nulls() {
                    for opt_cat_id in arr.iter() {
                        if let Some(cat_id) = opt_cat_id {
                            validity.push_unchecked(mapping.cat_to_str(cat_id.as_cat()).is_some());
                        } else {
                            validity.push_unchecked(false);
                        }
                    }
                } else {
                    for cat_id in arr.values_iter() {
                        validity.push_unchecked(mapping.cat_to_str(cat_id.as_cat()).is_some());
                    }
                }

                if arr.null_count() != validity.unset_bits() {
                    arr.set_validity(core::mem::take(&mut validity).into_opt_validity());
                } else {
                    validity.clear();
                }
            }
        }

        Self {
            phys: cat_ids,
            dtype,
            _phantom: PhantomData,
        }
    }

    /// Create a [`CategoricalChunked`] from a physical array and dtype.
    ///
    /// # Safety
    /// It's not checked that the indices are in-bounds or that the dtype is correct.
    pub unsafe fn from_cats_and_dtype_unchecked(
        cat_ids: ChunkedArray<T::PolarsPhysical>,
        dtype: DataType,
    ) -> Self {
        debug_assert!(dtype.cat_physical().ok() == Some(T::physical()));

        Self {
            phys: cat_ids,
            dtype,
            _phantom: PhantomData,
        }
    }

    /// Get a reference to the mapping of categorical types to the string values.
    pub fn get_mapping(&self) -> &Arc<CategoricalMapping> {
        let (DataType::Categorical(_, mapping) | DataType::Enum(_, mapping)) = self.dtype() else {
            unreachable!()
        };
        mapping
    }

    /// Create an [`Iterator`] that iterates over the `&str` values of the [`CategoricalChunked`].
    pub fn iter_str(&self) -> impl PolarsIterator<Item = Option<&str>> {
        let mapping = self.get_mapping();
        self.phys
            .iter()
            .map(|cat| unsafe { Some(mapping.cat_to_str_unchecked(cat?.as_cat())) })
    }

    /// Converts from strings to this CategoricalChunked.
    ///
    /// If this dtype is an Enum any non-existing strings get mapped to null.
    pub fn from_str_iter<'a, I: IntoIterator<Item = Option<&'a str>>>(
        name: PlSmallStr,
        dtype: DataType,
        strings: I,
    ) -> PolarsResult<Self> {
        let strings = strings.into_iter();

        let hint = strings.size_hint().0;
        let mut cat_ids = Vec::with_capacity(hint);
        let mut validity = BitmapBuilder::with_capacity(hint);

        match &dtype {
            DataType::Categorical(cats, mapping) => {
                assert!(cats.physical() == T::physical());
                for opt_s in strings {
                    cat_ids.push(if let Some(s) = opt_s {
                        T::Native::from_cat(mapping.insert_cat(s)?)
                    } else {
                        T::Native::zero()
                    });
                    validity.push(opt_s.is_some());
                }
            },
            DataType::Enum(fcats, mapping) => {
                assert!(fcats.physical() == T::physical());
                for opt_s in strings {
                    cat_ids.push(if let Some(cat) = opt_s.and_then(|s| mapping.get_cat(s)) {
                        validity.push(true);
                        T::Native::from_cat(cat)
                    } else {
                        validity.push(false);
                        T::Native::zero()
                    });
                }
            },
            _ => panic!("from_strings_and_dtype_strict called on non-categorical type"),
        }

        let arr = <T::PolarsPhysical as PolarsDataType>::Array::from_vec(cat_ids)
            .with_validity(validity.into_opt_validity());
        let phys = ChunkedArray::<T::PolarsPhysical>::with_chunk(name, arr);
        Ok(unsafe { Self::from_cats_and_dtype_unchecked(phys, dtype) })
    }

    pub fn to_arrow(&self, compat_level: CompatLevel) -> DictionaryArray<T::Native> {
        let keys = self.physical().rechunk();
        let keys = keys.downcast_as_array();
        let values = self
            .get_mapping()
            .to_arrow(compat_level != CompatLevel::oldest());
        let values_dtype = Box::new(values.dtype().clone());
        let dtype =
            ArrowDataType::Dictionary(<T::Native as DictionaryKey>::KEY_TYPE, values_dtype, false);
        unsafe { DictionaryArray::try_new_unchecked(dtype, keys.clone(), values).unwrap() }
    }
}

impl<T: PolarsCategoricalType> LogicalType for CategoricalChunked<T> {
    fn dtype(&self) -> &DataType {
        &self.dtype
    }

    fn get_any_value(&self, i: usize) -> PolarsResult<AnyValue<'_>> {
        polars_ensure!(i < self.len(), oob = i, self.len());
        Ok(unsafe { self.get_any_value_unchecked(i) })
    }

    unsafe fn get_any_value_unchecked(&self, i: usize) -> AnyValue<'_> {
        match self.phys.get_unchecked(i) {
            Some(i) => match &self.dtype {
                DataType::Enum(_, mapping) => AnyValue::Enum(i.as_cat(), mapping),
                DataType::Categorical(_, mapping) => AnyValue::Categorical(i.as_cat(), mapping),
                _ => unreachable!(),
            },
            None => AnyValue::Null,
        }
    }

    fn cast_with_options(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Series> {
        if &self.dtype == dtype {
            return Ok(self.clone().into_series());
        }

        match dtype {
            DataType::String => {
                let mapping = self.get_mapping();

                // TODO @ cat-rework:, if len >= mapping.upper_bound(), cast categories to ViewArray, then construct array of Views.

                let mut builder = StringChunkedBuilder::new(self.phys.name().clone(), self.len());
                let to_str = |cat_id: CatSize| unsafe { mapping.cat_to_str_unchecked(cat_id) };
                if !self.phys.has_nulls() {
                    for cat_id in self.phys.into_no_null_iter() {
                        builder.append_value(to_str(cat_id.as_cat()));
                    }
                } else {
                    for opt_cat_id in self.phys.into_iter() {
                        let opt_cat_id: Option<_> = opt_cat_id;
                        builder.append_option(opt_cat_id.map(|c| to_str(c.as_cat())));
                    }
                }

                let ca = builder.finish();
                Ok(ca.into_series())
            },

            DataType::Enum(fcats, _mapping) => {
                // TODO @ cat-rework: if len >= self.mapping().upper_bound(), remap categories then index into array.
                let ret = with_match_categorical_physical_type!(fcats.physical(), |$C| {
                    CategoricalChunked::<$C>::from_str_iter(
                        self.name().clone(),
                        dtype.clone(),
                        self.iter_str()
                    )?.into_series()
                });

                if options.is_strict() && self.null_count() != ret.null_count() {
                    handle_casting_failures(&self.clone().into_series(), &ret)?;
                }

                Ok(ret)
            },

            DataType::Categorical(cats, _mapping) => {
                // TODO @ cat-rework: if len >= self.mapping().upper_bound(), remap categories then index into array.
                Ok(
                    with_match_categorical_physical_type!(cats.physical(), |$C| {
                        CategoricalChunked::<$C>::from_str_iter(
                            self.name().clone(),
                            dtype.clone(),
                            self.iter_str()
                        )?.into_series()
                    }),
                )
            },

            // LEGACY
            // TODO @ cat-rework: remove after exposing to/from physical functions.
            dt if dt.is_integer() => self.phys.clone().cast_with_options(dtype, options),

            _ => polars_bail!(ComputeError: "cannot cast categorical types to {dtype:?}"),
        }
    }
}
