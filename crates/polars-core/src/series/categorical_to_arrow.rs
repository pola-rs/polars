use std::any::Any;

use arrow::array::builder::ArrayBuilder;
use arrow::types::NativeType;
use polars_compute::cast::cast_unchecked;

use crate::prelude::*;

pub struct CategoricalToArrowConverter {
    /// Converters keyed by the Arc address of `Arc<CategoricalMapping>`.
    pub converters: PlIndexMap<usize, CategoricalArrayToArrowConverter>,
    /// Persist the key remap to ensure consistent mapping across multiple calls.
    pub persist_remap: bool,
    /// Return only the keys array when going to arrow.
    pub output_keys_only: bool,
}

impl CategoricalToArrowConverter {
    pub fn array_to_arrow(
        &mut self,
        keys_arr: &dyn Array,
        dtype: &DataType,
        compat_level: CompatLevel,
    ) -> Box<dyn Array> {
        let (DataType::Categorical(_, mapping) | DataType::Enum(_, mapping)) = dtype else {
            unreachable!()
        };

        let key = Arc::as_ptr(mapping) as *const () as usize;
        let converter = self.converters.get_mut(&key).unwrap();

        with_match_categorical_physical_type!(dtype.cat_physical().unwrap(), |$C| {
            let keys_arr: &PrimitiveArray<<$C as PolarsCategoricalType>::Native> = keys_arr.as_any().downcast_ref().unwrap();

            converter.array_to_arrow(
                keys_arr,
                dtype,
                self.persist_remap,
                self.output_keys_only,
                compat_level
            )
        })
    }

    pub fn initialize(&mut self, dtype: &DataType) {
        use DataType::*;

        match dtype {
            Categorical(_categories, mapping) => {
                let key = Arc::as_ptr(mapping) as *const () as usize;

                if !self.converters.contains_key(&key) {
                    with_match_categorical_physical_type!(dtype.cat_physical().unwrap(), |$C| {
                        self.converters.insert(
                            key,
                            CategoricalArrayToArrowConverter::Categorical {
                                mapping: mapping.clone(),
                                key_remap: CategoricalKeyRemap::from(
                                    PlIndexSet::<<$C as PolarsCategoricalType>::Native>::with_capacity(
                                        mapping.num_cats_upper_bound()
                                    )
                                ),
                            },
                        );
                    })
                }
            },
            Enum(categories, mapping) => {
                let key = Arc::as_ptr(mapping) as *const () as usize;

                if !self.converters.contains_key(&key) {
                    self.converters.insert(
                        key,
                        CategoricalArrayToArrowConverter::Enum {
                            frozen: categories.clone(),
                            mapping: mapping.clone(),
                        },
                    );
                }
            },
            List(inner) => self.initialize(inner),
            #[cfg(feature = "dtype-array")]
            Array(inner, _width) => self.initialize(inner),
            #[cfg(feature = "dtype-struct")]
            Struct(fields) => {
                for field in fields {
                    self.initialize(field.dtype())
                }
            },
            _ => {
                debug_assert!(!dtype.is_nested())
            },
        }
    }
}

pub enum CategoricalArrayToArrowConverter {
    Categorical {
        mapping: Arc<CategoricalMapping>,
        key_remap: CategoricalKeyRemap,
    },
    /// Note, enum keys are not remapped.
    Enum {
        mapping: Arc<CategoricalMapping>,
        frozen: Arc<FrozenCategories>,
    },
}

impl CategoricalArrayToArrowConverter {
    fn get_categorical_mapping(&self) -> &CategoricalMapping {
        match self {
            Self::Categorical { mapping, .. } => mapping,
            Self::Enum { mapping, .. } => mapping,
        }
    }

    fn array_to_arrow<T>(
        &mut self,
        keys_arr: &PrimitiveArray<T>,
        dtype: &DataType,
        persist_remap: bool,
        output_keys_only: bool,
        compat_level: CompatLevel,
    ) -> Box<dyn Array>
    where
        T: DictionaryKey + NativeType + std::hash::Hash + Eq + TryFrom<usize> + Into<CatSize>,
    {
        let (DataType::Categorical(_, mapping) | DataType::Enum(_, mapping)) = dtype else {
            unreachable!()
        };

        let input_mapping_ptr = Arc::as_ptr(mapping);

        let keys_arr: PrimitiveArray<T> = match self {
            Self::Categorical { mapping, key_remap } => {
                // Note: Important for persisted remap correctness.
                assert_eq!(input_mapping_ptr, Arc::as_ptr(mapping));

                let key_remap: &mut PlIndexSet<T> = key_remap.as_any_mut().downcast_mut().unwrap();

                if !persist_remap {
                    key_remap.clear()
                }

                keys_arr
                    .iter()
                    .map(|x| {
                        x.map(|x: &T| {
                            let idx: usize = key_remap.insert_full(*x).0;
                            // Safety: Indexset of T cannot return an index exceeding T::MAX.
                            unsafe { T::try_from(idx).unwrap_unchecked() }
                        })
                    })
                    .collect()
            },
            Self::Enum { mapping, .. } => {
                assert_eq!(input_mapping_ptr, Arc::as_ptr(mapping));
                keys_arr.clone()
            },
        };

        if output_keys_only {
            return keys_arr.boxed();
        }

        let values = self.build_values_array(compat_level);

        let dictionary_dtype = ArrowDataType::Dictionary(
            <T as DictionaryKey>::KEY_TYPE,
            Box::new(values.dtype().clone()),
            false, // is_sorted
        );

        unsafe {
            DictionaryArray::<T>::try_new_unchecked(dictionary_dtype, keys_arr, values)
                .unwrap()
                .boxed()
        }
    }

    pub fn build_values_array(&self, compat_level: CompatLevel) -> Box<dyn Array> {
        match self {
            Self::Categorical {
                mapping: _,
                key_remap,
            } => match key_remap {
                CategoricalKeyRemap::U8(keys) => self.build_values_array_from_keys(
                    keys.iter().map(|x: &u8| *x as CatSize),
                    compat_level,
                ),
                CategoricalKeyRemap::U16(keys) => self.build_values_array_from_keys(
                    keys.iter().map(|x: &u16| *x as CatSize),
                    compat_level,
                ),
                CategoricalKeyRemap::U32(keys) => self.build_values_array_from_keys(
                    keys.iter().map(|x: &u32| *x as CatSize),
                    compat_level,
                ),
            },

            Self::Enum { frozen, .. } => {
                let array = frozen.categories();

                if compat_level.0 >= 1 {
                    array.to_boxed()
                } else {
                    // Note: Could be optimized to share the same converted array for older compat
                    // levels.
                    cast_unchecked(array, &ArrowDataType::LargeUtf8).unwrap()
                }
            },
        }
    }

    fn build_values_array_from_keys<I>(
        &self,
        keys_iter: I,
        compat_level: CompatLevel,
    ) -> Box<dyn Array>
    where
        I: ExactSizeIterator<Item = CatSize>,
    {
        let mapping = self.get_categorical_mapping();

        if compat_level != CompatLevel::oldest() {
            let mut builder = Utf8ViewArrayBuilder::new(ArrowDataType::Utf8View);
            builder.reserve(keys_iter.len());

            for x in keys_iter {
                builder.push_value_ignore_validity(mapping.cat_to_str(x).unwrap())
            }

            builder.freeze().to_boxed()
        } else {
            let mut builder: MutableUtf8Array<i64> = MutableUtf8Array::new();
            builder.reserve(keys_iter.len(), 0);

            for x in keys_iter {
                builder.push(Some(mapping.cat_to_str(x).unwrap()));
            }

            let out: Utf8Array<i64> = builder.into();
            out.boxed()
        }
    }
}

pub enum CategoricalKeyRemap {
    U8(PlIndexSet<u8>),
    U16(PlIndexSet<u16>),
    U32(PlIndexSet<u32>),
}

impl CategoricalKeyRemap {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        match self {
            Self::U8(v) => v as _,
            Self::U16(v) => v as _,
            Self::U32(v) => v as _,
        }
    }
}

impl From<PlIndexSet<u8>> for CategoricalKeyRemap {
    fn from(value: PlIndexSet<u8>) -> Self {
        Self::U8(value)
    }
}

impl From<PlIndexSet<u16>> for CategoricalKeyRemap {
    fn from(value: PlIndexSet<u16>) -> Self {
        Self::U16(value)
    }
}

impl From<PlIndexSet<u32>> for CategoricalKeyRemap {
    fn from(value: PlIndexSet<u32>) -> Self {
        Self::U32(value)
    }
}
