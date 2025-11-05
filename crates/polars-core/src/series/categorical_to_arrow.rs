use std::any::Any;

use arrow::array::builder::ArrayBuilder;
use arrow::types::NativeType;
use num_traits::AsPrimitive;
use polars_compute::cast::cast_unchecked;

use crate::prelude::*;

/// Categorical converter that prunes unused categories.
pub struct CategoricalToArrowConverter {
    /// Converters keyed by the Arc address of `Arc<CategoricalMapping>`.
    ///
    /// # Safety
    /// The `usize` key remains valid as `CategoricalArrayToArrowConverter` holds a ref-count to the
    /// `Arc<>` that the key is derived from.
    pub converters: PlIndexMap<usize, CategoricalArrayToArrowConverter>,
    /// Persist the key remap to ensure consistent mapping across multiple calls.
    pub persist_remap: bool,
    /// Return only the keys array when going to arrow.
    pub output_keys_only: bool,
}

impl CategoricalToArrowConverter {
    /// # Panics
    /// Panics if:
    /// * `keys_arr` is not of a `Categorical` or `Enum` type
    /// * The arc address of the `Arc<CategoricalMapping>` is not present within `self.converters`
    ///   (likely due to forgetting to call `initialize()` on this converter).
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

    /// Initializes categorical converters for all categorical mappings present in this dtype.
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
            _ => assert!(!dtype.is_nested()),
        }
    }
}

pub enum CategoricalArrayToArrowConverter {
    // # Safety
    // All enum variants must hold a ref to the `Arc<CategoricalMapping>`, as this enum is inserted
    // into a hashmap keyed by the address of the `Arc`ed value.
    //
    Categorical {
        mapping: Arc<CategoricalMapping>,
        key_remap: CategoricalKeyRemap,
    },
    /// Enum keys are not remapped, but we still track this variant to support
    /// the `build_values_array()` function.
    Enum {
        mapping: Arc<CategoricalMapping>,
        frozen: Arc<FrozenCategories>,
    },
}

impl CategoricalArrayToArrowConverter {
    fn array_to_arrow<T>(
        &mut self,
        keys_arr: &PrimitiveArray<T>,
        dtype: &DataType,
        persist_remap: bool,
        output_keys_only: bool,
        compat_level: CompatLevel,
    ) -> Box<dyn Array>
    where
        T: DictionaryKey + NativeType + std::hash::Hash + Eq,
        usize: AsPrimitive<T>,
    {
        let (DataType::Categorical(_, mapping) | DataType::Enum(_, mapping)) = dtype else {
            unreachable!()
        };

        let input_mapping_ptr: *const CategoricalMapping = Arc::as_ptr(mapping);

        let keys_arr: PrimitiveArray<T> = match self {
            Self::Categorical { mapping, key_remap } => {
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
                            // Indexset of T cannot return an index exceeding T::MAX.
                            let out: T = idx.as_();
                            out
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

    /// Build the values array of the dictionary:
    /// * If `Self` is `::Categorical`, this builds according to the current `key_remap` state:
    ///   * If `persist_remap` is `true`, this state will hold all the keys this converter has encountered.
    ///     It will otherwise hold only the keys seen from the last `array_to_arrow()` call.
    /// * If `Self` is `::Enum`, this returns the full set of values present in the Enum's `FrozenCategories`.
    pub fn build_values_array(&self, compat_level: CompatLevel) -> Box<dyn Array> {
        match self {
            Self::Categorical { mapping, key_remap } => match key_remap {
                CategoricalKeyRemap::U8(keys) => self.build_values_array_from_keys(
                    keys.iter().map(|x: &u8| *x as CatSize),
                    mapping,
                    compat_level,
                ),
                CategoricalKeyRemap::U16(keys) => self.build_values_array_from_keys(
                    keys.iter().map(|x: &u16| *x as CatSize),
                    mapping,
                    compat_level,
                ),
                CategoricalKeyRemap::U32(keys) => self.build_values_array_from_keys(
                    keys.iter().map(|x: &u32| *x as CatSize),
                    mapping,
                    compat_level,
                ),
            },

            Self::Enum { frozen, .. } => {
                let array: &Utf8ViewArray = frozen.categories();

                if compat_level != CompatLevel::oldest() {
                    array.to_boxed()
                } else {
                    // Note: Could store a once-init Utf8Array on the frozen categories to avoid
                    // building this multiple times for the oldest compat level.
                    cast_unchecked(array, &ArrowDataType::LargeUtf8).unwrap()
                }
            },
        }
    }

    fn build_values_array_from_keys<I>(
        &self,
        keys_iter: I,
        mapping: &CategoricalMapping,
        compat_level: CompatLevel,
    ) -> Box<dyn Array>
    where
        I: ExactSizeIterator<Item = CatSize>,
    {
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
