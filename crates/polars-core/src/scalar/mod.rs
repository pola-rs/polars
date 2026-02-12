mod from;
mod new;
#[cfg(any(feature = "serde", feature = "dsl-schema"))]
mod serde;

use std::hash::Hash;

use polars_error::PolarsResult;
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;

use crate::chunked_array::cast::CastOptions;
use crate::datatypes::{AnyValue, DataType};
use crate::prelude::{Column, Series};

#[derive(Clone, Debug, PartialEq)]
pub struct Scalar {
    dtype: DataType,
    value: AnyValue<'static>,
}

impl Hash for Scalar {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.dtype.hash(state);
        self.value.hash_impl(state, true);
    }
}

impl Default for Scalar {
    fn default() -> Self {
        Self {
            dtype: DataType::Null,
            value: AnyValue::Null,
        }
    }
}

impl Scalar {
    #[inline(always)]
    pub const fn new(dtype: DataType, value: AnyValue<'static>) -> Self {
        Self { dtype, value }
    }

    pub const fn null(dtype: DataType) -> Self {
        Self::new(dtype, AnyValue::Null)
    }

    pub fn new_idxsize(value: IdxSize) -> Self {
        value.into()
    }

    pub fn cast_with_options(self, dtype: &DataType, options: CastOptions) -> PolarsResult<Self> {
        if self.dtype() == dtype {
            return Ok(self);
        }

        // @Optimize: If we have fully fleshed out casting semantics, we could just specify the
        // cast on AnyValue.
        let s = self
            .into_series(PlSmallStr::from_static("scalar"))
            .cast_with_options(dtype, options)?;
        let value = s.get(0).unwrap();
        Ok(Self::new(s.dtype().clone(), value.into_static()))
    }

    #[inline(always)]
    pub fn is_null(&self) -> bool {
        self.value.is_null()
    }

    #[inline(always)]
    pub fn is_nan(&self) -> bool {
        self.value.is_nan()
    }

    #[inline(always)]
    pub fn into_value(self) -> AnyValue<'static> {
        self.value
    }

    #[inline(always)]
    pub fn value(&self) -> &AnyValue<'static> {
        &self.value
    }

    pub fn as_any_value(&self) -> AnyValue<'_> {
        self.value.clone()
    }

    pub fn into_series(self, name: PlSmallStr) -> Series {
        Series::from_any_values_and_dtype(name, &[self.as_any_value()], &self.dtype, true).unwrap()
    }

    /// Turn a scalar into a column with `length=1`.
    pub fn into_column(self, name: PlSmallStr) -> Column {
        Column::new_scalar(name, self, 1)
    }

    #[inline(always)]
    pub fn dtype(&self) -> &DataType {
        &self.dtype
    }

    #[inline(always)]
    pub fn update(&mut self, value: AnyValue<'static>) {
        self.value = value;
    }

    #[inline(always)]
    pub fn with_value(mut self, value: AnyValue<'static>) -> Self {
        self.update(value);
        self
    }

    #[inline(always)]
    pub fn any_value_mut(&mut self) -> &mut AnyValue<'static> {
        &mut self.value
    }

    pub fn to_physical(mut self) -> Scalar {
        self.dtype = self.dtype.to_physical();
        self.value = self.value.to_physical();
        self
    }
}

#[cfg(all(test, feature = "proptest", not(miri)))]
mod tests {
    use std::rc::Rc;

    use proptest::prelude::*;

    use crate::chunked_array::cast::CastOptions;
    use crate::datatypes::proptest::{
        AnyValueArbitraryOptions, AnyValueArbitrarySelection, DataTypeArbitraryOptions,
        DataTypeArbitrarySelection, anyvalue_strategy, dtypes_strategy,
    };
    use crate::scalar::Scalar;

    fn test_anyvalue_options() -> AnyValueArbitraryOptions {
        AnyValueArbitraryOptions {
            allowed_dtypes: AnyValueArbitrarySelection::all()
                & !AnyValueArbitrarySelection::CATEGORICAL
                & !AnyValueArbitrarySelection::CATEGORICAL_OWNED
                & !AnyValueArbitrarySelection::ENUM
                & !AnyValueArbitrarySelection::ENUM_OWNED
                & !AnyValueArbitrarySelection::OBJECT
                & !AnyValueArbitrarySelection::OBJECT_OWNED
                & !AnyValueArbitrarySelection::LIST
                & !AnyValueArbitrarySelection::ARRAY
                & !AnyValueArbitrarySelection::STRUCT
                & !AnyValueArbitrarySelection::STRUCT_OWNED
                & !AnyValueArbitrarySelection::DATETIME
                & !AnyValueArbitrarySelection::DATETIME_OWNED
                & !AnyValueArbitrarySelection::DATE
                & !AnyValueArbitrarySelection::TIME
                & !AnyValueArbitrarySelection::DURATION
                & !AnyValueArbitrarySelection::BINARY_OWNED,
            categories_range: 1..=3,
            ..Default::default()
        }
    }

    fn test_dtype_options() -> DataTypeArbitraryOptions {
        DataTypeArbitraryOptions {
            allowed_dtypes: DataTypeArbitrarySelection::all()
                & !DataTypeArbitrarySelection::CATEGORICAL
                & !DataTypeArbitrarySelection::ENUM
                & !DataTypeArbitrarySelection::OBJECT
                & !DataTypeArbitrarySelection::LIST
                & !DataTypeArbitrarySelection::ARRAY
                & !DataTypeArbitrarySelection::STRUCT
                & !DataTypeArbitrarySelection::DATETIME
                & !DataTypeArbitrarySelection::DATE
                & !DataTypeArbitrarySelection::TIME
                & !DataTypeArbitrarySelection::DURATION
                & !DataTypeArbitrarySelection::BINARY_OFFSET,
            categories_range: 1..=3,
            ..Default::default()
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]
        #[test]
        fn test_scalar_cast_with_options(
            source_value in anyvalue_strategy(Rc::new(test_anyvalue_options()), 0),
            target_dtype in dtypes_strategy(Rc::new(test_dtype_options()), 0),
        ) {
            let source_dtype = source_value.dtype();
            let scalar = Scalar::new(source_dtype.clone(), source_value.clone());

            let cast_result = scalar.cast_with_options(&target_dtype, CastOptions::default());

            if let Ok(casted_scalar) = cast_result {
                prop_assert_eq!(casted_scalar.dtype(), &target_dtype);
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]
        #[test]
        fn test_scalar_cast_identity(
            source_value in anyvalue_strategy(Rc::new(test_anyvalue_options()), 0),
        ) {
            let source_dtype = source_value.dtype();
            let scalar = Scalar::new(source_dtype.clone(), source_value);

            let result = scalar.clone().cast_with_options(&source_dtype, CastOptions::default());

            prop_assert!(result.is_ok());
            if let Ok(casted) = result {
                prop_assert_eq!(casted.dtype(), scalar.dtype());
            }
        }
    }
}
