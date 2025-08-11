use std::sync::Arc;

use polars_error::PolarsResult;
use polars_utils::pl_str::PlSmallStr;

use super::Scalar;
use crate::datatypes::time_unit::TimeUnit;
use crate::prelude::{AnyValue, DataType, TimeZone};
use crate::series::Series;

impl Scalar {
    #[cfg(feature = "dtype-date")]
    pub fn new_date(value: i32) -> Self {
        Scalar::new(DataType::Date, AnyValue::Date(value))
    }

    #[cfg(feature = "dtype-datetime")]
    pub fn new_datetime(value: i64, time_unit: TimeUnit, tz: Option<TimeZone>) -> Self {
        Scalar::new(
            DataType::Datetime(time_unit, tz.clone()),
            AnyValue::DatetimeOwned(value, time_unit, tz.map(Arc::new)),
        )
    }

    #[cfg(feature = "dtype-duration")]
    pub fn new_duration(value: i64, time_unit: TimeUnit) -> Self {
        Scalar::new(
            DataType::Duration(time_unit),
            AnyValue::Duration(value, time_unit),
        )
    }

    #[cfg(feature = "dtype-time")]
    pub fn new_time(value: i64) -> Self {
        Scalar::new(DataType::Time, AnyValue::Time(value))
    }

    pub fn new_list(values: Series) -> Self {
        Scalar::new(
            DataType::List(Box::new(values.dtype().clone())),
            AnyValue::List(values),
        )
    }

    #[cfg(feature = "dtype-array")]
    pub fn new_array(values: Series, width: usize) -> Self {
        Scalar::new(
            DataType::Array(Box::new(values.dtype().clone()), width),
            AnyValue::Array(values, width),
        )
    }

    #[cfg(feature = "dtype-decimal")]
    pub fn new_decimal(value: i128, scale: usize) -> Self {
        Scalar::new(
            DataType::Decimal(Some(38), Some(scale)),
            AnyValue::Decimal(value, scale),
        )
    }

    #[cfg(feature = "dtype-categorical")]
    pub fn new_enum(
        value: polars_dtype::categorical::CatSize,
        categories: &arrow::array::Utf8ViewArray,
    ) -> PolarsResult<Self> {
        use arrow::array::Array;
        use polars_dtype::categorical::FrozenCategories;

        assert_eq!(categories.null_count(), 0);

        let categories = FrozenCategories::new(categories.values_iter())?;
        let mapping = categories.mapping();
        Ok(Scalar::new(
            DataType::Enum(categories.clone(), mapping.clone()),
            AnyValue::EnumOwned(value, mapping.clone()),
        ))
    }

    #[cfg(feature = "dtype-categorical")]
    pub fn new_categorical(
        value: &str,
        name: PlSmallStr,
        namespace: PlSmallStr,
        physical: polars_dtype::categorical::CategoricalPhysical,
    ) -> PolarsResult<Self> {
        use polars_dtype::categorical::Categories;

        let categories = Categories::new(name, namespace, physical);
        let dt_mapping = categories.mapping();
        let av_mapping = categories.mapping();

        let value = av_mapping.insert_cat(value)?;

        Ok(Scalar::new(
            DataType::Categorical(categories, dt_mapping),
            AnyValue::CategoricalOwned(value, av_mapping),
        ))
    }
}
