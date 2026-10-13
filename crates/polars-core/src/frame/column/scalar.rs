use std::sync::OnceLock;

use polars_error::PolarsResult;
use polars_utils::broadcast::BroadcastLength;
use polars_utils::pl_str::PlSmallStr;

use super::{AnyValue, Column, DataType, IntoColumn, Scalar, Series};
use crate::chunked_array::cast::CastOptions;

/// A [`Column`] that consists of a repeated [`Scalar`]
///
/// This is lazily materialized into a [`Series`].
#[derive(Debug, Clone)]
pub struct ScalarColumn {
    name: PlSmallStr,
    // The value of this scalar may be unspecified when `length == 0`.
    scalar: Scalar,
    length: usize,

    // invariants:
    // materialized.name() == name
    // materialized.len() == length
    // materialized.dtype() == value.dtype
    // materialized[i] == value, for all 0 <= i < length
    /// A lazily materialized [`Series`] variant of this [`ScalarColumn`]
    materialized: OnceLock<Series>,
}

impl ScalarColumn {
    #[inline]
    pub fn new(name: PlSmallStr, scalar: Scalar, length: usize) -> Self {
        Self {
            name,
            scalar,
            length,

            materialized: OnceLock::new(),
        }
    }

    #[inline]
    pub fn new_empty(name: PlSmallStr, dtype: DataType) -> Self {
        Self {
            name,
            scalar: Scalar::new(dtype, AnyValue::Null),
            length: 0,

            materialized: OnceLock::new(),
        }
    }

    pub fn full_null(name: PlSmallStr, length: usize, dtype: DataType) -> Self {
        Self::new(name, Scalar::null(dtype), length)
    }

    pub fn name(&self) -> &PlSmallStr {
        &self.name
    }

    pub fn scalar(&self) -> &Scalar {
        &self.scalar
    }

    pub fn dtype(&self) -> &DataType {
        self.scalar.dtype()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn is_full_null(&self) -> bool {
        self.scalar.is_null()
    }

    fn _to_series(name: PlSmallStr, value: Scalar, length: usize) -> Series {
        let series = if length == 0 {
            Series::new_empty(name, value.dtype())
        } else {
            value.into_series(name).new_from_index(0, length)
        };

        debug_assert_eq!(series.len(), length);

        series
    }

    /// Materialize the [`ScalarColumn`] into a [`Series`].
    pub fn to_series(&self) -> Series {
        Self::_to_series(self.name.clone(), self.scalar.clone(), self.length)
    }

    /// Get the [`ScalarColumn`] as [`Series`] if it was already materialized.
    pub fn lazy_as_materialized_series(&self) -> Option<&Series> {
        self.materialized.get()
    }

    /// Get the [`ScalarColumn`] as [`Series`]
    ///
    /// This needs to materialize upon the first call. Afterwards, this is cached.
    pub fn as_materialized_series(&self) -> &Series {
        self.materialized.get_or_init(|| self.to_series())
    }

    /// Take the [`ScalarColumn`] and materialize as a [`Series`] if not already done.
    pub fn take_materialized_series(self) -> Series {
        self.materialized
            .into_inner()
            .unwrap_or_else(|| Self::_to_series(self.name, self.scalar, self.length))
    }

    /// Estimated size of the column, see [`Series::estimated_size`].
    ///
    /// If the column is not materialized and `expanded` is false, only the scalar value is
    /// counted. Otherwise, this is the size of the materialized [`Series`], computed without
    /// materializing it.
    pub fn estimated_size(&self, expanded: bool) -> usize {
        if let Some(s) = self.materialized.get() {
            return s.estimated_size();
        }
        let length = if expanded { self.length } else { 1 };
        estimated_repeated_size(self.dtype(), self.scalar.value(), length)
            .unwrap_or_else(|| self.as_single_value_series().estimated_size() * length)
    }

    /// Take the [`ScalarColumn`] as a series with a single value.
    ///
    /// If the [`ScalarColumn`] has `length=0` the resulting `Series` will also have `length=0`.
    pub fn as_single_value_series(&self) -> Series {
        self.as_n_values_series(1)
    }

    /// Take the [`ScalarColumn`] as a series with a `n` values.
    ///
    /// If the [`ScalarColumn`] has `length=0` the resulting `Series` will also have `length=0`.
    pub fn as_n_values_series(&self, n: usize) -> Series {
        let length = usize::min(n, self.length);

        match self.materialized.get() {
            // Don't take a refcount if we only want length-1 (or empty) - the materialized series
            // could be extremely large.
            Some(s) if length == self.length || length > 1 => s.head(Some(length)),
            _ => Self::_to_series(self.name.clone(), self.scalar.clone(), length),
        }
    }

    /// Create a new [`ScalarColumn`] from a `length=1` Series and expand it `length`.
    ///
    /// This will panic if the value cannot be made static or if the series has length `0`.
    #[inline]
    pub fn unit_scalar_from_series(series: Series) -> Self {
        assert_eq!(series.len(), 1);
        // SAFETY: We just did the bounds check
        let value = unsafe { series.get_unchecked(0) };
        let value = value.into_static();
        let value = Scalar::new(series.dtype().clone(), value);
        let mut sc = ScalarColumn::new(series.name().clone(), value, 1);
        sc.materialized = OnceLock::from(series);
        sc
    }

    /// Create a new [`ScalarColumn`] from a `length<=1` Series and expand it `length`.
    ///
    /// If `series` is empty and `length` is non-zero, a full-NULL column of `length` will be returned.
    ///
    /// This will panic if the value cannot be made static.
    pub fn from_single_value_series(series: Series, length: usize) -> Self {
        debug_assert!(series.len() <= 1);

        let value = if series.is_empty() {
            AnyValue::Null
        } else {
            unsafe { series.get_unchecked(0) }.into_static()
        };
        let value = Scalar::new(series.dtype().clone(), value);
        ScalarColumn::new(series.name().clone(), value, length)
    }

    /// Resize the [`ScalarColumn`] to new `length`.
    ///
    /// This reuses the materialized [`Series`], if `length <= self.length`.
    pub fn resize(&self, length: usize) -> ScalarColumn {
        if self.length == length {
            return self.clone();
        }

        // This is violates an invariant if this triggers, the scalar value is undefined if the
        // self.length == 0 so therefore we should never resize using that value.
        debug_assert!(length == 0 || self.length > 0);

        let mut resized = Self {
            name: self.name.clone(),
            scalar: self.scalar.clone(),
            length,
            materialized: OnceLock::new(),
        };

        if length == self.length || (length < self.length && length > 1) {
            if let Some(materialized) = self.materialized.get() {
                resized.materialized = OnceLock::from(materialized.head(Some(length)));
                debug_assert_eq!(resized.materialized.get().unwrap().len(), length);
            }
        }

        resized
    }

    /// Append `other` to `self`, keeping both unmaterialized.
    ///
    /// Returns whether that was possible. `self` is left untouched when it was not.
    pub fn try_append(&mut self, other: &Self) -> bool {
        // Unequal dtypes either need a cast or must raise.
        if self.dtype() != other.dtype() {
            return false;
        }

        // The value of a length-0 column is unspecified, so it takes on the other's.
        if other.is_empty() {
            return true;
        }
        if self.is_empty() {
            let name = std::mem::take(&mut self.name);
            *self = other.clone();
            self.rename(name);
            return true;
        }

        if !is_same_value(self.dtype(), self.scalar.value(), other.scalar.value()) {
            return false;
        }

        self.length += other.length;
        self.materialized.take();
        true
    }

    pub fn cast_with_options(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Self> {
        // @NOTE: We expect that when casting the materialized series mostly does not need change
        // the physical array. Therefore, we try to cast the entire materialized array if it is
        // available.

        match self.materialized.get() {
            Some(s) => {
                let materialized = s.cast_with_options(dtype, options)?;
                assert_eq!(self.length, materialized.len());

                let mut casted = if materialized.is_empty() {
                    Self::new_empty(materialized.name().clone(), materialized.dtype().clone())
                } else {
                    // SAFETY: Just did bounds check
                    let scalar = unsafe { materialized.get_unchecked(0) }.into_static();
                    Self::new(
                        materialized.name().clone(),
                        Scalar::new(materialized.dtype().clone(), scalar),
                        self.length,
                    )
                };
                casted.materialized = OnceLock::from(materialized);
                Ok(casted)
            },
            None => {
                let s = self
                    .as_single_value_series()
                    .cast_with_options(dtype, options)?;

                if self.length == 0 {
                    Ok(Self::new_empty(s.name().clone(), s.dtype().clone()))
                } else {
                    assert_eq!(1, s.len());
                    Ok(Self::from_single_value_series(s, self.length))
                }
            },
        }
    }

    pub fn strict_cast(&self, dtype: &DataType) -> PolarsResult<Self> {
        self.cast_with_options(dtype, CastOptions::Strict)
    }
    pub fn cast(&self, dtype: &DataType) -> PolarsResult<Self> {
        self.cast_with_options(dtype, CastOptions::NonStrict)
    }
    /// # Safety
    ///
    /// This can lead to invalid memory access in downstream code.
    pub unsafe fn cast_unchecked(&self, dtype: &DataType) -> PolarsResult<Self> {
        // @NOTE: We expect that when casting the materialized series mostly does not need change
        // the physical array. Therefore, we try to cast the entire materialized array if it is
        // available.

        match self.materialized.get() {
            Some(s) => {
                let materialized = s.cast_unchecked(dtype)?;
                assert_eq!(self.length, materialized.len());

                let mut casted = if materialized.is_empty() {
                    Self::new_empty(materialized.name().clone(), materialized.dtype().clone())
                } else {
                    // SAFETY: Just did bounds check
                    let scalar = unsafe { materialized.get_unchecked(0) }.into_static();
                    Self::new(
                        materialized.name().clone(),
                        Scalar::new(materialized.dtype().clone(), scalar),
                        self.length,
                    )
                };
                casted.materialized = OnceLock::from(materialized);
                Ok(casted)
            },
            None => {
                let s = self.as_single_value_series().cast_unchecked(dtype)?;
                assert_eq!(1, s.len());

                if self.length == 0 {
                    Ok(Self::new_empty(s.name().clone(), s.dtype().clone()))
                } else {
                    Ok(Self::from_single_value_series(s, self.length))
                }
            },
        }
    }

    pub fn rename(&mut self, name: PlSmallStr) -> &mut Self {
        if let Some(series) = self.materialized.get_mut() {
            series.rename(name.clone());
        }

        self.name = name;
        self
    }

    pub fn has_nulls(&self) -> bool {
        self.length != 0 && self.scalar.is_null()
    }

    pub fn drop_nulls(&self) -> Self {
        if self.scalar.is_null() {
            self.resize(0)
        } else {
            self.clone()
        }
    }

    pub fn into_nulls(mut self) -> Self {
        self.scalar.update(AnyValue::Null);
        self
    }

    /// Packs every element into a single-element list.
    pub fn to_unit_list(&self) -> Self {
        let mut slf = self.clone();
        slf.map_scalar(|s| Scalar::new_list(s.into_series(PlSmallStr::EMPTY)));
        slf
    }

    pub fn map_scalar(&mut self, map_scalar: impl Fn(Scalar) -> Scalar) {
        self.scalar = map_scalar(std::mem::take(&mut self.scalar));
        self.materialized.take();
    }
    pub fn with_value(&mut self, value: AnyValue<'static>) -> &mut Self {
        self.scalar.update(value);
        self.materialized.take();
        self
    }
}

/// Whether `l` and `r`, both of `dtype`, are the same value and not only equal ones.
///
/// For floats `0.0 == -0.0`, and for objects `==` calls into Python, so those are only
/// treated as the same when that is cheap to prove.
fn is_same_value(dtype: &DataType, l: &AnyValue, r: &AnyValue) -> bool {
    match (l, r) {
        (AnyValue::Null, AnyValue::Null) => true,
        (AnyValue::Float16(l), AnyValue::Float16(r)) => l.to_bits() == r.to_bits(),
        (AnyValue::Float32(l), AnyValue::Float32(r)) => l.to_bits() == r.to_bits(),
        (AnyValue::Float64(l), AnyValue::Float64(r)) => l.to_bits() == r.to_bits(),
        _ => {
            let mut eq_is_same = true;
            dtype.visit_with(|dtype| eq_is_same &= !(dtype.is_float() || dtype.is_object()));
            eq_is_same && l == r
        },
    }
}

/// [`Series::estimated_size`] of `value` repeated `length` times, as built by
/// [`Series::new_from_index`].
///
/// Returns `None` if the size cannot be derived from `dtype` and `value`.
fn estimated_repeated_size(dtype: &DataType, value: &AnyValue, length: usize) -> Option<usize> {
    let is_null = value.is_null();
    let validity_size = if is_null { length.div_ceil(8) } else { 0 };
    let size = match dtype {
        DataType::Null => 0,
        DataType::Boolean => length.div_ceil(8) + validity_size,
        // Views and validity are not counted for these types, see `estimated_bytes_size`.
        DataType::String | DataType::Binary if is_null => 0,
        DataType::String => length * value.extract_str()?.len(),
        DataType::Binary => length * value.extract_bytes()?.len(),
        DataType::List(_) => match value {
            AnyValue::List(s) => {
                estimated_repeated_series_size(s, length) + length * size_of::<i64>()
            },
            _ if is_null => length * size_of::<i64>() + validity_size,
            _ => return None,
        },
        #[cfg(feature = "dtype-array")]
        DataType::Array(inner, width) => match value {
            AnyValue::Array(s, _) => estimated_repeated_series_size(s, length),
            _ if is_null => {
                estimated_repeated_size(inner, &AnyValue::Null, length * width)? + validity_size
            },
            _ => return None,
        },
        #[cfg(feature = "dtype-struct")]
        DataType::Struct(fields) => match value {
            AnyValue::StructOwned(payload) => payload
                .0
                .iter()
                .zip(fields)
                .map(|(value, field)| estimated_repeated_size(field.dtype(), value, length))
                .sum::<Option<usize>>()?,
            _ if is_null => {
                fields
                    .iter()
                    .map(|field| estimated_repeated_size(field.dtype(), &AnyValue::Null, length))
                    .sum::<Option<usize>>()?
                    + validity_size
            },
            _ => return None,
        },
        #[cfg(feature = "dtype-extension")]
        DataType::Extension(_, storage) => return estimated_repeated_size(storage, value, length),
        _ => length * dtype.byte_width()? as usize + validity_size,
    };
    Some(size)
}

/// [`Series::estimated_size`] of `s` repeated `n` times.
fn estimated_repeated_series_size(s: &Series, n: usize) -> usize {
    let dtype = s.dtype();
    match dtype.byte_width() {
        // Bitmaps round up to whole bytes, so scaling the size of `s` would count too much.
        Some(width) if !dtype.is_nested() => {
            let len = n * s.len();
            let validity_size = if s.has_nulls() { len.div_ceil(8) } else { 0 };
            (len as f64 * width).ceil() as usize + validity_size
        },
        _ => n * s.estimated_size(),
    }
}

impl IntoColumn for ScalarColumn {
    #[inline(always)]
    fn into_column(self) -> Column {
        self.into()
    }
}

impl From<ScalarColumn> for Column {
    #[inline]
    fn from(value: ScalarColumn) -> Self {
        Self::Scalar(value)
    }
}

impl BroadcastLength for ScalarColumn {
    fn _broadcast_len(&self) -> usize {
        self.len()
    }

    fn _column_name(&self) -> Option<&str> {
        Some(self.name())
    }
}

#[cfg(feature = "dsl-schema")]
impl schemars::JsonSchema for ScalarColumn {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "ScalarColumn".into()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(concat!(module_path!(), "::", "ScalarColumn"))
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        serde_impl::SerializeWrap::json_schema(generator)
    }
}

#[cfg(feature = "serde")]
mod serde_impl {
    use std::sync::OnceLock;

    use polars_error::PolarsError;
    use polars_utils::pl_str::PlSmallStr;

    use super::ScalarColumn;
    use crate::frame::{Scalar, Series};

    #[derive(serde::Serialize, serde::Deserialize)]
    #[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
    pub struct SerializeWrap {
        name: PlSmallStr,
        /// Unit-length series for dispatching to IPC serialize
        unit_series: Series,
        length: usize,
    }

    impl From<&ScalarColumn> for SerializeWrap {
        fn from(value: &ScalarColumn) -> Self {
            Self {
                name: value.name.clone(),
                unit_series: value.scalar.clone().into_series(PlSmallStr::EMPTY),
                length: value.length,
            }
        }
    }

    impl TryFrom<SerializeWrap> for ScalarColumn {
        type Error = PolarsError;

        fn try_from(value: SerializeWrap) -> Result<Self, Self::Error> {
            let slf = Self {
                name: value.name,
                scalar: Scalar::new(
                    value.unit_series.dtype().clone(),
                    value.unit_series.get(0)?.into_static(),
                ),
                length: value.length,
                materialized: OnceLock::new(),
            };

            Ok(slf)
        }
    }

    impl serde::ser::Serialize for ScalarColumn {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            SerializeWrap::from(self).serialize(serializer)
        }
    }

    impl<'de> serde::de::Deserialize<'de> for ScalarColumn {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            use serde::de::Error;

            SerializeWrap::deserialize(deserializer)
                .and_then(|x| ScalarColumn::try_from(x).map_err(D::Error::custom))
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::*;

    fn check_estimated_size(s: Series) {
        let scalar = Scalar::new(s.dtype().clone(), s.get(0).unwrap().into_static());
        let unit_size = ScalarColumn::new(s.name().clone(), scalar.clone(), 1)
            .to_series()
            .estimated_size();
        let sc = ScalarColumn::new(s.name().clone(), scalar, 1001);
        assert_eq!(sc.estimated_size(true), sc.to_series().estimated_size());
        assert_eq!(sc.estimated_size(false), unit_size);

        let materialized_size = sc.as_materialized_series().estimated_size();
        assert_eq!(sc.estimated_size(false), materialized_size);
    }

    #[test]
    fn test_estimated_size() -> PolarsResult<()> {
        let list = Series::new("a".into(), [Series::new("".into(), [Some(1i32), None])]);
        let mut series = vec![
            Series::new("a".into(), [1i64]),
            Series::new("a".into(), [true]),
            Series::new("a".into(), ["a".repeat(20)]),
            Series::new("a".into(), [b"ab".as_slice()]),
            list.clone(),
        ];
        #[cfg(feature = "dtype-array")]
        series.push(list.cast(&DataType::Array(Box::new(DataType::Int32), 2))?);
        #[cfg(feature = "dtype-struct")]
        series.push(
            StructChunked::from_series(
                "a".into(),
                1,
                [Series::new("x".into(), [1i8]), list.clone()].iter(),
            )?
            .into_series(),
        );

        for s in series {
            check_estimated_size(Series::full_null("a".into(), 1, s.dtype()));
            check_estimated_size(s);
        }
        Ok(())
    }
}
