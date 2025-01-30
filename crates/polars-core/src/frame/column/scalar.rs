use std::sync::OnceLock;

use polars_error::PolarsResult;
use polars_utils::pl_str::PlSmallStr;

use super::{AnyValue, Column, DataType, IntoColumn, Scalar, Series};
use crate::chunked_array::cast::CastOptions;

/// A [`Column`] that consists of a repeated [`Scalar`]
///
/// This is lazily materialized into a [`Series`].
#[derive(Debug, Clone)]
pub struct ScalarColumn {
    name: PlSmallStr,
    // The value of this scalar may be incoherent when `length == 0`.
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

    pub fn name(&self) -> &PlSmallStr {
        &self.name
    }

    pub fn scalar(&self) -> &Scalar {
        &self.scalar
    }

    pub fn dtype(&self) -> &DataType {
        self.scalar.dtype()
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
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

    /// Create a new [`ScalarColumn`] from a `length=1` Series and expand it `length`.
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

    pub fn cast_with_options(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Self> {
        // @NOTE: We expect that when casting the materialized series mostly does not need change
        // the physical array. Therefore, we try to cast the entire materialized array if it is
        // available.

        match self.materialized.get() {
            Some(s) => {
                let materialized = s.cast_with_options(dtype, options)?;
                assert_eq!(self.length, materialized.len());

                let mut casted = if materialized.len() == 0 {
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

                let mut casted = if materialized.len() == 0 {
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

#[cfg(feature = "serde")]
mod serde_impl {
    use std::sync::OnceLock;

    use polars_error::PolarsError;
    use polars_utils::pl_str::PlSmallStr;

    use super::ScalarColumn;
    use crate::frame::{Scalar, Series};

    #[derive(serde::Serialize, serde::Deserialize)]
    struct SerializeWrap {
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
