use std::borrow::Cow;
use std::convert::identity;
use std::sync::{Arc, OnceLock};

use polars_error::{PolarsResult, polars_ensure};
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;

use super::{AnyValue, Column, DataType, Field, IntoColumn, Series};
use crate::chunked_array::cast::CastOptions;
use crate::frame::Scalar;
use crate::series::IsSorted;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PartitionedColumn {
    name: PlSmallStr,

    values: Series,
    ends: Arc<[IdxSize]>,

    #[cfg_attr(feature = "serde", serde(skip))]
    materialized: OnceLock<Series>,
}

impl IntoColumn for PartitionedColumn {
    fn into_column(self) -> Column {
        Column::Partitioned(self)
    }
}

impl From<PartitionedColumn> for Column {
    fn from(value: PartitionedColumn) -> Self {
        value.into_column()
    }
}

fn verify_invariants(values: &Series, ends: &[IdxSize]) -> PolarsResult<()> {
    polars_ensure!(
        values.len() == ends.len(),
        ComputeError: "partitioned column `values` length does not match `ends` length ({} != {})",
        values.len(),
        ends.len()
    );

    for vs in ends.windows(2) {
        polars_ensure!(
            vs[0] <= vs[1],
            ComputeError: "partitioned column `ends` are not monotonely non-decreasing",
        );
    }

    Ok(())
}

impl PartitionedColumn {
    pub fn new(name: PlSmallStr, values: Series, ends: Arc<[IdxSize]>) -> Self {
        Self::try_new(name, values, ends).unwrap()
    }

    /// # Safety
    ///
    /// Safe if:
    /// - `values.len() == ends.len()`
    /// - all values can have `dtype`
    /// - `ends` is monotonely non-decreasing
    pub unsafe fn new_unchecked(name: PlSmallStr, values: Series, ends: Arc<[IdxSize]>) -> Self {
        if cfg!(debug_assertions) {
            verify_invariants(&values, ends.as_ref()).unwrap();
        }

        let values = values.rechunk();
        Self {
            name,
            values,
            ends,
            materialized: OnceLock::new(),
        }
    }

    pub fn try_new(name: PlSmallStr, values: Series, ends: Arc<[IdxSize]>) -> PolarsResult<Self> {
        verify_invariants(&values, ends.as_ref())?;

        // SAFETY: Invariants checked before
        Ok(unsafe { Self::new_unchecked(name, values, ends) })
    }

    pub fn new_empty(name: PlSmallStr, dtype: DataType) -> Self {
        Self {
            name,
            values: Series::new_empty(PlSmallStr::EMPTY, &dtype),
            ends: Arc::default(),

            materialized: OnceLock::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.ends.last().map_or(0, |last| *last as usize)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn name(&self) -> &PlSmallStr {
        &self.name
    }

    pub fn dtype(&self) -> &DataType {
        self.values.dtype()
    }

    #[inline]
    pub fn field(&self) -> Cow<Field> {
        match self.lazy_as_materialized_series() {
            None => Cow::Owned(Field::new(self.name().clone(), self.dtype().clone())),
            Some(s) => s.field(),
        }
    }

    pub fn rename(&mut self, name: PlSmallStr) -> &mut Self {
        self.name = name;
        self
    }

    fn _to_series(name: PlSmallStr, values: &Series, ends: &[IdxSize]) -> Series {
        let dtype = values.dtype();
        let mut column = Column::Series(Series::new_empty(name, dtype).into());

        let mut prev_offset = 0;
        for (i, &offset) in ends.iter().enumerate() {
            // @TODO: Optimize
            let length = offset - prev_offset;
            column
                .extend(&Column::new_scalar(
                    PlSmallStr::EMPTY,
                    Scalar::new(dtype.clone(), values.get(i).unwrap().into_static()),
                    length as usize,
                ))
                .unwrap();
            prev_offset = offset;
        }

        debug_assert_eq!(column.len(), prev_offset as usize);

        column.take_materialized_series()
    }

    /// Materialize the [`PartitionedColumn`] into a [`Series`].
    fn to_series(&self) -> Series {
        Self::_to_series(self.name.clone(), &self.values, &self.ends)
    }

    /// Get the [`PartitionedColumn`] as [`Series`] if it was already materialized.
    pub fn lazy_as_materialized_series(&self) -> Option<&Series> {
        self.materialized.get()
    }

    /// Get the [`PartitionedColumn`] as [`Series`]
    ///
    /// This needs to materialize upon the first call. Afterwards, this is cached.
    pub fn as_materialized_series(&self) -> &Series {
        self.materialized.get_or_init(|| self.to_series())
    }

    /// Take the [`PartitionedColumn`] and materialize as a [`Series`] if not already done.
    pub fn take_materialized_series(self) -> Series {
        self.materialized
            .into_inner()
            .unwrap_or_else(|| Self::_to_series(self.name, &self.values, &self.ends))
    }

    pub fn apply_unary_elementwise(&self, f: impl Fn(&Series) -> Series) -> Self {
        let result = f(&self.values).rechunk();
        assert_eq!(self.values.len(), result.len());
        unsafe { Self::new_unchecked(self.name.clone(), result, self.ends.clone()) }
    }

    pub fn try_apply_unary_elementwise(
        &self,
        f: impl Fn(&Series) -> PolarsResult<Series>,
    ) -> PolarsResult<Self> {
        let result = f(&self.values)?.rechunk();
        assert_eq!(self.values.len(), result.len());
        Ok(unsafe { Self::new_unchecked(self.name.clone(), result, self.ends.clone()) })
    }

    pub fn extend_constant(&self, value: AnyValue, n: usize) -> PolarsResult<Self> {
        let mut new_ends = self.ends.to_vec();
        // @TODO: IdxSize checks
        let new_length = (self.len() + n) as IdxSize;

        let values = if !self.is_empty() && self.values.last().value() == &value {
            *new_ends.last_mut().unwrap() = new_length;
            self.values.clone()
        } else {
            new_ends.push(new_length);
            self.values.extend_constant(value, 1)?
        };

        Ok(unsafe { Self::new_unchecked(self.name.clone(), values, new_ends.into()) })
    }

    pub unsafe fn get_unchecked(&self, index: usize) -> AnyValue {
        debug_assert!(index < self.len());

        // Common situation get_unchecked(0)
        if index < self.ends[0] as usize {
            return unsafe { self.get_unchecked(0) };
        }

        let value_idx = self
            .ends
            .binary_search(&(index as IdxSize))
            .map_or_else(identity, identity);

        self.get_unchecked(value_idx)
    }

    pub fn min_reduce(&self) -> PolarsResult<Scalar> {
        self.values.min_reduce()
    }
    pub fn max_reduce(&self) -> Result<Scalar, polars_error::PolarsError> {
        self.values.max_reduce()
    }

    pub fn reverse(&self) -> Self {
        let values = self.values.reverse();
        let mut ends = Vec::with_capacity(self.ends.len());

        let mut offset = 0;
        ends.extend(self.ends.windows(2).rev().map(|vs| {
            offset += vs[1] - vs[0];
            offset
        }));
        ends.push(self.len() as IdxSize);

        unsafe { Self::new_unchecked(self.name.clone(), values, ends.into()) }
    }

    pub fn set_sorted_flag(&mut self, sorted: IsSorted) {
        self.values.set_sorted_flag(sorted);
    }

    pub fn cast_with_options(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Self> {
        let values = self.values.cast_with_options(dtype, options)?;
        Ok(unsafe { Self::new_unchecked(self.name.clone(), values, self.ends.clone()) })
    }

    pub fn strict_cast(&self, dtype: &DataType) -> PolarsResult<Self> {
        let values = self.values.strict_cast(dtype)?;
        Ok(unsafe { Self::new_unchecked(self.name.clone(), values, self.ends.clone()) })
    }

    pub fn cast(&self, dtype: &DataType) -> PolarsResult<Self> {
        let values = self.values.cast(dtype)?;
        Ok(unsafe { Self::new_unchecked(self.name.clone(), values, self.ends.clone()) })
    }

    pub unsafe fn cast_unchecked(&self, dtype: &DataType) -> PolarsResult<Self> {
        let values = unsafe { self.values.cast_unchecked(dtype) }?;
        Ok(unsafe { Self::new_unchecked(self.name.clone(), values, self.ends.clone()) })
    }

    pub fn null_count(&self) -> usize {
        match self.lazy_as_materialized_series() {
            Some(s) => s.null_count(),
            None => {
                // @partition-opt
                self.as_materialized_series().null_count()
            },
        }
    }

    pub fn clear(&self) -> Self {
        Self::new_empty(self.name.clone(), self.values.dtype().clone())
    }

    pub fn partitions(&self) -> &Series {
        &self.values
    }
    pub fn partition_ends(&self) -> &[IdxSize] {
        &self.ends
    }

    pub fn or_reduce(&self) -> PolarsResult<Scalar> {
        self.values.or_reduce()
    }

    pub fn and_reduce(&self) -> PolarsResult<Scalar> {
        self.values.and_reduce()
    }
}
