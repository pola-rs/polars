use std::borrow::Cow;
use std::convert::identity;
use std::sync::{Arc, OnceLock};

use arrow::array::BooleanArray;
use polars_error::constants::LENGTH_LIMIT_MSG;
use polars_error::{polars_bail, polars_ensure, PolarsResult};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::IdxSize;

use super::{AnyValue, ChunkCompareEq, Column, DataType, Field, IntoColumn, ScalarColumn, Series};
use crate::chunked_array::cast::CastOptions;
use crate::frame::Scalar;
use crate::series::IsSorted;
use crate::utils::verify_usize_sum_to_idxsize;

/// A column that utilizes run-length encoding to save space and simplify evaluation.
///
/// This way `["a", ... 100 times ..., "a", "b", ... 300 times ..., "b", "c", ... 50 times... "c"]`
/// can be encodes as `["a", "b", "c"]` and `[100, 300, 50]`. We say `("a", 100)`, `("b", 300)` and
/// `("c", 50)` are partitions.
///
/// This column could be formed from doing a `sort` or from doing a appending two
/// [`ScalarColumn`][super::ScalarColumn]s.
#[derive(Debug, Clone)]
pub struct PartitionedColumn {
    name: PlSmallStr,

    /// Partition values.
    ///
    /// The same value be appear multiple times, even after one another (although that should be
    /// avoided if possible).
    ///
    /// Invariants:
    /// - the name of the [`Series`] is undefined and should not be relied on
    /// - the `i`-th value in the [`Series`] belongs to the `i`-th partition.
    /// - the [`DataType`] of the [`Series`] is the [`DataType`] of the [`PartitionedColumn`].
    /// - always rechunked as one chunk
    values: Series,
    /// End indexes of each partition.
    ///
    /// Invariant:
    /// - must be monotonely non-decreasing (ends[i] <= ends[i+1]).
    /// - ends.len() == values.len()
    ends: Arc<[IdxSize]>,

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

    polars_ensure!(
        values.n_chunks() == 1,
        ComputeError: "partitioned column `values` are not rechunked",
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
        let mut column = Column::Series(Series::new_empty(name, dtype));

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

    /// Merge consecutive partitions with the same value.
    pub fn remove_redudancies(&self) -> Self {
        let mut c = self.clone();
        c.remove_redudancies_mut();
        c
    }

    /// Merge consecutive partitions with the same value.
    pub fn remove_redudancies_mut(&mut self) -> &mut Self {
        let num_partitions = self.values.len();
        if num_partitions <= 1 {
            return self;
        }

        let lhs = self.values.slice(0, num_partitions - 1);
        let rhs = self.values.slice(1, num_partitions - 1);

        // Create a Bitmap which which partitions are redudant
        let redudancy_mask = lhs.equal_missing(&rhs).unwrap();
        let redudancy_mask = redudancy_mask.chunks()[0]
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        let redudancy_mask = match redudancy_mask.validity() {
            None => redudancy_mask.values().clone(),
            Some(v) => redudancy_mask.values() & v,
        };

        // If no partitions are redudant, just return.
        if redudancy_mask.set_bits() == 0 {
            return self;
        }

        let num_output_partitions = redudancy_mask.unset_bits();

        // Create the new `values` and `ends` for the PartitionedColumn without redudancies.
        let mut offset = 0;
        let mut acc_length = 0;
        let mut ends = Vec::with_capacity(num_output_partitions);
        let mut indices = Vec::with_capacity(num_output_partitions);
        for (i, (is_needed, length)) in redudancy_mask
            .iter()
            .zip(self.partition_lengths_iter())
            .enumerate()
        {
            acc_length += length;
            if is_needed {
                ends.push(offset + acc_length);
                offset += acc_length;
                indices.push(i as IdxSize);
            }
        }

        self.ends = ends.into();
        self.values = unsafe { self.values.take_slice_unchecked(&indices) };

        self
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

    pub fn partition_lengths_iter<'a>(&'a self) -> impl ExactSizeIterator<Item = IdxSize> + 'a {
        (0..self.ends.len()).map(|i| self.ends[i] - self.ends.get(i).copied().unwrap_or_default())
    }

    pub fn partition_iter<'a>(
        &'a self,
    ) -> impl ExactSizeIterator<Item = (AnyValue<'a>, IdxSize)> + 'a {
        (0..self.ends.len()).map(|i| {
            (
                self.values.get(i).unwrap(),
                self.ends[i] - self.ends.get(i).copied().unwrap_or_default(),
            )
        })
    }

    pub fn null_count(&self) -> usize {
        match self.lazy_as_materialized_series() {
            Some(s) => s.null_count(),
            None => {
                let values_nc = self.values.null_count();

                if values_nc == 0 {
                    0
                } else {
                    let mut null_count = 0;
                    let validity = self.values.chunks()[0].validity().unwrap();
                    for (v, length) in validity.iter().zip(self.partition_lengths_iter()) {
                        if !v {
                            null_count += length;
                        }
                    }
                    null_count as usize
                }
            },
        }
    }

    pub fn clear(&self) -> Self {
        Self::new_empty(self.name.clone(), self.values.dtype().clone())
    }

    /// Append a [`PartitionedColumn`] to this [`PartitionedColumn`].
    pub fn append(&mut self, rhs: &PartitionedColumn) -> PolarsResult<&mut Self> {
        self.values.extend(&rhs.values)?;
        self.values = self.values.rechunk();
        let mut ends = Vec::with_capacity(self.num_partitions() + rhs.num_partitions());
        ends.extend_from_slice(&self.ends);
        let lhs_offset = self.len() as IdxSize;

        verify_usize_sum_to_idxsize(self.len(), rhs.len())?;

        ends.extend(rhs.ends.iter().map(|&v| v + lhs_offset));
        self.ends = ends.into();

        Ok(self)
    }

    /// Append a [`ScalarColumn`] to this [`PartitionedColumn`].
    pub fn append_scalar(&mut self, rhs: &ScalarColumn) -> PolarsResult<&mut Self> {
        self.values.extend(&rhs.as_single_value_series())?;
        self.values = self.values.rechunk();
        let mut ends = Vec::with_capacity(self.num_partitions() + 1);
        ends.extend_from_slice(&self.ends);

        verify_usize_sum_to_idxsize(self.len(), rhs.len())?;

        ends.push((self.len() + rhs.len()) as IdxSize);
        self.ends = ends.into();

        Ok(self)
    }

    /// Prepend a [`ScalarColumn`] to this [`PartitionedColumn`] and return the result.
    pub fn prepend_scalar(&self, lhs: &ScalarColumn) -> PolarsResult<Self> {
        let mut values = lhs.as_single_value_series();
        values.extend(&self.values)?;
        values = values.rechunk();
        let mut ends = Vec::with_capacity(self.num_partitions() + 1);

        ends.push(lhs.len() as IdxSize);

        verify_usize_sum_to_idxsize(self.len(), lhs.len())?;

        let lhs_offset = lhs.len() as IdxSize;
        ends.extend(self.ends.iter().map(|&v| v + lhs_offset));
        let ends = ends.into();

        Ok(unsafe { Self::new_unchecked(lhs.name().clone(), values, ends) })
    }

    fn num_partitions(&self) -> usize {
        self.ends.len()
    }
}
