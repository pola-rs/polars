#![allow(unsafe_op_in_unsafe_fn)]
//! DataFrame module.
use arrow::datatypes::ArrowSchemaRef;
use polars_row::ArrayRef;
use polars_utils::UnitVec;
use polars_utils::itertools::Itertools;
use rayon::prelude::*;

use crate::chunked_array::flags::StatisticsFlags;
#[cfg(feature = "algorithm_group_by")]
use crate::chunked_array::ops::unique::is_unique_helper;
use crate::prelude::gather::check_bounds_ca;
use crate::prelude::*;
#[cfg(feature = "row_hash")]
use crate::utils::split_df;
use crate::utils::{Container, NoNull, slice_offsets, try_get_supertype};
use crate::{HEAD_DEFAULT_LENGTH, TAIL_DEFAULT_LENGTH};

#[cfg(feature = "dataframe_arithmetic")]
mod arithmetic;
pub mod builder;
mod chunks;
pub use chunks::chunk_df_for_writing;
mod broadcast;
pub mod column;
mod dataframe;
mod filter;
mod projection;
pub use dataframe::DataFrame;
use filter::filter_zero_width;
use projection::{AmortizedColumnSelector, LINEAR_SEARCH_LIMIT};

pub mod explode;
mod from;
#[cfg(feature = "algorithm_group_by")]
pub mod group_by;
pub(crate) mod horizontal;
#[cfg(feature = "proptest")]
pub mod proptest;
#[cfg(any(feature = "rows", feature = "object"))]
pub mod row;
mod top_k;
mod upstream_traits;
mod validation;

use arrow::record_batch::{RecordBatch, RecordBatchT};
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

use crate::POOL;
#[cfg(feature = "row_hash")]
use crate::hashing::_df_rows_to_hashes_threaded_vertical;
use crate::prelude::sort::arg_sort;
use crate::series::IsSorted;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Hash, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[strum(serialize_all = "snake_case")]
pub enum UniqueKeepStrategy {
    /// Keep the first unique row.
    First,
    /// Keep the last unique row.
    Last,
    /// Keep None of the unique rows.
    None,
    /// Keep any of the unique rows
    /// This allows more optimizations
    #[default]
    Any,
}

impl DataFrame {
    pub fn materialized_column_iter(&self) -> impl ExactSizeIterator<Item = &Series> {
        self.columns().iter().map(Column::as_materialized_series)
    }

    /// Returns an estimation of the total (heap) allocated size of the `DataFrame` in bytes.
    ///
    /// # Implementation
    /// This estimation is the sum of the size of its buffers, validity, including nested arrays.
    /// Multiple arrays may share buffers and bitmaps. Therefore, the size of 2 arrays is not the
    /// sum of the sizes computed from this function. In particular, [`StructArray`]'s size is an upper bound.
    ///
    /// When an array is sliced, its allocated size remains constant because the buffer unchanged.
    /// However, this function will yield a smaller number. This is because this function returns
    /// the visible size of the buffer, not its total capacity.
    ///
    /// FFI buffers are included in this estimation.
    pub fn estimated_size(&self) -> usize {
        self.columns().iter().map(Column::estimated_size).sum()
    }

    pub fn try_apply_columns(
        &self,
        func: impl Fn(&Column) -> PolarsResult<Column> + Send + Sync,
    ) -> PolarsResult<Vec<Column>> {
        return inner(self, &func);

        fn inner(
            slf: &DataFrame,
            func: &(dyn Fn(&Column) -> PolarsResult<Column> + Send + Sync),
        ) -> PolarsResult<Vec<Column>> {
            slf.columns().iter().map(func).collect()
        }
    }

    pub fn apply_columns(&self, func: impl Fn(&Column) -> Column + Send + Sync) -> Vec<Column> {
        return inner(self, &func);

        fn inner(slf: &DataFrame, func: &(dyn Fn(&Column) -> Column + Send + Sync)) -> Vec<Column> {
            slf.columns().iter().map(func).collect()
        }
    }

    pub fn try_apply_columns_par(
        &self,
        func: impl Fn(&Column) -> PolarsResult<Column> + Send + Sync,
    ) -> PolarsResult<Vec<Column>> {
        return inner(self, &func);

        fn inner(
            slf: &DataFrame,
            func: &(dyn Fn(&Column) -> PolarsResult<Column> + Send + Sync),
        ) -> PolarsResult<Vec<Column>> {
            POOL.install(|| slf.columns().par_iter().map(func).collect())
        }
    }

    pub fn apply_columns_par(&self, func: impl Fn(&Column) -> Column + Send + Sync) -> Vec<Column> {
        return inner(self, &func);

        fn inner(slf: &DataFrame, func: &(dyn Fn(&Column) -> Column + Send + Sync)) -> Vec<Column> {
            POOL.install(|| slf.columns().par_iter().map(func).collect())
        }
    }

    /// Reserve additional slots into the chunks of the series.
    pub(crate) fn reserve_chunks(&mut self, additional: usize) {
        for s in unsafe { self.columns_mut_retain_schema() } {
            if let Column::Series(s) = s {
                // SAFETY:
                // do not modify the data, simply resize.
                unsafe { s.chunks_mut().reserve(additional) }
            }
        }
    }
    pub fn new_from_index(&self, index: usize, height: usize) -> Self {
        let new_cols = self.apply_columns(|c| c.new_from_index(index, height));

        unsafe { Self::_new_unchecked_impl(height, new_cols).with_schema_from(self) }
    }

    /// Create a new `DataFrame` with the given schema, only containing nulls.
    pub fn full_null(schema: &Schema, height: usize) -> Self {
        let columns = schema
            .iter_fields()
            .map(|f| Column::full_null(f.name().clone(), height, f.dtype()))
            .collect();

        unsafe { DataFrame::_new_unchecked_impl(height, columns) }
    }

    /// Ensure this DataFrame matches the given schema. Casts null columns to
    /// the expected schema if necessary (but nothing else).
    pub fn ensure_matches_schema(&mut self, schema: &Schema) -> PolarsResult<()> {
        let mut did_cast = false;
        let cached_schema = self.cached_schema().cloned();

        for (col, (name, dt)) in unsafe { self.columns_mut() }.iter_mut().zip(schema.iter()) {
            polars_ensure!(
                col.name() == name,
                SchemaMismatch: "column name mismatch: expected {:?}, found {:?}",
                name,
                col.name()
            );

            let needs_cast = col.dtype().matches_schema_type(dt)?;

            if needs_cast {
                *col = col.cast(dt)?;
                did_cast = true;
            }
        }

        if !did_cast {
            unsafe { self.set_opt_schema(cached_schema) };
        }

        Ok(())
    }

    /// Add a new column at index 0 that counts the rows.
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Name" => ["James", "Mary", "John", "Patricia"])?;
    /// assert_eq!(df1.shape(), (4, 1));
    ///
    /// let df2: DataFrame = df1.with_row_index("Id".into(), None)?;
    /// assert_eq!(df2.shape(), (4, 2));
    /// println!("{}", df2);
    ///
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    ///  shape: (4, 2)
    ///  +-----+----------+
    ///  | Id  | Name     |
    ///  | --- | ---      |
    ///  | u32 | str      |
    ///  +=====+==========+
    ///  | 0   | James    |
    ///  +-----+----------+
    ///  | 1   | Mary     |
    ///  +-----+----------+
    ///  | 2   | John     |
    ///  +-----+----------+
    ///  | 3   | Patricia |
    ///  +-----+----------+
    /// ```
    pub fn with_row_index(&self, name: PlSmallStr, offset: Option<IdxSize>) -> PolarsResult<Self> {
        let mut new_columns = Vec::with_capacity(self.width() + 1);
        let offset = offset.unwrap_or(0);

        if self.get_column_index(&name).is_some() {
            polars_bail!(duplicate = name)
        }

        let col = Column::new_row_index(name, offset, self.height())?;
        new_columns.push(col);
        new_columns.extend_from_slice(self.columns());

        Ok(unsafe { DataFrame::new_unchecked(self.height(), new_columns) })
    }

    /// Add a row index column in place.
    ///
    /// # Safety
    /// The caller should ensure the DataFrame does not already contain a column with the given name.
    ///
    /// # Panics
    /// Panics if the resulting column would reach or overflow IdxSize::MAX.
    pub unsafe fn with_row_index_mut(
        &mut self,
        name: PlSmallStr,
        offset: Option<IdxSize>,
    ) -> &mut Self {
        debug_assert!(
            self.get_column_index(&name).is_none(),
            "with_row_index_mut(): column with name {} already exists",
            &name
        );

        let offset = offset.unwrap_or(0);
        let col = Column::new_row_index(name, offset, self.height()).unwrap();

        unsafe { self.columns_mut() }.insert(0, col);
        self
    }

    /// Shrink the capacity of this DataFrame to fit its length.
    pub fn shrink_to_fit(&mut self) {
        // Don't parallelize this. Memory overhead
        for s in unsafe { self.columns_mut_retain_schema() } {
            s.shrink_to_fit();
        }
    }

    /// Aggregate all the chunks in the DataFrame to a single chunk in parallel.
    /// This may lead to more peak memory consumption.
    pub fn rechunk_mut_par(&mut self) -> &mut Self {
        if self.columns().iter().any(|c| c.n_chunks() > 1) {
            POOL.install(|| {
                unsafe { self.columns_mut_retain_schema() }
                    .par_iter_mut()
                    .for_each(|c| *c = c.rechunk());
            })
        }

        self
    }

    /// Rechunks all columns to only have a single chunk.
    pub fn rechunk_mut(&mut self) -> &mut Self {
        // SAFETY: We never adjust the length or names of the columns.
        let columns = unsafe { self.columns_mut() };

        for col in columns.iter_mut().filter(|c| c.n_chunks() > 1) {
            *col = col.rechunk();
        }

        self
    }

    /// Returns true if the chunks of the columns do not align and re-chunking should be done
    pub fn should_rechunk(&self) -> bool {
        // Fast check. It is also needed for correctness, as code below doesn't check if the number
        // of chunks is equal.
        if !self
            .columns()
            .iter()
            .filter_map(|c| c.as_series().map(|s| s.n_chunks()))
            .all_equal()
        {
            return true;
        }

        // From here we check chunk lengths.
        let mut chunk_lengths = self.materialized_column_iter().map(|s| s.chunk_lengths());
        match chunk_lengths.next() {
            None => false,
            Some(first_column_chunk_lengths) => {
                // Fast Path for single Chunk Series
                if first_column_chunk_lengths.size_hint().0 == 1 {
                    return chunk_lengths.any(|cl| cl.size_hint().0 != 1);
                }
                // Always rechunk if we have more chunks than rows.
                // except when we have an empty df containing a single chunk
                let height = self.height();
                let n_chunks = first_column_chunk_lengths.size_hint().0;
                if n_chunks > height && !(height == 0 && n_chunks == 1) {
                    return true;
                }
                // Slow Path for multi Chunk series
                let v: Vec<_> = first_column_chunk_lengths.collect();
                for cl in chunk_lengths {
                    if cl.enumerate().any(|(idx, el)| Some(&el) != v.get(idx)) {
                        return true;
                    }
                }
                false
            },
        }
    }

    /// Ensure all the chunks in the [`DataFrame`] are aligned.
    pub fn align_chunks_par(&mut self) -> &mut Self {
        if self.should_rechunk() {
            self.rechunk_mut_par()
        } else {
            self
        }
    }

    /// Ensure all the chunks in the [`DataFrame`] are aligned.
    pub fn align_chunks(&mut self) -> &mut Self {
        if self.should_rechunk() {
            self.rechunk_mut()
        } else {
            self
        }
    }

    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Language" => ["Rust", "Python"],
    ///                         "Designer" => ["Graydon Hoare", "Guido van Rossum"])?;
    ///
    /// assert_eq!(df.get_column_names(), &["Language", "Designer"]);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn get_column_names(&self) -> Vec<&PlSmallStr> {
        self.columns().iter().map(|s| s.name()).collect()
    }

    /// Get the [`Vec<PlSmallStr>`] representing the column names.
    pub fn get_column_names_owned(&self) -> Vec<PlSmallStr> {
        self.columns().iter().map(|s| s.name().clone()).collect()
    }

    /// Set the column names.
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let mut df: DataFrame = df!("Mathematical set" => ["ℕ", "ℤ", "𝔻", "ℚ", "ℝ", "ℂ"])?;
    /// df.set_column_names(&["Set"])?;
    ///
    /// assert_eq!(df.get_column_names(), &["Set"]);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn set_column_names<T>(&mut self, new_names: &[T]) -> PolarsResult<()>
    where
        T: AsRef<str>,
    {
        polars_ensure!(
            new_names.len() == self.width(),
            ShapeMismatch: "{} column names provided for a DataFrame of width {}",
            new_names.len(), self.width()
        );

        validation::ensure_names_unique(new_names)?;

        *unsafe { self.columns_mut() } = std::mem::take(unsafe { self.columns_mut() })
            .into_iter()
            .zip(new_names)
            .map(|(c, name)| c.with_name(PlSmallStr::from_str(name.as_ref())))
            .collect();

        Ok(())
    }

    /// Get the data types of the columns in the [`DataFrame`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let venus_air: DataFrame = df!("Element" => ["Carbon dioxide", "Nitrogen"],
    ///                                "Fraction" => [0.965, 0.035])?;
    ///
    /// assert_eq!(venus_air.dtypes(), &[DataType::String, DataType::Float64]);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn dtypes(&self) -> Vec<DataType> {
        self.columns().iter().map(|s| s.dtype().clone()).collect()
    }

    /// The number of chunks for the first column.
    pub fn first_col_n_chunks(&self) -> usize {
        match self.columns().iter().find_map(|col| col.as_series()) {
            None if self.width() == 0 => 0,
            None => 1,
            Some(s) => s.n_chunks(),
        }
    }

    /// The highest number of chunks for any column.
    pub fn max_n_chunks(&self) -> usize {
        self.columns()
            .iter()
            .map(|s| s.as_series().map(|s| s.n_chunks()).unwrap_or(1))
            .max()
            .unwrap_or(0)
    }

    /// Generate the schema fields of the [`DataFrame`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let earth: DataFrame = df!("Surface type" => ["Water", "Land"],
    ///                            "Fraction" => [0.708, 0.292])?;
    ///
    /// let f1: Field = Field::new("Surface type".into(), DataType::String);
    /// let f2: Field = Field::new("Fraction".into(), DataType::Float64);
    ///
    /// assert_eq!(earth.fields(), &[f1, f2]);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn fields(&self) -> Vec<Field> {
        self.columns()
            .iter()
            .map(|s| s.field().into_owned())
            .collect()
    }

    /// Add multiple [`Series`] to a [`DataFrame`].
    /// The added `Series` are required to have the same length.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Element" => ["Copper", "Silver", "Gold"])?;
    /// let s1 = Column::new("Proton".into(), [29, 47, 79]);
    /// let s2 = Column::new("Electron".into(), [29, 47, 79]);
    ///
    /// let df2: DataFrame = df1.hstack(&[s1, s2])?;
    /// assert_eq!(df2.shape(), (3, 3));
    /// println!("{}", df2);
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (3, 3)
    /// +---------+--------+----------+
    /// | Element | Proton | Electron |
    /// | ---     | ---    | ---      |
    /// | str     | i32    | i32      |
    /// +=========+========+==========+
    /// | Copper  | 29     | 29       |
    /// +---------+--------+----------+
    /// | Silver  | 47     | 47       |
    /// +---------+--------+----------+
    /// | Gold    | 79     | 79       |
    /// +---------+--------+----------+
    /// ```
    pub fn hstack(&self, columns: &[Column]) -> PolarsResult<Self> {
        let mut new_cols = Vec::with_capacity(self.width() + columns.len());

        new_cols.extend(self.columns().iter().cloned());
        new_cols.extend_from_slice(columns);

        DataFrame::new(self.height(), new_cols)
    }
    /// Concatenate a [`DataFrame`] to this [`DataFrame`] and return as newly allocated [`DataFrame`].
    ///
    /// If many `vstack` operations are done, it is recommended to call [`DataFrame::align_chunks_par`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Element" => ["Copper", "Silver", "Gold"],
    ///                          "Melting Point (K)" => [1357.77, 1234.93, 1337.33])?;
    /// let df2: DataFrame = df!("Element" => ["Platinum", "Palladium"],
    ///                          "Melting Point (K)" => [2041.4, 1828.05])?;
    ///
    /// let df3: DataFrame = df1.vstack(&df2)?;
    ///
    /// assert_eq!(df3.shape(), (5, 2));
    /// println!("{}", df3);
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (5, 2)
    /// +-----------+-------------------+
    /// | Element   | Melting Point (K) |
    /// | ---       | ---               |
    /// | str       | f64               |
    /// +===========+===================+
    /// | Copper    | 1357.77           |
    /// +-----------+-------------------+
    /// | Silver    | 1234.93           |
    /// +-----------+-------------------+
    /// | Gold      | 1337.33           |
    /// +-----------+-------------------+
    /// | Platinum  | 2041.4            |
    /// +-----------+-------------------+
    /// | Palladium | 1828.05           |
    /// +-----------+-------------------+
    /// ```
    pub fn vstack(&self, other: &DataFrame) -> PolarsResult<Self> {
        let mut df = self.clone();
        df.vstack_mut(other)?;
        Ok(df)
    }

    /// Concatenate a [`DataFrame`] to this [`DataFrame`]
    ///
    /// If many `vstack` operations are done, it is recommended to call [`DataFrame::align_chunks_par`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let mut df1: DataFrame = df!("Element" => ["Copper", "Silver", "Gold"],
    ///                          "Melting Point (K)" => [1357.77, 1234.93, 1337.33])?;
    /// let df2: DataFrame = df!("Element" => ["Platinum", "Palladium"],
    ///                          "Melting Point (K)" => [2041.4, 1828.05])?;
    ///
    /// df1.vstack_mut(&df2)?;
    ///
    /// assert_eq!(df1.shape(), (5, 2));
    /// println!("{}", df1);
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (5, 2)
    /// +-----------+-------------------+
    /// | Element   | Melting Point (K) |
    /// | ---       | ---               |
    /// | str       | f64               |
    /// +===========+===================+
    /// | Copper    | 1357.77           |
    /// +-----------+-------------------+
    /// | Silver    | 1234.93           |
    /// +-----------+-------------------+
    /// | Gold      | 1337.33           |
    /// +-----------+-------------------+
    /// | Platinum  | 2041.4            |
    /// +-----------+-------------------+
    /// | Palladium | 1828.05           |
    /// +-----------+-------------------+
    /// ```
    pub fn vstack_mut(&mut self, other: &DataFrame) -> PolarsResult<&mut Self> {
        if self.width() != other.width() {
            polars_ensure!(
                self.shape() == (0, 0),
                ShapeMismatch:
                "unable to append to a DataFrame of shape {:?} with a DataFrame of width {}",
                self.shape(), other.width(),
            );

            self.clone_from(other);

            return Ok(self);
        }

        let new_height = usize::checked_add(self.height(), other.height()).unwrap();

        unsafe { self.columns_mut_retain_schema() }
            .iter_mut()
            .zip(other.columns())
            .try_for_each::<_, PolarsResult<_>>(|(left, right)| {
                ensure_can_extend(&*left, right)?;
                left.append(right).map_err(|e| {
                    e.context(format!("failed to vstack column '{}'", right.name()).into())
                })?;
                Ok(())
            })?;

        unsafe { self.set_height(new_height) };

        Ok(self)
    }

    pub fn vstack_mut_owned(&mut self, other: DataFrame) -> PolarsResult<&mut Self> {
        if self.width() != other.width() {
            polars_ensure!(
                self.shape() == (0, 0),
                ShapeMismatch:
                "unable to append to a DataFrame of width {} with a DataFrame of width {}",
                self.width(), other.width(),
            );

            *self = other;

            return Ok(self);
        }

        let new_height = usize::checked_add(self.height(), other.height()).unwrap();

        unsafe { self.columns_mut_retain_schema() }
            .iter_mut()
            .zip(other.into_columns())
            .try_for_each::<_, PolarsResult<_>>(|(left, right)| {
                ensure_can_extend(&*left, &right)?;
                let right_name = right.name().clone();
                left.append_owned(right).map_err(|e| {
                    e.context(format!("failed to vstack column '{right_name}'").into())
                })?;
                Ok(())
            })?;

        unsafe { self.set_height(new_height) };

        Ok(self)
    }

    /// Concatenate a [`DataFrame`] to this [`DataFrame`]
    ///
    /// If many `vstack` operations are done, it is recommended to call [`DataFrame::align_chunks_par`].
    ///
    /// # Panics
    /// Panics if the schema's don't match.
    pub fn vstack_mut_unchecked(&mut self, other: &DataFrame) -> &mut Self {
        let new_height = usize::checked_add(self.height(), other.height()).unwrap();

        unsafe { self.columns_mut_retain_schema() }
            .iter_mut()
            .zip(other.columns())
            .for_each(|(left, right)| {
                left.append(right)
                    .map_err(|e| {
                        e.context(format!("failed to vstack column '{}'", right.name()).into())
                    })
                    .expect("should not fail");
            });

        unsafe { self.set_height(new_height) };

        self
    }

    /// Concatenate a [`DataFrame`] to this [`DataFrame`]
    ///
    /// If many `vstack` operations are done, it is recommended to call [`DataFrame::align_chunks_par`].
    ///
    /// # Panics
    /// Panics if the schema's don't match.
    pub fn vstack_mut_owned_unchecked(&mut self, other: DataFrame) -> &mut Self {
        let new_height = usize::checked_add(self.height(), other.height()).unwrap();

        unsafe { self.columns_mut_retain_schema() }
            .iter_mut()
            .zip(other.into_columns())
            .for_each(|(left, right)| {
                left.append_owned(right).expect("should not fail");
            });

        unsafe { self.set_height(new_height) };

        self
    }

    /// Extend the memory backed by this [`DataFrame`] with the values from `other`.
    ///
    /// Different from [`vstack`](Self::vstack) which adds the chunks from `other` to the chunks of this [`DataFrame`]
    /// `extend` appends the data from `other` to the underlying memory locations and thus may cause a reallocation.
    ///
    /// If this does not cause a reallocation, the resulting data structure will not have any extra chunks
    /// and thus will yield faster queries.
    ///
    /// Prefer `extend` over `vstack` when you want to do a query after a single append. For instance during
    /// online operations where you add `n` rows and rerun a query.
    ///
    /// Prefer `vstack` over `extend` when you want to append many times before doing a query. For instance
    /// when you read in multiple files and when to store them in a single `DataFrame`. In the latter case, finish the sequence
    /// of `append` operations with a [`rechunk`](Self::align_chunks_par).
    pub fn extend(&mut self, other: &DataFrame) -> PolarsResult<()> {
        polars_ensure!(
            self.width() == other.width(),
            ShapeMismatch:
            "unable to extend a DataFrame of width {} with a DataFrame of width {}",
            self.width(), other.width(),
        );

        let new_height = usize::checked_add(self.height(), other.height()).unwrap();

        unsafe { self.columns_mut_retain_schema() }
            .iter_mut()
            .zip(other.columns())
            .try_for_each::<_, PolarsResult<_>>(|(left, right)| {
                ensure_can_extend(&*left, right)?;
                left.extend(right).map_err(|e| {
                    e.context(format!("failed to extend column '{}'", right.name()).into())
                })?;
                Ok(())
            })?;

        unsafe { self.set_height(new_height) };

        Ok(())
    }

    /// Remove a column by name and return the column removed.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let mut df: DataFrame = df!("Animal" => ["Tiger", "Lion", "Great auk"],
    ///                             "IUCN" => ["Endangered", "Vulnerable", "Extinct"])?;
    ///
    /// let s1: PolarsResult<Column> = df.drop_in_place("Average weight");
    /// assert!(s1.is_err());
    ///
    /// let s2: Column = df.drop_in_place("Animal")?;
    /// assert_eq!(s2, Column::new("Animal".into(), &["Tiger", "Lion", "Great auk"]));
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn drop_in_place(&mut self, name: &str) -> PolarsResult<Column> {
        let idx = self.try_get_column_index(name)?;
        Ok(unsafe { self.columns_mut() }.remove(idx))
    }

    /// Return a new [`DataFrame`] where all null values are dropped.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Country" => ["Malta", "Liechtenstein", "North Korea"],
    ///                         "Tax revenue (% GDP)" => [Some(32.7), None, None])?;
    /// assert_eq!(df1.shape(), (3, 2));
    ///
    /// let df2: DataFrame = df1.drop_nulls::<String>(None)?;
    /// assert_eq!(df2.shape(), (1, 2));
    /// println!("{}", df2);
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (1, 2)
    /// +---------+---------------------+
    /// | Country | Tax revenue (% GDP) |
    /// | ---     | ---                 |
    /// | str     | f64                 |
    /// +=========+=====================+
    /// | Malta   | 32.7                |
    /// +---------+---------------------+
    /// ```
    pub fn drop_nulls<S>(&self, subset: Option<&[S]>) -> PolarsResult<Self>
    where
        for<'a> &'a S: AsRef<str>,
    {
        if let Some(v) = subset {
            let v = self.select_to_vec(v)?;
            self._drop_nulls_impl(v.as_slice())
        } else {
            self._drop_nulls_impl(self.columns())
        }
    }

    fn _drop_nulls_impl(&self, subset: &[Column]) -> PolarsResult<Self> {
        // fast path for no nulls in df
        if subset.iter().all(|s| !s.has_nulls()) {
            return Ok(self.clone());
        }

        let mut iter = subset.iter();

        let mask = iter
            .next()
            .ok_or_else(|| polars_err!(NoData: "no data to drop nulls from"))?;
        let mut mask = mask.is_not_null();

        for c in iter {
            mask = mask & c.is_not_null();
        }
        self.filter(&mask)
    }

    /// Drop a column by name.
    /// This is a pure method and will return a new [`DataFrame`] instead of modifying
    /// the current one in place.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Ray type" => ["α", "β", "X", "γ"])?;
    /// let df2: DataFrame = df1.drop("Ray type")?;
    ///
    /// assert_eq!(df2.width(), 0);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn drop(&self, name: &str) -> PolarsResult<Self> {
        let idx = self.try_get_column_index(name)?;
        let mut new_cols = Vec::with_capacity(self.width() - 1);

        self.columns().iter().enumerate().for_each(|(i, s)| {
            if i != idx {
                new_cols.push(s.clone())
            }
        });

        Ok(unsafe { DataFrame::_new_unchecked_impl(self.height(), new_cols) })
    }

    /// Drop columns that are in `names`.
    pub fn drop_many<I, S>(&self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<PlSmallStr>,
    {
        let names: PlHashSet<PlSmallStr> = names.into_iter().map(|s| s.into()).collect();
        self.drop_many_amortized(&names)
    }

    /// Drop columns that are in `names` without allocating a [`HashSet`](std::collections::HashSet).
    pub fn drop_many_amortized(&self, names: &PlHashSet<PlSmallStr>) -> DataFrame {
        if names.is_empty() {
            return self.clone();
        }
        let mut new_cols = Vec::with_capacity(self.width().saturating_sub(names.len()));
        self.columns().iter().for_each(|s| {
            if !names.contains(s.name()) {
                new_cols.push(s.clone())
            }
        });

        unsafe { DataFrame::new_unchecked(self.height(), new_cols) }
    }

    /// Insert a new column at a given index without checking for duplicates.
    /// This can leave the [`DataFrame`] at an invalid state
    fn insert_column_no_namecheck(
        &mut self,
        index: usize,
        column: Column,
    ) -> PolarsResult<&mut Self> {
        if self.shape() == (0, 0) {
            unsafe { self.set_height(column.len()) };
        }

        polars_ensure!(
            column.len() == self.height(),
            ShapeMismatch:
            "unable to add a column of length {} to a DataFrame of height {}",
            column.len(), self.height(),
        );

        unsafe { self.columns_mut() }.insert(index, column);
        Ok(self)
    }

    /// Insert a new column at a given index.
    pub fn insert_column(&mut self, index: usize, column: Column) -> PolarsResult<&mut Self> {
        let name = column.name();

        polars_ensure!(
            self.get_column_index(name).is_none(),
            Duplicate:
            "column with name {:?} is already present in the DataFrame", name
        );

        self.insert_column_no_namecheck(index, column)
    }

    /// Add a new column to this [`DataFrame`] or replace an existing one. Broadcasts unit-length
    /// columns.
    pub fn with_column(&mut self, mut column: Column) -> PolarsResult<&mut Self> {
        if self.shape() == (0, 0) {
            unsafe { self.set_height(column.len()) };
        }

        if column.len() != self.height() && column.len() == 1 {
            column = column.new_from_index(0, self.height());
        }

        polars_ensure!(
            column.len() == self.height(),
            ShapeMismatch: "unable to add a column of length {} to a DataFrame of height {}",
            column.len(), self.height(),
        );

        if let Some(i) = self.get_column_index(column.name()) {
            *unsafe { self.columns_mut() }.get_mut(i).unwrap() = column
        } else {
            unsafe { self.columns_mut() }.push(column)
        };

        Ok(self)
    }

    /// Adds a column to the [`DataFrame`] without doing any checks
    /// on length or duplicates.
    ///
    /// # Safety
    /// The caller must ensure `column.len() == self.height()` .
    pub unsafe fn push_column_unchecked(&mut self, column: Column) -> &mut Self {
        unsafe { self.columns_mut() }.push(column);
        self
    }

    /// Add or replace columns to this [`DataFrame`] or replace an existing one.
    /// Broadcasts unit-length columns, and uses an existing schema to amortize lookups.
    pub fn with_columns_mut(
        &mut self,
        columns: impl IntoIterator<Item = Column>,
        output_schema: &Schema,
    ) -> PolarsResult<()> {
        let columns = columns.into_iter();

        unsafe {
            self.columns_mut_retain_schema()
                .reserve(columns.size_hint().0)
        }

        for c in columns {
            self.with_column_and_schema_mut(c, output_schema)?;
        }

        Ok(())
    }

    fn with_column_and_schema_mut(
        &mut self,
        mut column: Column,
        output_schema: &Schema,
    ) -> PolarsResult<&mut Self> {
        if self.shape() == (0, 0) {
            unsafe { self.set_height(column.len()) };
        }

        if column.len() != self.height() && column.len() == 1 {
            column = column.new_from_index(0, self.height());
        }

        polars_ensure!(
            column.len() == self.height(),
            ShapeMismatch:
            "unable to add a column of length {} to a DataFrame of height {}",
            column.len(), self.height(),
        );

        let i = output_schema
            .index_of(column.name())
            .or_else(|| self.get_column_index(column.name()))
            .unwrap_or(self.width());

        if i < self.width() {
            *unsafe { self.columns_mut() }.get_mut(i).unwrap() = column
        } else if i == self.width() {
            unsafe { self.columns_mut() }.push(column)
        } else {
            // Unordered column insertion is not handled.
            panic!()
        }

        Ok(self)
    }

    /// Get a row in the [`DataFrame`]. Beware this is slow.
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// fn example(df: &mut DataFrame, idx: usize) -> Option<Vec<AnyValue>> {
    ///     df.get(idx)
    /// }
    /// ```
    pub fn get(&self, idx: usize) -> Option<Vec<AnyValue<'_>>> {
        (idx < self.height()).then(|| self.columns().iter().map(|c| c.get(idx).unwrap()).collect())
    }

    /// Select a [`Series`] by index.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Star" => ["Sun", "Betelgeuse", "Sirius A", "Sirius B"],
    ///                         "Absolute magnitude" => [4.83, -5.85, 1.42, 11.18])?;
    ///
    /// let s1: Option<&Column> = df.select_at_idx(0);
    /// let s2 = Column::new("Star".into(), ["Sun", "Betelgeuse", "Sirius A", "Sirius B"]);
    ///
    /// assert_eq!(s1, Some(&s2));
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn select_at_idx(&self, idx: usize) -> Option<&Column> {
        self.columns().get(idx)
    }

    /// Get column index of a [`Series`] by name.
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Name" => ["Player 1", "Player 2", "Player 3"],
    ///                         "Health" => [100, 200, 500],
    ///                         "Mana" => [250, 100, 0],
    ///                         "Strength" => [30, 150, 300])?;
    ///
    /// assert_eq!(df.get_column_index("Name"), Some(0));
    /// assert_eq!(df.get_column_index("Health"), Some(1));
    /// assert_eq!(df.get_column_index("Mana"), Some(2));
    /// assert_eq!(df.get_column_index("Strength"), Some(3));
    /// assert_eq!(df.get_column_index("Haste"), None);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn get_column_index(&self, name: &str) -> Option<usize> {
        if let Some(schema) = self.cached_schema() {
            schema.index_of(name)
        } else if self.width() <= LINEAR_SEARCH_LIMIT {
            self.columns().iter().position(|s| s.name() == name)
        } else {
            self.schema().index_of(name)
        }
    }

    /// Get column index of a [`Series`] by name.
    pub fn try_get_column_index(&self, name: &str) -> PolarsResult<usize> {
        self.get_column_index(name)
            .ok_or_else(|| polars_err!(col_not_found = name))
    }

    /// Select a single column by name.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let s1 = Column::new("Password".into(), ["123456", "[]B$u$g$s$B#u#n#n#y[]{}"]);
    /// let s2 = Column::new("Robustness".into(), ["Weak", "Strong"]);
    /// let df: DataFrame = DataFrame::new_infer_height(vec![s1.clone(), s2])?;
    ///
    /// assert_eq!(df.column("Password")?, &s1);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn column(&self, name: &str) -> PolarsResult<&Column> {
        let idx = self.try_get_column_index(name)?;
        Ok(self.select_at_idx(idx).unwrap())
    }

    /// Select column(s) from this [`DataFrame`] and return a new [`DataFrame`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// fn example(df: &DataFrame) -> PolarsResult<DataFrame> {
    ///     df.select(["foo", "bar"])
    /// }
    /// ```
    pub fn select<I, S>(&self, names: I) -> PolarsResult<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        DataFrame::new(self.height(), self.select_to_vec(names)?)
    }

    /// Does not check for duplicates.
    ///
    /// # Safety
    /// `names` must not contain duplicates.
    pub unsafe fn select_unchecked<I, S>(&self, names: I) -> PolarsResult<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Ok(unsafe { DataFrame::new_unchecked(self.height(), self.select_to_vec(names)?) })
    }

    /// Select column(s) from this [`DataFrame`] and return them into a [`Vec`].
    ///
    /// This does not error on duplicate selections.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Name" => ["Methane", "Ethane", "Propane"],
    ///                         "Carbon" => [1, 2, 3],
    ///                         "Hydrogen" => [4, 6, 8])?;
    /// let sv: Vec<Column> = df.select_to_vec(["Carbon", "Hydrogen"])?;
    ///
    /// assert_eq!(df["Carbon"], sv[0]);
    /// assert_eq!(df["Hydrogen"], sv[1]);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn select_to_vec(
        &self,
        selection: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> PolarsResult<Vec<Column>> {
        AmortizedColumnSelector::new(self).select_multiple(selection)
    }

    /// Take the [`DataFrame`] rows by a boolean mask.
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// fn example(df: &DataFrame) -> PolarsResult<DataFrame> {
    ///     let mask = df.column("sepal_width")?.is_not_null();
    ///     df.filter(&mask)
    /// }
    /// ```
    pub fn filter(&self, mask: &BooleanChunked) -> PolarsResult<Self> {
        if self.width() == 0 {
            filter_zero_width(self.height(), mask)
        } else {
            let new_columns: Vec<Column> = self.try_apply_columns_par(|s| s.filter(mask))?;
            let out = unsafe {
                DataFrame::new_unchecked(new_columns[0].len(), new_columns).with_schema_from(self)
            };

            Ok(out)
        }
    }

    /// Same as `filter` but does not parallelize.
    pub fn filter_seq(&self, mask: &BooleanChunked) -> PolarsResult<Self> {
        if self.width() == 0 {
            filter_zero_width(self.height(), mask)
        } else {
            let new_columns: Vec<Column> = self.try_apply_columns(|s| s.filter(mask))?;
            let out = unsafe {
                DataFrame::new_unchecked(new_columns[0].len(), new_columns).with_schema_from(self)
            };

            Ok(out)
        }
    }

    /// Gather [`DataFrame`] rows by index values.
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// fn example(df: &DataFrame) -> PolarsResult<DataFrame> {
    ///     let idx = IdxCa::new("idx".into(), [0, 1, 9]);
    ///     df.take(&idx)
    /// }
    /// ```
    pub fn take(&self, indices: &IdxCa) -> PolarsResult<Self> {
        check_bounds_ca(indices, self.height().try_into().unwrap_or(IdxSize::MAX))?;

        let new_cols = self.apply_columns_par(|c| {
            assert_eq!(c.len(), self.height());
            unsafe { c.take_unchecked(indices) }
        });

        Ok(unsafe { DataFrame::new_unchecked(indices.len(), new_cols).with_schema_from(self) })
    }

    /// # Safety
    /// The indices must be in-bounds.
    pub unsafe fn take_unchecked(&self, idx: &IdxCa) -> Self {
        self.take_unchecked_impl(idx, true)
    }

    /// # Safety
    /// The indices must be in-bounds.
    #[cfg(feature = "algorithm_group_by")]
    pub unsafe fn gather_group_unchecked(&self, group: &GroupsIndicator) -> Self {
        match group {
            GroupsIndicator::Idx((_, indices)) => unsafe {
                self.take_slice_unchecked_impl(indices.as_slice(), false)
            },
            GroupsIndicator::Slice([offset, len]) => self.slice(*offset as i64, *len as usize),
        }
    }

    /// # Safety
    /// The indices must be in-bounds.
    pub unsafe fn take_unchecked_impl(&self, idx: &IdxCa, allow_threads: bool) -> Self {
        let cols = if allow_threads && POOL.current_num_threads() > 1 {
            POOL.install(|| {
                if POOL.current_num_threads() > self.width() {
                    let stride = usize::max(idx.len().div_ceil(POOL.current_num_threads()), 256);
                    if self.height() / stride >= 2 {
                        self.apply_columns_par(|c| {
                            // Nested types initiate a rechunk in their take_unchecked implementation.
                            // If we do not rechunk, it will result in rechunk storms downstream.
                            let c = if c.dtype().is_nested() {
                                &c.rechunk()
                            } else {
                                c
                            };

                            (0..idx.len().div_ceil(stride))
                                .into_par_iter()
                                .map(|i| c.take_unchecked(&idx.slice((i * stride) as i64, stride)))
                                .reduce(
                                    || Column::new_empty(c.name().clone(), c.dtype()),
                                    |mut a, b| {
                                        a.append_owned(b).unwrap();
                                        a
                                    },
                                )
                        })
                    } else {
                        self.apply_columns_par(|c| c.take_unchecked(idx))
                    }
                } else {
                    self.apply_columns_par(|c| c.take_unchecked(idx))
                }
            })
        } else {
            self.apply_columns(|s| s.take_unchecked(idx))
        };

        unsafe { DataFrame::new_unchecked(idx.len(), cols).with_schema_from(self) }
    }

    /// # Safety
    /// The indices must be in-bounds.
    pub unsafe fn take_slice_unchecked(&self, idx: &[IdxSize]) -> Self {
        self.take_slice_unchecked_impl(idx, true)
    }

    /// # Safety
    /// The indices must be in-bounds.
    pub unsafe fn take_slice_unchecked_impl(&self, idx: &[IdxSize], allow_threads: bool) -> Self {
        let cols = if allow_threads && POOL.current_num_threads() > 1 {
            POOL.install(|| {
                if POOL.current_num_threads() > self.width() {
                    let stride = usize::max(idx.len().div_ceil(POOL.current_num_threads()), 256);
                    if self.height() / stride >= 2 {
                        self.apply_columns_par(|c| {
                            // Nested types initiate a rechunk in their take_unchecked implementation.
                            // If we do not rechunk, it will result in rechunk storms downstream.
                            let c = if c.dtype().is_nested() {
                                &c.rechunk()
                            } else {
                                c
                            };

                            (0..idx.len().div_ceil(stride))
                                .into_par_iter()
                                .map(|i| {
                                    let idx = &idx[i * stride..];
                                    let idx = &idx[..idx.len().min(stride)];
                                    c.take_slice_unchecked(idx)
                                })
                                .reduce(
                                    || Column::new_empty(c.name().clone(), c.dtype()),
                                    |mut a, b| {
                                        a.append_owned(b).unwrap();
                                        a
                                    },
                                )
                        })
                    } else {
                        self.apply_columns_par(|s| s.take_slice_unchecked(idx))
                    }
                } else {
                    self.apply_columns_par(|s| s.take_slice_unchecked(idx))
                }
            })
        } else {
            self.apply_columns(|s| s.take_slice_unchecked(idx))
        };
        unsafe { DataFrame::new_unchecked(idx.len(), cols).with_schema_from(self) }
    }

    /// Rename a column in the [`DataFrame`].
    ///
    /// Should not be called in a loop as that can lead to quadratic behavior.
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// fn example(df: &mut DataFrame) -> PolarsResult<&mut DataFrame> {
    ///     let original_name = "foo";
    ///     let new_name = "bar";
    ///     df.rename(original_name, new_name.into())
    /// }
    /// ```
    pub fn rename(&mut self, column: &str, name: PlSmallStr) -> PolarsResult<&mut Self> {
        if column == name.as_str() {
            return Ok(self);
        }
        polars_ensure!(
            !self.schema().contains(&name),
            Duplicate: "column rename attempted with already existing name \"{name}\""
        );

        self.get_column_index(column)
            .and_then(|idx| unsafe { self.columns_mut() }.get_mut(idx))
            .ok_or_else(|| polars_err!(col_not_found = column))
            .map(|c| c.rename(name))?;

        Ok(self)
    }

    pub fn rename_many<'a>(
        &mut self,
        renames: impl Iterator<Item = (&'a str, PlSmallStr)>,
    ) -> PolarsResult<&mut Self> {
        let mut schema_arc = self.schema().clone();
        let schema = Arc::make_mut(&mut schema_arc);

        for (from, to) in renames {
            if from == to.as_str() {
                continue;
            }

            polars_ensure!(
                !schema.contains(&to),
                Duplicate: "column rename attempted with already existing name \"{to}\""
            );

            match schema.get_full(from) {
                None => polars_bail!(col_not_found = from),
                Some((idx, _, _)) => {
                    let (n, _) = schema.get_at_index_mut(idx).unwrap();
                    *n = to.clone();
                    unsafe { self.columns_mut() }
                        .get_mut(idx)
                        .unwrap()
                        .rename(to);
                },
            }
        }

        unsafe { self.set_schema(schema_arc) };

        Ok(self)
    }

    /// Sort [`DataFrame`] in place.
    ///
    /// See [`DataFrame::sort`] for more instruction.
    pub fn sort_in_place(
        &mut self,
        by: impl IntoIterator<Item = impl AsRef<str>>,
        sort_options: SortMultipleOptions,
    ) -> PolarsResult<&mut Self> {
        let by_column = self.select_to_vec(by)?;

        let mut out = self.sort_impl(by_column, sort_options, None)?;
        unsafe { out.set_schema_from(self) };

        *self = out;

        Ok(self)
    }

    #[doc(hidden)]
    /// This is the dispatch of Self::sort, and exists to reduce compile bloat by monomorphization.
    pub fn sort_impl(
        &self,
        by_column: Vec<Column>,
        sort_options: SortMultipleOptions,
        slice: Option<(i64, usize)>,
    ) -> PolarsResult<Self> {
        if by_column.is_empty() {
            // If no columns selected, any order (including original order) is correct.
            return if let Some((offset, len)) = slice {
                Ok(self.slice(offset, len))
            } else {
                Ok(self.clone())
            };
        }

        for column in &by_column {
            if column.dtype().is_object() {
                polars_bail!(
                    InvalidOperation: "column '{}' has a dtype of '{}', which does not support sorting", column.name(), column.dtype()
                )
            }
        }

        // note that the by_column argument also contains evaluated expression from
        // polars-lazy that may not even be present in this dataframe. therefore
        // when we try to set the first columns as sorted, we ignore the error as
        // expressions are not present (they are renamed to _POLARS_SORT_COLUMN_i.
        let first_descending = sort_options.descending[0];
        let first_by_column = by_column[0].name().to_string();

        let set_sorted = |df: &mut DataFrame| {
            // Mark the first sort column as sorted; if the column does not exist it
            // is ok, because we sorted by an expression not present in the dataframe
            let _ = df.apply(&first_by_column, |s| {
                let mut s = s.clone();
                if first_descending {
                    s.set_sorted_flag(IsSorted::Descending)
                } else {
                    s.set_sorted_flag(IsSorted::Ascending)
                }
                s
            });
        };

        if self.shape_has_zero() {
            let mut out = self.clone();
            set_sorted(&mut out);
            return Ok(out);
        }

        if let Some((0, k)) = slice {
            if k < self.height() {
                return self.bottom_k_impl(k, by_column, sort_options);
            }
        }
        // Check if the required column is already sorted; if so we can exit early
        // We can do so when there is only one column to sort by, for multiple columns
        // it will be complicated to do so
        #[cfg(feature = "dtype-categorical")]
        let is_not_categorical_enum =
            !(matches!(by_column[0].dtype(), DataType::Categorical(_, _))
                || matches!(by_column[0].dtype(), DataType::Enum(_, _)));

        #[cfg(not(feature = "dtype-categorical"))]
        #[allow(non_upper_case_globals)]
        const is_not_categorical_enum: bool = true;

        if by_column.len() == 1 && is_not_categorical_enum {
            let required_sorting = if sort_options.descending[0] {
                IsSorted::Descending
            } else {
                IsSorted::Ascending
            };
            // If null count is 0 then nulls_last doesnt matter
            // Safe to get value at last position since the dataframe is not empty (taken care above)
            let no_sorting_required = (by_column[0].is_sorted_flag() == required_sorting)
                && ((by_column[0].null_count() == 0)
                    || by_column[0].get(by_column[0].len() - 1).unwrap().is_null()
                        == sort_options.nulls_last[0]);

            if no_sorting_required {
                return if let Some((offset, len)) = slice {
                    Ok(self.slice(offset, len))
                } else {
                    Ok(self.clone())
                };
            }
        }

        let has_nested = by_column.iter().any(|s| s.dtype().is_nested());
        let allow_threads = sort_options.multithreaded;

        // a lot of indirection in both sorting and take
        let mut df = self.clone();
        let df = df.rechunk_mut_par();
        let mut take = match (by_column.len(), has_nested) {
            (1, false) => {
                let s = &by_column[0];
                let options = SortOptions {
                    descending: sort_options.descending[0],
                    nulls_last: sort_options.nulls_last[0],
                    multithreaded: sort_options.multithreaded,
                    maintain_order: sort_options.maintain_order,
                    limit: sort_options.limit,
                };
                // fast path for a frame with a single series
                // no need to compute the sort indices and then take by these indices
                // simply sort and return as frame
                if df.width() == 1 && df.try_get_column_index(s.name().as_str()).is_ok() {
                    let mut out = s.sort_with(options)?;
                    if let Some((offset, len)) = slice {
                        out = out.slice(offset, len);
                    }
                    return Ok(out.into_frame());
                }
                s.arg_sort(options)
            },
            _ => arg_sort(&by_column, sort_options)?,
        };

        if let Some((offset, len)) = slice {
            take = take.slice(offset, len);
        }

        // SAFETY:
        // the created indices are in bounds
        let mut df = unsafe { df.take_unchecked_impl(&take, allow_threads) };
        set_sorted(&mut df);
        Ok(df)
    }

    /// Create a `DataFrame` that has fields for all the known runtime metadata for each column.
    ///
    /// This dataframe does not necessarily have a specified schema and may be changed at any
    /// point. It is primarily used for debugging.
    pub fn _to_metadata(&self) -> DataFrame {
        let num_columns = self.width();

        let mut column_names =
            StringChunkedBuilder::new(PlSmallStr::from_static("column_name"), num_columns);
        let mut repr_ca = StringChunkedBuilder::new(PlSmallStr::from_static("repr"), num_columns);
        let mut sorted_asc_ca =
            BooleanChunkedBuilder::new(PlSmallStr::from_static("sorted_asc"), num_columns);
        let mut sorted_dsc_ca =
            BooleanChunkedBuilder::new(PlSmallStr::from_static("sorted_dsc"), num_columns);
        let mut fast_explode_list_ca =
            BooleanChunkedBuilder::new(PlSmallStr::from_static("fast_explode_list"), num_columns);
        let mut materialized_at_ca =
            StringChunkedBuilder::new(PlSmallStr::from_static("materialized_at"), num_columns);

        for col in self.columns() {
            let flags = col.get_flags();

            let (repr, materialized_at) = match col {
                Column::Series(s) => ("series", s.materialized_at()),
                Column::Scalar(_) => ("scalar", None),
            };
            let sorted_asc = flags.contains(StatisticsFlags::IS_SORTED_ASC);
            let sorted_dsc = flags.contains(StatisticsFlags::IS_SORTED_DSC);
            let fast_explode_list = flags.contains(StatisticsFlags::CAN_FAST_EXPLODE_LIST);

            column_names.append_value(col.name().clone());
            repr_ca.append_value(repr);
            sorted_asc_ca.append_value(sorted_asc);
            sorted_dsc_ca.append_value(sorted_dsc);
            fast_explode_list_ca.append_value(fast_explode_list);
            materialized_at_ca.append_option(materialized_at.map(|v| format!("{v:#?}")));
        }

        unsafe {
            DataFrame::new_unchecked(
                self.width(),
                vec![
                    column_names.finish().into_column(),
                    repr_ca.finish().into_column(),
                    sorted_asc_ca.finish().into_column(),
                    sorted_dsc_ca.finish().into_column(),
                    fast_explode_list_ca.finish().into_column(),
                    materialized_at_ca.finish().into_column(),
                ],
            )
        }
    }
    /// Return a sorted clone of this [`DataFrame`].
    ///
    /// In many cases the output chunks will be continuous in memory but this is not guaranteed
    /// # Example
    ///
    /// Sort by a single column with default options:
    /// ```
    /// # use polars_core::prelude::*;
    /// fn sort_by_sepal_width(df: &DataFrame) -> PolarsResult<DataFrame> {
    ///     df.sort(["sepal_width"], Default::default())
    /// }
    /// ```
    /// Sort by a single column with specific order:
    /// ```
    /// # use polars_core::prelude::*;
    /// fn sort_with_specific_order(df: &DataFrame, descending: bool) -> PolarsResult<DataFrame> {
    ///     df.sort(
    ///         ["sepal_width"],
    ///         SortMultipleOptions::new()
    ///             .with_order_descending(descending)
    ///     )
    /// }
    /// ```
    /// Sort by multiple columns with specifying order for each column:
    /// ```
    /// # use polars_core::prelude::*;
    /// fn sort_by_multiple_columns_with_specific_order(df: &DataFrame) -> PolarsResult<DataFrame> {
    ///     df.sort(
    ///         ["sepal_width", "sepal_length"],
    ///         SortMultipleOptions::new()
    ///             .with_order_descending_multi([false, true])
    ///     )
    /// }
    /// ```
    /// See [`SortMultipleOptions`] for more options.
    ///
    /// Also see [`DataFrame::sort_in_place`].
    pub fn sort(
        &self,
        by: impl IntoIterator<Item = impl AsRef<str>>,
        sort_options: SortMultipleOptions,
    ) -> PolarsResult<Self> {
        let mut df = self.clone();
        df.sort_in_place(by, sort_options)?;
        Ok(df)
    }

    /// Replace a column with a [`Column`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let mut df: DataFrame = df!("Country" => ["United States", "China"],
    ///                         "Area (km²)" => [9_833_520, 9_596_961])?;
    /// let s: Column = Column::new("Country".into(), ["USA", "PRC"]);
    ///
    /// assert!(df.replace("Nation", s.clone()).is_err());
    /// assert!(df.replace("Country", s).is_ok());
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn replace(&mut self, column: &str, new_col: Column) -> PolarsResult<&mut Self> {
        self.apply(column, |_| new_col)
    }

    /// Replace column at index `idx` with a [`Series`].
    ///
    /// # Example
    ///
    /// ```ignored
    /// # use polars_core::prelude::*;
    /// let s0 = Series::new("foo".into(), ["ham", "spam", "egg"]);
    /// let s1 = Series::new("ascii".into(), [70, 79, 79]);
    /// let mut df = DataFrame::new_infer_height(vec![s0, s1])?;
    ///
    /// // Add 32 to get lowercase ascii values
    /// df.replace_column(1, df.select_at_idx(1).unwrap() + 32);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn replace_column(&mut self, index: usize, new_column: Column) -> PolarsResult<&mut Self> {
        polars_ensure!(
            index < self.width(),
            ShapeMismatch:
            "unable to replace at index {}, the DataFrame has only {} columns",
            index, self.width(),
        );

        polars_ensure!(
            new_column.len() == self.height(),
            ShapeMismatch:
            "unable to replace a column, series length {} doesn't match the DataFrame height {}",
            new_column.len(), self.height(),
        );

        unsafe { *self.columns_mut().get_mut(index).unwrap() = new_column };

        Ok(self)
    }

    /// Apply a closure to a column. This is the recommended way to do in place modification.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let s0 = Column::new("foo".into(), ["ham", "spam", "egg"]);
    /// let s1 = Column::new("names".into(), ["Jean", "Claude", "van"]);
    /// let mut df = DataFrame::new_infer_height(vec![s0, s1])?;
    ///
    /// fn str_to_len(str_val: &Column) -> Column {
    ///     str_val.str()
    ///         .unwrap()
    ///         .into_iter()
    ///         .map(|opt_name: Option<&str>| {
    ///             opt_name.map(|name: &str| name.len() as u32)
    ///          })
    ///         .collect::<UInt32Chunked>()
    ///         .into_column()
    /// }
    ///
    /// // Replace the names column by the length of the names.
    /// df.apply("names", str_to_len);
    /// # Ok::<(), PolarsError>(())
    /// ```
    /// Results in:
    ///
    /// ```text
    /// +--------+-------+
    /// | foo    |       |
    /// | ---    | names |
    /// | str    | u32   |
    /// +========+=======+
    /// | "ham"  | 4     |
    /// +--------+-------+
    /// | "spam" | 6     |
    /// +--------+-------+
    /// | "egg"  | 3     |
    /// +--------+-------+
    /// ```
    pub fn apply<F, C>(&mut self, name: &str, f: F) -> PolarsResult<&mut Self>
    where
        F: FnOnce(&Column) -> C,
        C: IntoColumn,
    {
        let idx = self.try_get_column_index(name)?;
        self.apply_at_idx(idx, f)?;
        Ok(self)
    }

    /// Apply a closure to a column at index `idx`. This is the recommended way to do in place
    /// modification.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let s0 = Column::new("foo".into(), ["ham", "spam", "egg"]);
    /// let s1 = Column::new("ascii".into(), [70, 79, 79]);
    /// let mut df = DataFrame::new_infer_height(vec![s0, s1])?;
    ///
    /// // Add 32 to get lowercase ascii values
    /// df.apply_at_idx(1, |s| s + 32);
    /// # Ok::<(), PolarsError>(())
    /// ```
    /// Results in:
    ///
    /// ```text
    /// +--------+-------+
    /// | foo    | ascii |
    /// | ---    | ---   |
    /// | str    | i32   |
    /// +========+=======+
    /// | "ham"  | 102   |
    /// +--------+-------+
    /// | "spam" | 111   |
    /// +--------+-------+
    /// | "egg"  | 111   |
    /// +--------+-------+
    /// ```
    pub fn apply_at_idx<F, C>(&mut self, idx: usize, f: F) -> PolarsResult<&mut Self>
    where
        F: FnOnce(&Column) -> C,
        C: IntoColumn,
    {
        let df_height = self.height();
        let width = self.width();

        let cached_schema = self.cached_schema().cloned();

        let col = unsafe { self.columns_mut() }.get_mut(idx).ok_or_else(|| {
            polars_err!(
                ComputeError: "invalid column index: {} for a DataFrame with {} columns",
                idx, width
            )
        })?;

        let mut new_col = f(col).into_column();

        if new_col.len() != df_height && new_col.len() == 1 {
            new_col = new_col.new_from_index(0, df_height);
        }

        polars_ensure!(
            new_col.len() == df_height,
            ShapeMismatch:
            "apply_at_idx: resulting Series has length {} while the DataFrame has height {}",
            new_col.len(), df_height
        );

        new_col = new_col.with_name(col.name().clone());
        let col_before = std::mem::replace(col, new_col);

        if col.dtype() == col_before.dtype() {
            unsafe { self.set_opt_schema(cached_schema) };
        }

        Ok(self)
    }

    /// Apply a closure that may fail to a column at index `idx`. This is the recommended way to do in place
    /// modification.
    ///
    /// # Example
    ///
    /// This is the idiomatic way to replace some values a column of a `DataFrame` given range of indexes.
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let s0 = Column::new("foo".into(), ["ham", "spam", "egg", "bacon", "quack"]);
    /// let s1 = Column::new("values".into(), [1, 2, 3, 4, 5]);
    /// let mut df = DataFrame::new_infer_height(vec![s0, s1])?;
    ///
    /// let idx = vec![0, 1, 4];
    ///
    /// df.try_apply("foo", |c| {
    ///     c.str()?
    ///     .scatter_with(idx, |opt_val| opt_val.map(|string| format!("{}-is-modified", string)))
    /// });
    /// # Ok::<(), PolarsError>(())
    /// ```
    /// Results in:
    ///
    /// ```text
    /// +---------------------+--------+
    /// | foo                 | values |
    /// | ---                 | ---    |
    /// | str                 | i32    |
    /// +=====================+========+
    /// | "ham-is-modified"   | 1      |
    /// +---------------------+--------+
    /// | "spam-is-modified"  | 2      |
    /// +---------------------+--------+
    /// | "egg"               | 3      |
    /// +---------------------+--------+
    /// | "bacon"             | 4      |
    /// +---------------------+--------+
    /// | "quack-is-modified" | 5      |
    /// +---------------------+--------+
    /// ```
    pub fn try_apply_at_idx<F, C>(&mut self, idx: usize, f: F) -> PolarsResult<&mut Self>
    where
        F: FnOnce(&Column) -> PolarsResult<C>,
        C: IntoColumn,
    {
        let df_height = self.height();
        let width = self.width();

        let cached_schema = self.cached_schema().cloned();

        let col = unsafe { self.columns_mut() }.get_mut(idx).ok_or_else(|| {
            polars_err!(
                ComputeError: "invalid column index: {} for a DataFrame with {} columns",
                idx, width
            )
        })?;

        let mut new_col = f(col).map(|c| c.into_column())?;

        polars_ensure!(
            new_col.len() == df_height,
            ShapeMismatch:
            "try_apply_at_idx: resulting Series has length {} while the DataFrame has height {}",
            new_col.len(), df_height
        );

        // make sure the name remains the same after applying the closure
        new_col = new_col.with_name(col.name().clone());
        let col_before = std::mem::replace(col, new_col);

        if col.dtype() == col_before.dtype() {
            unsafe { self.set_opt_schema(cached_schema) };
        }

        Ok(self)
    }

    /// Apply a closure that may fail to a column. This is the recommended way to do in place
    /// modification.
    ///
    /// # Example
    ///
    /// This is the idiomatic way to replace some values a column of a `DataFrame` given a boolean mask.
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let s0 = Column::new("foo".into(), ["ham", "spam", "egg", "bacon", "quack"]);
    /// let s1 = Column::new("values".into(), [1, 2, 3, 4, 5]);
    /// let mut df = DataFrame::new_infer_height(vec![s0, s1])?;
    ///
    /// // create a mask
    /// let values = df.column("values")?.as_materialized_series();
    /// let mask = values.lt_eq(1)? | values.gt_eq(5_i32)?;
    ///
    /// df.try_apply("foo", |c| {
    ///     c.str()?
    ///     .set(&mask, Some("not_within_bounds"))
    /// });
    /// # Ok::<(), PolarsError>(())
    /// ```
    /// Results in:
    ///
    /// ```text
    /// +---------------------+--------+
    /// | foo                 | values |
    /// | ---                 | ---    |
    /// | str                 | i32    |
    /// +=====================+========+
    /// | "not_within_bounds" | 1      |
    /// +---------------------+--------+
    /// | "spam"              | 2      |
    /// +---------------------+--------+
    /// | "egg"               | 3      |
    /// +---------------------+--------+
    /// | "bacon"             | 4      |
    /// +---------------------+--------+
    /// | "not_within_bounds" | 5      |
    /// +---------------------+--------+
    /// ```
    pub fn try_apply<F, C>(&mut self, column: &str, f: F) -> PolarsResult<&mut Self>
    where
        F: FnOnce(&Series) -> PolarsResult<C>,
        C: IntoColumn,
    {
        let idx = self.try_get_column_index(column)?;
        self.try_apply_at_idx(idx, |c| f(c.as_materialized_series()))
    }

    /// Slice the [`DataFrame`] along the rows.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Fruit" => ["Apple", "Grape", "Grape", "Fig", "Fig"],
    ///                         "Color" => ["Green", "Red", "White", "White", "Red"])?;
    /// let sl: DataFrame = df.slice(2, 3);
    ///
    /// assert_eq!(sl.shape(), (3, 2));
    /// println!("{}", sl);
    /// # Ok::<(), PolarsError>(())
    /// ```
    /// Output:
    /// ```text
    /// shape: (3, 2)
    /// +-------+-------+
    /// | Fruit | Color |
    /// | ---   | ---   |
    /// | str   | str   |
    /// +=======+=======+
    /// | Grape | White |
    /// +-------+-------+
    /// | Fig   | White |
    /// +-------+-------+
    /// | Fig   | Red   |
    /// +-------+-------+
    /// ```
    #[must_use]
    pub fn slice(&self, offset: i64, length: usize) -> Self {
        if offset == 0 && length == self.height() {
            return self.clone();
        }

        if length == 0 {
            return self.clear();
        }

        let cols = self.apply_columns(|s| s.slice(offset, length));

        let height = if let Some(fst) = cols.first() {
            fst.len()
        } else {
            let (_, length) = slice_offsets(offset, length, self.height());
            length
        };

        unsafe { DataFrame::_new_unchecked_impl(height, cols).with_schema_from(self) }
    }

    /// Split [`DataFrame`] at the given `offset`.
    pub fn split_at(&self, offset: i64) -> (Self, Self) {
        let (a, b) = self.columns().iter().map(|s| s.split_at(offset)).unzip();

        let (idx, _) = slice_offsets(offset, 0, self.height());

        let a = unsafe { DataFrame::new_unchecked(idx, a).with_schema_from(self) };
        let b = unsafe { DataFrame::new_unchecked(self.height() - idx, b).with_schema_from(self) };
        (a, b)
    }

    #[must_use]
    pub fn clear(&self) -> Self {
        let cols = self.columns().iter().map(|s| s.clear()).collect::<Vec<_>>();
        unsafe { DataFrame::_new_unchecked_impl(0, cols).with_schema_from(self) }
    }

    #[must_use]
    pub fn slice_par(&self, offset: i64, length: usize) -> Self {
        if offset == 0 && length == self.height() {
            return self.clone();
        }
        let columns = self.apply_columns_par(|s| s.slice(offset, length));
        unsafe { DataFrame::new_unchecked(length, columns).with_schema_from(self) }
    }

    #[must_use]
    pub fn _slice_and_realloc(&self, offset: i64, length: usize) -> Self {
        if offset == 0 && length == self.height() {
            return self.clone();
        }
        // @scalar-opt
        let columns = self.apply_columns(|s| {
            let mut out = s.slice(offset, length);
            out.shrink_to_fit();
            out
        });
        unsafe { DataFrame::new_unchecked(length, columns).with_schema_from(self) }
    }

    /// Get the head of the [`DataFrame`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let countries: DataFrame =
    ///     df!("Rank by GDP (2021)" => [1, 2, 3, 4, 5],
    ///         "Continent" => ["North America", "Asia", "Asia", "Europe", "Europe"],
    ///         "Country" => ["United States", "China", "Japan", "Germany", "United Kingdom"],
    ///         "Capital" => ["Washington", "Beijing", "Tokyo", "Berlin", "London"])?;
    /// assert_eq!(countries.shape(), (5, 4));
    ///
    /// println!("{}", countries.head(Some(3)));
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (3, 4)
    /// +--------------------+---------------+---------------+------------+
    /// | Rank by GDP (2021) | Continent     | Country       | Capital    |
    /// | ---                | ---           | ---           | ---        |
    /// | i32                | str           | str           | str        |
    /// +====================+===============+===============+============+
    /// | 1                  | North America | United States | Washington |
    /// +--------------------+---------------+---------------+------------+
    /// | 2                  | Asia          | China         | Beijing    |
    /// +--------------------+---------------+---------------+------------+
    /// | 3                  | Asia          | Japan         | Tokyo      |
    /// +--------------------+---------------+---------------+------------+
    /// ```
    #[must_use]
    pub fn head(&self, length: Option<usize>) -> Self {
        let new_height = usize::min(self.height(), length.unwrap_or(HEAD_DEFAULT_LENGTH));
        let new_cols = self.apply_columns(|c| c.head(Some(new_height)));

        unsafe { DataFrame::new_unchecked(new_height, new_cols).with_schema_from(self) }
    }

    /// Get the tail of the [`DataFrame`].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let countries: DataFrame =
    ///     df!("Rank (2021)" => [105, 106, 107, 108, 109],
    ///         "Apple Price (€/kg)" => [0.75, 0.70, 0.70, 0.65, 0.52],
    ///         "Country" => ["Kosovo", "Moldova", "North Macedonia", "Syria", "Turkey"])?;
    /// assert_eq!(countries.shape(), (5, 3));
    ///
    /// println!("{}", countries.tail(Some(2)));
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (2, 3)
    /// +-------------+--------------------+---------+
    /// | Rank (2021) | Apple Price (€/kg) | Country |
    /// | ---         | ---                | ---     |
    /// | i32         | f64                | str     |
    /// +=============+====================+=========+
    /// | 108         | 0.65               | Syria   |
    /// +-------------+--------------------+---------+
    /// | 109         | 0.52               | Turkey  |
    /// +-------------+--------------------+---------+
    /// ```
    #[must_use]
    pub fn tail(&self, length: Option<usize>) -> Self {
        let new_height = usize::min(self.height(), length.unwrap_or(TAIL_DEFAULT_LENGTH));
        let new_cols = self.apply_columns(|c| c.tail(Some(new_height)));

        unsafe { DataFrame::new_unchecked(new_height, new_cols).with_schema_from(self) }
    }

    /// Iterator over the rows in this [`DataFrame`] as Arrow RecordBatches.
    ///
    /// # Panics
    ///
    /// Panics if the [`DataFrame`] that is passed is not rechunked.
    ///
    /// This responsibility is left to the caller as we don't want to take mutable references here,
    /// but we also don't want to rechunk here, as this operation is costly and would benefit the caller
    /// as well.
    pub fn iter_chunks(
        &self,
        compat_level: CompatLevel,
        parallel: bool,
    ) -> impl Iterator<Item = RecordBatch> + '_ {
        debug_assert!(!self.should_rechunk(), "expected equal chunks");

        if self.width() == 0 {
            return RecordBatchIterWrap::new_zero_width(self.height());
        }

        // If any of the columns is binview and we don't convert `compat_level` we allow parallelism
        // as we must allocate arrow strings/binaries.
        let must_convert = compat_level.0 == 0;
        let parallel = parallel
            && must_convert
            && self.width() > 1
            && self
                .columns()
                .iter()
                .any(|s| matches!(s.dtype(), DataType::String | DataType::Binary));

        RecordBatchIterWrap::Batches(RecordBatchIter {
            df: self,
            schema: Arc::new(
                self.columns()
                    .iter()
                    .map(|c| c.field().to_arrow(compat_level))
                    .collect(),
            ),
            idx: 0,
            n_chunks: usize::max(1, self.first_col_n_chunks()),
            compat_level,
            parallel,
        })
    }

    /// Iterator over the rows in this [`DataFrame`] as Arrow RecordBatches as physical values.
    ///
    /// # Panics
    ///
    /// Panics if the [`DataFrame`] that is passed is not rechunked.
    ///
    /// This responsibility is left to the caller as we don't want to take mutable references here,
    /// but we also don't want to rechunk here, as this operation is costly and would benefit the caller
    /// as well.
    pub fn iter_chunks_physical(&self) -> impl Iterator<Item = RecordBatch> + '_ {
        debug_assert!(!self.should_rechunk());

        if self.width() == 0 {
            return RecordBatchIterWrap::new_zero_width(self.height());
        }

        RecordBatchIterWrap::PhysicalBatches(PhysRecordBatchIter {
            schema: Arc::new(
                self.columns()
                    .iter()
                    .map(|c| c.field().to_arrow(CompatLevel::newest()))
                    .collect(),
            ),
            arr_iters: self
                .materialized_column_iter()
                .map(|s| s.chunks().iter())
                .collect(),
        })
    }

    /// Get a [`DataFrame`] with all the columns in reversed order.
    #[must_use]
    pub fn reverse(&self) -> Self {
        let new_cols = self.apply_columns(Column::reverse);
        unsafe { DataFrame::new_unchecked(self.height(), new_cols).with_schema_from(self) }
    }

    /// Shift the values by a given period and fill the parts that will be empty due to this operation
    /// with `Nones`.
    ///
    /// See the method on [Series](crate::series::SeriesTrait::shift) for more info on the `shift` operation.
    #[must_use]
    pub fn shift(&self, periods: i64) -> Self {
        let col = self.apply_columns_par(|s| s.shift(periods));
        unsafe { DataFrame::new_unchecked(self.height(), col).with_schema_from(self) }
    }

    /// Replace None values with one of the following strategies:
    /// * Forward fill (replace None with the previous value)
    /// * Backward fill (replace None with the next value)
    /// * Mean fill (replace None with the mean of the whole array)
    /// * Min fill (replace None with the minimum of the whole array)
    /// * Max fill (replace None with the maximum of the whole array)
    ///
    /// See the method on [Series](crate::series::Series::fill_null) for more info on the `fill_null` operation.
    pub fn fill_null(&self, strategy: FillNullStrategy) -> PolarsResult<Self> {
        let col = self.try_apply_columns_par(|s| s.fill_null(strategy))?;

        Ok(unsafe { DataFrame::new_unchecked(self.height(), col) })
    }

    /// Pipe different functions/ closure operations that work on a DataFrame together.
    pub fn pipe<F, B>(self, f: F) -> PolarsResult<B>
    where
        F: Fn(DataFrame) -> PolarsResult<B>,
    {
        f(self)
    }

    /// Pipe different functions/ closure operations that work on a DataFrame together.
    pub fn pipe_mut<F, B>(&mut self, f: F) -> PolarsResult<B>
    where
        F: Fn(&mut DataFrame) -> PolarsResult<B>,
    {
        f(self)
    }

    /// Pipe different functions/ closure operations that work on a DataFrame together.
    pub fn pipe_with_args<F, B, Args>(self, f: F, args: Args) -> PolarsResult<B>
    where
        F: Fn(DataFrame, Args) -> PolarsResult<B>,
    {
        f(self, args)
    }
    /// Drop duplicate rows from a [`DataFrame`].
    /// *This fails when there is a column of type List in DataFrame*
    ///
    /// Stable means that the order is maintained. This has a higher cost than an unstable distinct.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df = df! {
    ///               "flt" => [1., 1., 2., 2., 3., 3.],
    ///               "int" => [1, 1, 2, 2, 3, 3, ],
    ///               "str" => ["a", "a", "b", "b", "c", "c"]
    ///           }?;
    ///
    /// println!("{}", df.unique_stable(None, UniqueKeepStrategy::First, None)?);
    /// # Ok::<(), PolarsError>(())
    /// ```
    /// Returns
    ///
    /// ```text
    /// +-----+-----+-----+
    /// | flt | int | str |
    /// | --- | --- | --- |
    /// | f64 | i32 | str |
    /// +=====+=====+=====+
    /// | 1   | 1   | "a" |
    /// +-----+-----+-----+
    /// | 2   | 2   | "b" |
    /// +-----+-----+-----+
    /// | 3   | 3   | "c" |
    /// +-----+-----+-----+
    /// ```
    #[cfg(feature = "algorithm_group_by")]
    pub fn unique_stable(
        &self,
        subset: Option<&[String]>,
        keep: UniqueKeepStrategy,
        slice: Option<(i64, usize)>,
    ) -> PolarsResult<DataFrame> {
        self.unique_impl(
            true,
            subset.map(|v| v.iter().map(|x| PlSmallStr::from_str(x.as_str())).collect()),
            keep,
            slice,
        )
    }

    /// Unstable distinct. See [`DataFrame::unique_stable`].
    #[cfg(feature = "algorithm_group_by")]
    pub fn unique<I, S>(
        &self,
        subset: Option<&[String]>,
        keep: UniqueKeepStrategy,
        slice: Option<(i64, usize)>,
    ) -> PolarsResult<DataFrame> {
        self.unique_impl(
            false,
            subset.map(|v| v.iter().map(|x| PlSmallStr::from_str(x.as_str())).collect()),
            keep,
            slice,
        )
    }

    #[cfg(feature = "algorithm_group_by")]
    pub fn unique_impl(
        &self,
        maintain_order: bool,
        subset: Option<Vec<PlSmallStr>>,
        keep: UniqueKeepStrategy,
        slice: Option<(i64, usize)>,
    ) -> PolarsResult<Self> {
        if self.width() == 0 {
            let height = usize::min(self.height(), 1);
            return Ok(DataFrame::empty_with_height(height));
        }

        let names = subset.unwrap_or_else(|| self.get_column_names_owned());
        let mut df = self.clone();
        // take on multiple chunks is terrible
        df.rechunk_mut_par();

        let columns = match (keep, maintain_order) {
            (UniqueKeepStrategy::First | UniqueKeepStrategy::Any, true) => {
                let gb = df.group_by_stable(names)?;
                let groups = gb.get_groups();
                let (offset, len) = slice.unwrap_or((0, groups.len()));
                let groups = groups.slice(offset, len);
                df.apply_columns_par(|s| unsafe { s.agg_first(&groups) })
            },
            (UniqueKeepStrategy::Last, true) => {
                // maintain order by last values, so the sorted groups are not correct as they
                // are sorted by the first value
                let gb = df.group_by_stable(names)?;
                let groups = gb.get_groups();

                let last_idx: NoNull<IdxCa> = groups
                    .iter()
                    .map(|g| match g {
                        GroupsIndicator::Idx((_first, idx)) => idx[idx.len() - 1],
                        GroupsIndicator::Slice([first, len]) => first + len - 1,
                    })
                    .collect();

                let mut last_idx = last_idx.into_inner().sort(false);

                if let Some((offset, len)) = slice {
                    last_idx = last_idx.slice(offset, len);
                }

                let last_idx = NoNull::new(last_idx);
                let out = unsafe { df.take_unchecked(&last_idx) };
                return Ok(out);
            },
            (UniqueKeepStrategy::First | UniqueKeepStrategy::Any, false) => {
                let gb = df.group_by(names)?;
                let groups = gb.get_groups();
                let (offset, len) = slice.unwrap_or((0, groups.len()));
                let groups = groups.slice(offset, len);
                df.apply_columns_par(|s| unsafe { s.agg_first(&groups) })
            },
            (UniqueKeepStrategy::Last, false) => {
                let gb = df.group_by(names)?;
                let groups = gb.get_groups();
                let (offset, len) = slice.unwrap_or((0, groups.len()));
                let groups = groups.slice(offset, len);
                df.apply_columns_par(|s| unsafe { s.agg_last(&groups) })
            },
            (UniqueKeepStrategy::None, _) => {
                let df_part = df.select(names)?;
                let mask = df_part.is_unique()?;
                let mut filtered = df.filter(&mask)?;

                if let Some((offset, len)) = slice {
                    filtered = filtered.slice(offset, len);
                }
                return Ok(filtered);
            },
        };
        Ok(unsafe { DataFrame::new_unchecked_infer_height(columns).with_schema_from(self) })
    }

    /// Get a mask of all the unique rows in the [`DataFrame`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Company" => ["Apple", "Microsoft"],
    ///                         "ISIN" => ["US0378331005", "US5949181045"])?;
    /// let ca: ChunkedArray<BooleanType> = df.is_unique()?;
    ///
    /// assert!(ca.all());
    /// # Ok::<(), PolarsError>(())
    /// ```
    #[cfg(feature = "algorithm_group_by")]
    pub fn is_unique(&self) -> PolarsResult<BooleanChunked> {
        let gb = self.group_by(self.get_column_names_owned())?;
        let groups = gb.get_groups();
        Ok(is_unique_helper(
            groups,
            self.height() as IdxSize,
            true,
            false,
        ))
    }

    /// Get a mask of all the duplicated rows in the [`DataFrame`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Company" => ["Alphabet", "Alphabet"],
    ///                         "ISIN" => ["US02079K3059", "US02079K1079"])?;
    /// let ca: ChunkedArray<BooleanType> = df.is_duplicated()?;
    ///
    /// assert!(!ca.all());
    /// # Ok::<(), PolarsError>(())
    /// ```
    #[cfg(feature = "algorithm_group_by")]
    pub fn is_duplicated(&self) -> PolarsResult<BooleanChunked> {
        let gb = self.group_by(self.get_column_names_owned())?;
        let groups = gb.get_groups();
        Ok(is_unique_helper(
            groups,
            self.height() as IdxSize,
            false,
            true,
        ))
    }

    /// Create a new [`DataFrame`] that shows the null counts per column.
    #[must_use]
    pub fn null_count(&self) -> Self {
        let cols =
            self.apply_columns(|c| Column::new(c.name().clone(), [c.null_count() as IdxSize]));
        unsafe { Self::new_unchecked(1, cols) }
    }

    /// Hash and combine the row values
    #[cfg(feature = "row_hash")]
    pub fn hash_rows(
        &mut self,
        hasher_builder: Option<PlSeedableRandomStateQuality>,
    ) -> PolarsResult<UInt64Chunked> {
        let dfs = split_df(self, POOL.current_num_threads(), false);
        let (cas, _) = _df_rows_to_hashes_threaded_vertical(&dfs, hasher_builder)?;

        let mut iter = cas.into_iter();
        let mut acc_ca = iter.next().unwrap();
        for ca in iter {
            acc_ca.append(&ca)?;
        }
        Ok(acc_ca.rechunk().into_owned())
    }

    /// Get the supertype of the columns in this DataFrame
    pub fn get_supertype(&self) -> Option<PolarsResult<DataType>> {
        self.columns()
            .iter()
            .map(|s| Ok(s.dtype().clone()))
            .reduce(|acc, b| try_get_supertype(&acc?, &b.unwrap()))
    }

    /// Take by index values given by the slice `idx`.
    /// # Warning
    /// Be careful with allowing threads when calling this in a large hot loop
    /// every thread split may be on rayon stack and lead to SO
    #[doc(hidden)]
    pub unsafe fn _take_unchecked_slice(&self, idx: &[IdxSize], allow_threads: bool) -> Self {
        self._take_unchecked_slice_sorted(idx, allow_threads, IsSorted::Not)
    }

    /// Take by index values given by the slice `idx`. Use this over `_take_unchecked_slice`
    /// if the index value in `idx` are sorted. This will maintain sorted flags.
    ///
    /// # Warning
    /// Be careful with allowing threads when calling this in a large hot loop
    /// every thread split may be on rayon stack and lead to SO
    #[doc(hidden)]
    pub unsafe fn _take_unchecked_slice_sorted(
        &self,
        idx: &[IdxSize],
        allow_threads: bool,
        sorted: IsSorted,
    ) -> Self {
        #[cfg(debug_assertions)]
        {
            if idx.len() > 2 {
                use crate::series::IsSorted;

                match sorted {
                    IsSorted::Ascending => {
                        assert!(idx[0] <= idx[idx.len() - 1]);
                    },
                    IsSorted::Descending => {
                        assert!(idx[0] >= idx[idx.len() - 1]);
                    },
                    _ => {},
                }
            }
        }
        let mut ca = IdxCa::mmap_slice(PlSmallStr::EMPTY, idx);
        ca.set_sorted_flag(sorted);
        self.take_unchecked_impl(&ca, allow_threads)
    }
    #[cfg(all(feature = "partition_by", feature = "algorithm_group_by"))]
    #[doc(hidden)]
    pub fn _partition_by_impl(
        &self,
        cols: &[PlSmallStr],
        stable: bool,
        include_key: bool,
        parallel: bool,
    ) -> PolarsResult<Vec<DataFrame>> {
        let selected_keys = self.select_to_vec(cols.iter().cloned())?;
        let groups = self.group_by_with_series(selected_keys, parallel, stable)?;
        let groups = groups.into_groups();

        // drop key columns prior to calculation if requested
        let df = if include_key {
            self.clone()
        } else {
            self.drop_many(cols.iter().cloned())
        };

        if parallel {
            // don't parallelize this
            // there is a lot of parallelization in take and this may easily SO
            POOL.install(|| {
                match groups.as_ref() {
                    GroupsType::Idx(idx) => {
                        // Rechunk as the gather may rechunk for every group #17562.
                        let mut df = df.clone();
                        df.rechunk_mut_par();
                        Ok(idx
                            .into_par_iter()
                            .map(|(_, group)| {
                                // groups are in bounds
                                unsafe {
                                    df._take_unchecked_slice_sorted(
                                        group,
                                        false,
                                        IsSorted::Ascending,
                                    )
                                }
                            })
                            .collect())
                    },
                    GroupsType::Slice { groups, .. } => Ok(groups
                        .into_par_iter()
                        .map(|[first, len]| df.slice(*first as i64, *len as usize))
                        .collect()),
                }
            })
        } else {
            match groups.as_ref() {
                GroupsType::Idx(idx) => {
                    // Rechunk as the gather may rechunk for every group #17562.
                    let mut df = df;
                    df.rechunk_mut();
                    Ok(idx
                        .into_iter()
                        .map(|(_, group)| {
                            // groups are in bounds
                            unsafe {
                                df._take_unchecked_slice_sorted(group, false, IsSorted::Ascending)
                            }
                        })
                        .collect())
                },
                GroupsType::Slice { groups, .. } => Ok(groups
                    .iter()
                    .map(|[first, len]| df.slice(*first as i64, *len as usize))
                    .collect()),
            }
        }
    }

    /// Split into multiple DataFrames partitioned by groups
    #[cfg(feature = "partition_by")]
    pub fn partition_by<I, S>(&self, cols: I, include_key: bool) -> PolarsResult<Vec<DataFrame>>
    where
        I: IntoIterator<Item = S>,
        S: Into<PlSmallStr>,
    {
        let cols: UnitVec<PlSmallStr> = cols.into_iter().map(Into::into).collect();
        self._partition_by_impl(cols.as_slice(), false, include_key, true)
    }

    /// Split into multiple DataFrames partitioned by groups
    /// Order of the groups are maintained.
    #[cfg(feature = "partition_by")]
    pub fn partition_by_stable<I, S>(
        &self,
        cols: I,
        include_key: bool,
    ) -> PolarsResult<Vec<DataFrame>>
    where
        I: IntoIterator<Item = S>,
        S: Into<PlSmallStr>,
    {
        let cols: UnitVec<PlSmallStr> = cols.into_iter().map(Into::into).collect();
        self._partition_by_impl(cols.as_slice(), true, include_key, true)
    }

    /// Unnest the given `Struct` columns. This means that the fields of the `Struct` type will be
    /// inserted as columns.
    #[cfg(feature = "dtype-struct")]
    pub fn unnest(
        &self,
        cols: impl IntoIterator<Item = impl Into<PlSmallStr>>,
        separator: Option<&str>,
    ) -> PolarsResult<DataFrame> {
        self.unnest_impl(cols.into_iter().map(Into::into).collect(), separator)
    }

    #[cfg(feature = "dtype-struct")]
    fn unnest_impl(
        &self,
        cols: PlHashSet<PlSmallStr>,
        separator: Option<&str>,
    ) -> PolarsResult<DataFrame> {
        let mut new_cols = Vec::with_capacity(std::cmp::min(self.width() * 2, self.width() + 128));
        let mut count = 0;
        for s in self.columns() {
            if cols.contains(s.name()) {
                let ca = s.struct_()?.clone();
                new_cols.extend(ca.fields_as_series().into_iter().map(|mut f| {
                    if let Some(separator) = &separator {
                        f.rename(polars_utils::format_pl_smallstr!(
                            "{}{}{}",
                            s.name(),
                            separator,
                            f.name()
                        ));
                    }
                    Column::from(f)
                }));
                count += 1;
            } else {
                new_cols.push(s.clone())
            }
        }
        if count != cols.len() {
            // one or more columns not found
            // the code below will return an error with the missing name
            let schema = self.schema();
            for col in cols {
                let _ = schema
                    .get(col.as_str())
                    .ok_or_else(|| polars_err!(col_not_found = col))?;
            }
        }

        DataFrame::new_infer_height(new_cols)
    }

    pub fn append_record_batch(&mut self, rb: RecordBatchT<ArrayRef>) -> PolarsResult<()> {
        // @Optimize: this does a lot of unnecessary allocations. We should probably have a
        // append_chunk or something like this. It is just quite difficult to make that safe.
        let df = DataFrame::from(rb);
        polars_ensure!(
            self.schema() == df.schema(),
            SchemaMismatch: "cannot append record batch with different schema\n\n
        Got {:?}\nexpected: {:?}", df.schema(), self.schema(),
        );
        self.vstack_mut_owned_unchecked(df);
        Ok(())
    }
}

pub struct RecordBatchIter<'a> {
    df: &'a DataFrame,
    schema: ArrowSchemaRef,
    idx: usize,
    n_chunks: usize,
    compat_level: CompatLevel,
    parallel: bool,
}

impl Iterator for RecordBatchIter<'_> {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.n_chunks {
            return None;
        }

        // Create a batch of the columns with the same chunk no.
        let batch_cols: Vec<ArrayRef> = if self.parallel {
            let iter = self
                .df
                .columns()
                .par_iter()
                .map(Column::as_materialized_series)
                .map(|s| s.to_arrow(self.idx, self.compat_level));
            POOL.install(|| iter.collect())
        } else {
            self.df
                .columns()
                .iter()
                .map(Column::as_materialized_series)
                .map(|s| s.to_arrow(self.idx, self.compat_level))
                .collect()
        };

        let length = batch_cols.first().map_or(0, |arr| arr.len());

        self.idx += 1;

        Some(RecordBatch::new(length, self.schema.clone(), batch_cols))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.n_chunks - self.idx;
        (n, Some(n))
    }
}

pub struct PhysRecordBatchIter<'a> {
    schema: ArrowSchemaRef,
    arr_iters: Vec<std::slice::Iter<'a, ArrayRef>>,
}

impl Iterator for PhysRecordBatchIter<'_> {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        let arrs = self
            .arr_iters
            .iter_mut()
            .map(|phys_iter| phys_iter.next().cloned())
            .collect::<Option<Vec<_>>>()?;

        let length = arrs.first().map_or(0, |arr| arr.len());
        Some(RecordBatch::new(length, self.schema.clone(), arrs))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if let Some(iter) = self.arr_iters.first() {
            iter.size_hint()
        } else {
            (0, None)
        }
    }
}

pub enum RecordBatchIterWrap<'a> {
    ZeroWidth {
        remaining_height: usize,
        chunk_size: usize,
    },
    Batches(RecordBatchIter<'a>),
    PhysicalBatches(PhysRecordBatchIter<'a>),
}

impl<'a> RecordBatchIterWrap<'a> {
    fn new_zero_width(height: usize) -> Self {
        Self::ZeroWidth {
            remaining_height: height,
            chunk_size: polars_config::config().ideal_morsel_size() as usize,
        }
    }
}

impl Iterator for RecordBatchIterWrap<'_> {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::ZeroWidth {
                remaining_height,
                chunk_size,
            } => {
                let n = usize::min(*remaining_height, *chunk_size);
                *remaining_height -= n;

                (n > 0).then(|| RecordBatch::new(n, ArrowSchemaRef::default(), vec![]))
            },
            Self::Batches(v) => v.next(),
            Self::PhysicalBatches(v) => v.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::ZeroWidth {
                remaining_height,
                chunk_size,
            } => {
                let n = remaining_height.div_ceil(*chunk_size);
                (n, Some(n))
            },
            Self::Batches(v) => v.size_hint(),
            Self::PhysicalBatches(v) => v.size_hint(),
        }
    }
}

// utility to test if we can vstack/extend the columns
fn ensure_can_extend(left: &Column, right: &Column) -> PolarsResult<()> {
    polars_ensure!(
        left.name() == right.name(),
        ShapeMismatch: "unable to vstack, column names don't match: {:?} and {:?}",
        left.name(), right.name(),
    );
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;

    fn create_frame() -> DataFrame {
        let s0 = Column::new("days".into(), [0, 1, 2].as_ref());
        let s1 = Column::new("temp".into(), [22.1, 19.9, 7.].as_ref());
        DataFrame::new_infer_height(vec![s0, s1]).unwrap()
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_recordbatch_iterator() {
        let df = df!(
            "foo" => [1, 2, 3, 4, 5]
        )
        .unwrap();
        let mut iter = df.iter_chunks(CompatLevel::newest(), false);
        assert_eq!(5, iter.next().unwrap().len());
        assert!(iter.next().is_none());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_select() {
        let df = create_frame();
        assert_eq!(
            df.column("days")
                .unwrap()
                .as_series()
                .unwrap()
                .equal(1)
                .unwrap()
                .sum(),
            Some(1)
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_filter_broadcast_on_string_col() {
        let col_name = "some_col";
        let v = vec!["test".to_string()];
        let s0 = Column::new(PlSmallStr::from_str(col_name), v);
        let mut df = DataFrame::new_infer_height(vec![s0]).unwrap();

        df = df
            .filter(
                &df.column(col_name)
                    .unwrap()
                    .as_materialized_series()
                    .equal("")
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            df.column(col_name)
                .unwrap()
                .as_materialized_series()
                .n_chunks(),
            1
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_filter_broadcast_on_list_col() {
        let s1 = Series::new(PlSmallStr::EMPTY, [true, false, true]);
        let ll: ListChunked = [&s1].iter().copied().collect();

        let mask = BooleanChunked::from_slice(PlSmallStr::EMPTY, &[false]);
        let new = ll.filter(&mask).unwrap();

        assert_eq!(new.chunks.len(), 1);
        assert_eq!(new.len(), 0);
    }

    #[test]
    fn slice() {
        let df = create_frame();
        let sliced_df = df.slice(0, 2);
        assert_eq!(sliced_df.shape(), (2, 2));
    }

    #[test]
    fn rechunk_false() {
        let df = create_frame();
        assert!(!df.should_rechunk())
    }

    #[test]
    fn rechunk_true() -> PolarsResult<()> {
        let mut base = df!(
            "a" => [1, 2, 3],
            "b" => [1, 2, 3]
        )?;

        // Create a series with multiple chunks
        let mut s = Series::new("foo".into(), 0..2);
        let s2 = Series::new("bar".into(), 0..1);
        s.append(&s2)?;

        // Append series to frame
        let out = base.with_column(s.into_column())?;

        // Now we should rechunk
        assert!(out.should_rechunk());
        Ok(())
    }

    #[test]
    fn test_duplicate_column() {
        let mut df = df! {
            "foo" => [1, 2, 3]
        }
        .unwrap();
        // check if column is replaced
        assert!(
            df.with_column(Column::new("foo".into(), &[1, 2, 3]))
                .is_ok()
        );
        assert!(
            df.with_column(Column::new("bar".into(), &[1, 2, 3]))
                .is_ok()
        );
        assert!(df.column("bar").is_ok())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn distinct() {
        let df = df! {
            "flt" => [1., 1., 2., 2., 3., 3.],
            "int" => [1, 1, 2, 2, 3, 3, ],
            "str" => ["a", "a", "b", "b", "c", "c"]
        }
        .unwrap();
        let df = df
            .unique_stable(None, UniqueKeepStrategy::First, None)
            .unwrap()
            .sort(["flt"], SortMultipleOptions::default())
            .unwrap();
        let valid = df! {
            "flt" => [1., 2., 3.],
            "int" => [1, 2, 3],
            "str" => ["a", "b", "c"]
        }
        .unwrap();
        assert!(df.equals(&valid));
    }

    #[test]
    fn test_vstack() {
        // check that it does not accidentally rechunks
        let mut df = df! {
            "flt" => [1., 1., 2., 2., 3., 3.],
            "int" => [1, 1, 2, 2, 3, 3, ],
            "str" => ["a", "a", "b", "b", "c", "c"]
        }
        .unwrap();

        df.vstack_mut(&df.slice(0, 3)).unwrap();
        assert_eq!(df.first_col_n_chunks(), 2)
    }

    #[test]
    fn test_vstack_on_empty_dataframe() {
        let mut df = DataFrame::empty();

        let df_data = df! {
            "flt" => [1., 1., 2., 2., 3., 3.],
            "int" => [1, 1, 2, 2, 3, 3, ],
            "str" => ["a", "a", "b", "b", "c", "c"]
        }
        .unwrap();

        df.vstack_mut(&df_data).unwrap();
        assert_eq!(df.height(), 6)
    }

    #[test]
    fn test_unique_keep_none_with_slice() {
        let df = df! {
            "x" => [1, 2, 3, 2, 1]
        }
        .unwrap();
        let out = df
            .unique_stable(
                Some(&["x".to_string()][..]),
                UniqueKeepStrategy::None,
                Some((0, 2)),
            )
            .unwrap();
        let expected = df! {
            "x" => [3]
        }
        .unwrap();
        assert!(out.equals(&expected));
    }

    #[test]
    #[cfg(feature = "dtype-i8")]
    fn test_apply_result_schema() {
        let mut df = df! {
            "x" => [1, 2, 3, 2, 1]
        }
        .unwrap();

        let schema_before = df.schema().clone();
        df.apply("x", |f| f.cast(&DataType::Int8).unwrap()).unwrap();
        assert_ne!(&schema_before, df.schema());
    }
}
