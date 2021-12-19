//! DataFrame module.
use std::borrow::Cow;
use std::collections::HashSet;
use std::iter::{FromIterator, Iterator};
use std::mem;
use std::ops;
use std::sync::Arc;

use ahash::{AHashSet, RandomState};
use arrow::record_batch::RecordBatch;
use itertools::Itertools;
use rayon::prelude::*;

use crate::chunked_array::ops::unique::is_unique_helper;
use crate::frame::select::Selection;
use crate::prelude::*;
use crate::utils::{
    accumulate_dataframes_horizontal, accumulate_dataframes_vertical, split_ca, split_df, NoNull,
};

mod arithmetic;
#[cfg(feature = "asof_join")]
pub(crate) mod asof_join;
#[cfg(feature = "cross_join")]
pub(crate) mod cross_join;
pub mod explode;
pub mod groupby;
pub mod hash_join;
#[cfg(feature = "rows")]
pub mod row;
pub mod select;
mod upstream_traits;

#[cfg(feature = "sort_multiple")]
use crate::prelude::sort::prepare_argsort;
use crate::vector_hasher::boost_hash_combine;
#[cfg(feature = "row_hash")]
use crate::vector_hasher::df_rows_to_hashes_threaded;
use crate::POOL;
use hashbrown::HashMap;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::hash::{BuildHasher, Hash, Hasher};

#[derive(Copy, Clone, Debug)]
pub enum NullStrategy {
    Ignore,
    Propagate,
}

/// A contiguous growable collection of `Series` that have the same length.
///
/// ## Use declarations
///
/// All the common tools can be found in [`polars_core::prelude`] (or in `polars::prelude`).
///
/// ```rust
/// use polars_core::prelude::*; // if the crate polars-core is used directly
/// // use polars::prelude::*;      if the crate polars is used
/// ```
///
/// # Initialization
/// ## Default
///
/// A `DataFrame` can be initialized empty:
///
/// ```rust
/// # use polars_core::prelude::*;
/// let df = DataFrame::default();
/// assert!(df.is_empty());
/// ```
///
/// ## Wrapping a `Vec<Series>`
///
/// A `DataFrame` is built upon a `Vec<Series>` where the `Series` have the same length.
///
/// ```rust
/// # use polars_core::prelude::*;
/// let s1 = Series::new("Fruit", &["Apple", "Apple", "Pear"]);
/// let s2 = Series::new("Color", &["Red", "Yellow", "Green"]);
///
/// let df: Result<DataFrame> = DataFrame::new(vec![s1, s2]);
/// ```
///
/// ## Using a macro
///
/// The [`df!`] macro is a convenient method:
///
/// ```rust
/// # use polars_core::prelude::*;
/// let df: Result<DataFrame> = df!("Fruit" => &["Apple", "Apple", "Pear"],
///                                 "Color" => &["Red", "Yellow", "Green"]);
/// ```
///
/// ## Using a CSV file
///
/// See the [`polars_io::csv::CsvReader`].
///
/// # Indexing
/// ## By a number
///
/// The `Index<usize>` is implemented for the `DataFrame`.
///
/// ```rust
/// # use polars_core::prelude::*;
/// let df = df!("Fruit" => &["Apple", "Apple", "Pear"],
///              "Color" => &["Red", "Yellow", "Green"])?;
///
/// assert_eq!(df[0], Series::new("Fruit", &["Apple", "Apple", "Pear"]));
/// assert_eq!(df[1], Series::new("Color", &["Red", "Yellow", "Green"]));
/// # Ok::<(), PolarsError>(())
/// ```
///
/// ## By a `Series` name
///
/// ```rust
/// # use polars_core::prelude::*;
/// let df = df!("Fruit" => &["Apple", "Apple", "Pear"],
///              "Color" => &["Red", "Yellow", "Green"])?;
///
/// assert_eq!(df["Fruit"], Series::new("Fruit", &["Apple", "Apple", "Pear"]));
/// assert_eq!(df["Color"], Series::new("Color", &["Red", "Yellow", "Green"]));
/// # Ok::<(), PolarsError>(())
/// ```
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DataFrame {
    pub(crate) columns: Vec<Series>,
}

fn duplicate_err(name: &str) -> Result<()> {
    Err(PolarsError::Duplicate(
        format!("Column with name: '{}' has more than one occurrences", name).into(),
    ))
}

impl DataFrame {
    /// Get the index of the column.
    fn check_name_to_idx(&self, name: &str) -> Result<usize> {
        self.find_idx_by_name(name)
            .ok_or_else(|| PolarsError::NotFound(name.into()))
    }

    fn check_already_present(&self, name: &str) -> Result<()> {
        if self.columns.iter().any(|s| s.name() == name) {
            Err(PolarsError::Duplicate(
                format!("column with name: '{}' already present in DataFrame", name).into(),
            ))
        } else {
            Ok(())
        }
    }

    fn hash_names(&self) -> HashSet<String, RandomState> {
        let mut set = HashSet::with_capacity_and_hasher(self.columns.len(), RandomState::default());
        for s in &self.columns {
            set.insert(s.name().to_string());
        }
        set
    }

    /// Create a DataFrame from a Vector of Series.
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// let s0 = Series::new("days", [0, 1, 2].as_ref());
    /// let s1 = Series::new("temp", [22.1, 19.9, 7.].as_ref());
    ///
    /// let df = DataFrame::new(vec![s0, s1])?;
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn new<S: IntoSeries>(columns: Vec<S>) -> Result<Self> {
        let mut first_len = None;

        let shape_err = || {
            Err(PolarsError::ShapeMisMatch(
                "Could not create a new DataFrame from Series. The Series have different lengths"
                    .into(),
            ))
        };

        let series_cols = if S::is_series() {
            // Safety:
            // we are guarded by the type system here.
            let series_cols = unsafe { std::mem::transmute::<Vec<S>, Vec<Series>>(columns) };
            let mut names = PlHashSet::with_capacity(series_cols.len());

            for s in &series_cols {
                match first_len {
                    Some(len) => {
                        if s.len() != len {
                            return shape_err();
                        }
                    }
                    None => first_len = Some(s.len()),
                }
                let name = s.name();

                if names.contains(name) {
                    duplicate_err(name)?
                }

                names.insert(name);
            }
            // we drop early as the brchk thinks the &str borrows are used when calling the drop
            // of both `series_cols` and `names`
            drop(names);
            series_cols
        } else {
            let mut series_cols = Vec::with_capacity(columns.len());
            let mut names = PlHashSet::with_capacity(columns.len());

            // check for series length equality and convert into series in one pass
            for s in columns {
                let series = s.into_series();
                match first_len {
                    Some(len) => {
                        if series.len() != len {
                            return shape_err();
                        }
                    }
                    None => first_len = Some(series.len()),
                }
                // we have aliasing borrows so we must allocate a string
                let name = series.name().to_string();

                if names.contains(&name) {
                    duplicate_err(&name)?
                }

                series_cols.push(series);
                names.insert(name);
            }
            drop(names);
            series_cols
        };

        Ok(DataFrame {
            columns: series_cols,
        })
    }

    /// Removes the last `Series` from the `DataFrame` and returns it, or [`None`] if it is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let s1 = Series::new("Ocean", &["Atlantic", "Indian"]);
    /// let s2 = Series::new("Area (km²)", &[106_460_000, 70_560_000]);
    /// let mut df = DataFrame::new(vec![s1.clone(), s2.clone()])?;
    ///
    /// assert_eq!(df.pop(), Some(s2));
    /// assert_eq!(df.pop(), Some(s1));
    /// assert_eq!(df.pop(), None);
    /// assert!(df.is_empty());
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn pop(&mut self) -> Option<Series> {
        self.columns.pop()
    }

    /// Add a new column at index 0 that counts the rows.
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Name" => &["James", "Mary", "John", "Patricia"])?;
    /// assert_eq!(df1.shape(), (4, 1));
    ///
    /// let df2: DataFrame = df1.with_row_count("Id")?;
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
    pub fn with_row_count(&self, name: &str) -> Result<Self> {
        let mut columns = Vec::with_capacity(self.columns.len() + 1);
        columns.push(
            UInt32Chunked::new_from_aligned_vec(name, (0..self.height() as u32).collect())
                .into_series(),
        );

        self.columns.iter().for_each(|s| columns.push(s.clone()));
        DataFrame::new(columns)
    }

    /// Create a new `DataFrame` but does not check the length or duplicate occurrence of the `Series`.
    ///
    /// It is advised to use [Series::new](Series::new) in favor of this method.
    ///
    /// # Panic
    /// It is the callers responsibility to uphold the contract of all `Series`
    /// having an equal length, if not this may panic down the line.
    pub fn new_no_checks(columns: Vec<Series>) -> DataFrame {
        DataFrame { columns }
    }

    /// Aggregate all chunks to contiguous memory.
    pub fn agg_chunks(&self) -> Self {
        // Don't parallelize this. Memory overhead
        let f = |s: &Series| s.rechunk();
        let cols = self.columns.iter().map(f).collect();
        DataFrame::new_no_checks(cols)
    }

    /// Shrink the capacity of this DataFrame to fit it's length.
    pub fn shrink_to_fit(&mut self) {
        // Don't parallelize this. Memory overhead
        for s in &mut self.columns {
            s.shrink_to_fit();
        }
    }

    /// Aggregate all the chunks in the DataFrame to a single chunk.
    pub fn as_single_chunk(&mut self) -> &mut Self {
        // Don't parallelize this. Memory overhead
        for s in &mut self.columns {
            *s = s.rechunk();
        }
        self
    }

    /// Aggregate all the chunks in the DataFrame to a single chunk in parallel.
    /// This may lead to more peak memory consumption.
    pub fn as_single_chunk_par(&mut self) -> &mut Self {
        self.columns = POOL.install(|| self.columns.par_iter().map(|s| s.rechunk()).collect());
        self
    }

    /// Ensure all the chunks in the DataFrame are aligned.
    pub fn rechunk(&mut self) -> &mut Self {
        let hb = RandomState::default();
        let hb2 = RandomState::with_seeds(392498, 98132457, 0, 412059);
        if self
            .columns
            .iter()
            // The idea is that we creat a hash of the chunk lengths.
            // Consisting of the combined hash + the sum (asumming collision probablility is nihil)
            // if not, we can add more hashes.
            // the old solution to this was clone all lengths to a vec and compare the vecs
            .map(|s| {
                s.chunk_lengths().map(|i| i as u64).fold(
                    (0u64, 0u64, s.n_chunks()),
                    |(lhash, lh2, n), rval| {
                        let mut h = hb.build_hasher();
                        rval.hash(&mut h);
                        let rhash = h.finish();
                        let mut h = hb2.build_hasher();
                        rval.hash(&mut h);
                        let rh2 = h.finish();
                        (
                            boost_hash_combine(lhash, rhash),
                            boost_hash_combine(lh2, rh2),
                            n,
                        )
                    },
                )
            })
            .all_equal()
        {
            self
        } else {
            self.as_single_chunk_par()
        }
    }

    /// Get the `DataFrame` schema.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Thing" => &["Observable universe", "Human stupidity"],
    ///                         "Diameter (m)" => &[8.8e26, f64::INFINITY])?;
    ///
    /// let f1: Field = Field::new("Thing", DataType::Utf8);
    /// let f2: Field = Field::new("Diameter (m)", DataType::Float64);
    /// let sc: Schema = Schema::new(vec![f1, f2]);
    ///
    /// assert_eq!(df.schema(), sc);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn schema(&self) -> Schema {
        let fields = Self::create_fields(&self.columns);
        Schema::new(fields)
    }

    /// Get a reference to the `DataFrame` columns.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Name" => &["Adenine", "Cytosine", "Guanine", "Thymine"],
    ///                         "Symbol" => &["A", "C", "G", "T"])?;
    /// let columns: &Vec<Series> = df.get_columns();
    ///
    /// assert_eq!(columns[0].name(), "Name");
    /// assert_eq!(columns[1].name(), "Symbol");
    /// # Ok::<(), PolarsError>(())
    /// ```
    #[inline]
    pub fn get_columns(&self) -> &Vec<Series> {
        &self.columns
    }

    /// Iterator over the columns as `Series`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let s1: Series = Series::new("Name", &["Pythagoras' theorem", "Shannon entropy"]);
    /// let s2: Series = Series::new("Formula", &["a²+b²=c²", "H=-Σ[P(x)log|P(x)|]"]);
    /// let df: DataFrame = DataFrame::new(vec![s1.clone(), s2.clone()])?;
    ///
    /// let mut iterator = df.iter();
    ///
    /// assert_eq!(iterator.next(), Some(&s1));
    /// assert_eq!(iterator.next(), Some(&s2));
    /// assert_eq!(iterator.next(), None);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn iter(&self) -> std::slice::Iter<'_, Series> {
        self.columns.iter()
    }

    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Language" => &["Rust", "Python"],
    ///                         "Designer" => &["Graydon Hoare", "Guido van Rossum"])?;
    ///
    /// assert_eq!(df.get_column_names(), &["Language", "Designer"]);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn get_column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|s| s.name()).collect()
    }

    /// Set the column names.
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let mut df: DataFrame = df!("Mathematical set" => &["ℕ", "ℤ", "𝔻", "ℚ", "ℝ", "ℂ"])?;
    /// df.set_column_names(&["Set"])?;
    ///
    /// assert_eq!(df.get_column_names(), &["Set"]);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn set_column_names<S: AsRef<str>>(&mut self, names: &[S]) -> Result<()> {
        if names.len() != self.columns.len() {
            return Err(PolarsError::ShapeMisMatch("the provided slice with column names has not the same size as the DataFrame's width".into()));
        }
        let unique_names: AHashSet<&str, ahash::RandomState> =
            AHashSet::from_iter(names.iter().map(|name| name.as_ref()));
        if unique_names.len() != self.columns.len() {
            return Err(PolarsError::SchemaMisMatch(
                "duplicate column names found".into(),
            ));
        }

        let columns = mem::take(&mut self.columns);
        self.columns = columns
            .into_iter()
            .zip(names)
            .map(|(s, name)| {
                let mut s = s;
                s.rename(name.as_ref());
                s
            })
            .collect();
        Ok(())
    }

    /// Get the data types of the columns in the DataFrame.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let venus_air: DataFrame = df!("Element" => &["Carbon dioxide", "Nitrogen"],
    ///                                "Fraction" => &[0.965, 0.035])?;
    ///     
    /// assert_eq!(venus_air.dtypes(), &[DataType::Utf8, DataType::Float64]);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn dtypes(&self) -> Vec<DataType> {
        self.columns.iter().map(|s| s.dtype().clone()).collect()
    }

    /// The number of chunks per column
    pub fn n_chunks(&self) -> Result<usize> {
        Ok(self
            .columns
            .get(0)
            .ok_or_else(|| {
                PolarsError::NoData("Can not determine number of chunks if there is no data".into())
            })?
            .chunks()
            .len())
    }

    /// Get fields from the columns.
    fn create_fields(columns: &[Series]) -> Vec<Field> {
        columns.iter().map(|s| s.field().into_owned()).collect()
    }

    /// Get a reference to the schema fields of the `DataFrame`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let earth: DataFrame = df!("Surface type" => &["Water", "Land"],
    ///                            "Fraction" => &[0.708, 0.292])?;
    ///     
    /// let f1: Field = Field::new("Surface type", DataType::Utf8);
    /// let f2: Field = Field::new("Fraction", DataType::Float64);
    ///
    /// assert_eq!(earth.fields(), &[f1, f2]);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn fields(&self) -> Vec<Field> {
        self.columns
            .iter()
            .map(|s| s.field().into_owned())
            .collect()
    }

    /// Get (height, width) of the `DataFrame`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df0: DataFrame = DataFrame::default();
    /// let df1: DataFrame = df!("1" => &[1, 2, 3, 4, 5])?;
    /// let df2: DataFrame = df!("1" => &[1, 2, 3, 4, 5],
    ///                          "2" => &[1, 2, 3, 4, 5])?;
    ///
    /// assert_eq!(df0.shape(), (0 ,0));
    /// assert_eq!(df1.shape(), (5, 1));
    /// assert_eq!(df2.shape(), (5, 2));
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn shape(&self) -> (usize, usize) {
        match self.columns.as_slice() {
            &[] => (0, 0),
            v => (v[0].len(), v.len()),
        }
    }

    /// Get the width of the `DataFrame` which is the number of columns.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df0: DataFrame = DataFrame::default();
    /// let df1: DataFrame = df!("Series 1" => &[0; 0])?;
    /// let df2: DataFrame = df!("Series 1" => &[0; 0],
    ///                          "Series 2" => &[0; 0])?;
    ///
    /// assert_eq!(df0.width(), 0);
    /// assert_eq!(df1.width(), 1);
    /// assert_eq!(df2.width(), 2);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn width(&self) -> usize {
        self.columns.len()
    }

    /// Get the height of the `DataFrame` which is the number of rows.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df0: DataFrame = DataFrame::default();
    /// let df1: DataFrame = df!("Currency" => &["€", "$"])?;
    /// let df2: DataFrame = df!("Currency" => &["€", "$", "¥", "£", "₿"])?;
    ///
    /// assert_eq!(df0.height(), 0);
    /// assert_eq!(df1.height(), 2);
    /// assert_eq!(df2.height(), 5);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn height(&self) -> usize {
        self.shape().0
    }

    /// Check if the `DataFrame` is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = DataFrame::default();
    /// assert!(df1.is_empty());
    ///
    /// let df2: DataFrame = df!("First name" => &["Forever"],
    ///                          "Last name" => &["Alone"])?;
    /// assert!(!df2.is_empty());
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    pub(crate) fn hstack_mut_no_checks(&mut self, columns: &[Series]) -> &mut Self {
        for col in columns {
            self.columns.push(col.clone());
        }
        self
    }

    /// Add multiple `Series` to a `DataFrame`.
    /// The added `Series` are required to have the same length.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// fn stack(df: &mut DataFrame, columns: &[Series]) {
    ///     df.hstack_mut(columns);
    /// }
    /// ```
    pub fn hstack_mut(&mut self, columns: &[Series]) -> Result<&mut Self> {
        let mut names = self.hash_names();
        let height = self.height();
        // first loop check validity. We don't do this in a single pass otherwise
        // this DataFrame is already modified when an error occurs.
        for col in columns {
            if col.len() != height && height != 0 {
                return Err(PolarsError::ShapeMisMatch(
                    format!("Could not horizontally stack Series. The Series length {} differs from the DataFrame height: {}", col.len(), height).into()));
            }

            let name = col.name();
            if names.contains(name) {
                return Err(PolarsError::Duplicate(
                    format!(
                        "Cannot do hstack operation. Column with name: {} already exists",
                        name
                    )
                    .into(),
                ));
            }
            names.insert(name.to_string());
        }
        Ok(self.hstack_mut_no_checks(columns))
    }

    /// Add multiple `Series` to a `DataFrame`.
    /// The added `Series` are required to have the same length.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Element" => &["Copper", "Silver", "Gold"])?;
    /// let s1: Series = Series::new("Proton", &[29, 47, 79]);
    /// let s2: Series = Series::new("Electron", &[29, 47, 79]);
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
    pub fn hstack(&self, columns: &[Series]) -> Result<Self> {
        let mut new_cols = self.columns.clone();
        new_cols.extend_from_slice(columns);
        DataFrame::new(new_cols)
    }

    /// Concatenate a `DataFrame` to this `DataFrame` and return as newly allocated `DataFrame`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Element" => &["Copper", "Silver", "Gold"],
    ///                          "Melting Point (K)" => &[1357.77, 1234.93, 1337.33])?;
    /// let df2: DataFrame = df!("Element" => &["Platinum", "Palladium"],
    ///                          "Melting Point (K)" => &[2041.4, 1828.05])?;
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
    pub fn vstack(&self, columns: &DataFrame) -> Result<Self> {
        let mut df = self.clone();
        df.vstack_mut(columns)?;
        Ok(df)
    }

    /// Concatenate a DataFrame to this DataFrame
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let mut df1: DataFrame = df!("Element" => &["Copper", "Silver", "Gold"],
    ///                          "Melting Point (K)" => &[1357.77, 1234.93, 1337.33])?;
    /// let df2: DataFrame = df!("Element" => &["Platinum", "Palladium"],
    ///                          "Melting Point (K)" => &[2041.4, 1828.05])?;
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
    pub fn vstack_mut(&mut self, df: &DataFrame) -> Result<&mut Self> {
        if self.width() != df.width() {
            return Err(PolarsError::ShapeMisMatch(
                format!("Could not vertically stack DataFrame. The DataFrames appended width {} differs from the parent DataFrames width {}", self.width(), df.width()).into()
            ));
        }

        self.columns
            .iter_mut()
            .zip(df.columns.iter())
            .try_for_each(|(left, right)| {
                if left.dtype() != right.dtype() || left.name() != right.name() {
                    if left.dtype() != right.dtype() {
                        return Err(PolarsError::SchemaMisMatch(
                            format!(
                                "cannot vstack: because column datatypes (dtypes) in the two DataFrames do not match for \
                                left.name='{}' with left.dtype={} != right.dtype={} with right.name='{}'",
                                left.name(),
                                left.dtype(),
                                right.dtype(),
                                right.name()
                            )
                            .into(),
                        ));
                    }
                    else {
                        return Err(PolarsError::SchemaMisMatch(
                            format!(
                                "cannot vstack: because column names in the two DataFrames do not match for \
                                left.name='{}' != right.name='{}'",
                                left.name(),
                                right.name()
                            )
                            .into(),
                        ));
                    }
                }

                left.append(right).expect("should not fail");
                Ok(())
            })?;
        // don't rechunk here. Chunks in columns always match.
        Ok(self)
    }

    /// Remove a column by name and return the column removed.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let mut df: DataFrame = df!("Animal" => &["Tiger", "Lion", "Great auk"],
    ///                             "IUCN" => &["Endangered", "Vulnerable", "Extinct"])?;
    ///
    /// let s1: Result<Series> = df.drop_in_place("Average weight");
    /// assert!(s1.is_err());
    ///
    /// let s2: Series = df.drop_in_place("Animal")?;
    /// assert_eq!(s2, Series::new("Animal", &["Tiger", "Lion", "Great auk"]));
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn drop_in_place(&mut self, name: &str) -> Result<Series> {
        let idx = self.check_name_to_idx(name)?;
        Ok(self.columns.remove(idx))
    }

    /// Return a new `DataFrame` where all null values are dropped.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Country" => &["Malta", "Liechtenstein", "North Korea"],
    ///                         "Tax revenue (% GDP)" => &[Some(32.7), None, None])?;
    /// assert_eq!(df1.shape(), (3, 2));
    ///
    /// let df2: DataFrame = df1.drop_nulls(None)?;
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
    pub fn drop_nulls(&self, subset: Option<&[String]>) -> Result<Self> {
        let selected_series;

        let mut iter = match subset {
            Some(cols) => {
                selected_series = self.select_series(&cols)?;
                selected_series.iter()
            }
            None => self.columns.iter(),
        };

        // fast path for no nulls in df
        if iter.clone().all(|s| !s.has_validity()) {
            return Ok(self.clone());
        }

        let mask = iter
            .next()
            .ok_or_else(|| PolarsError::NoData("No data to drop nulls from".into()))?;
        let mut mask = mask.is_not_null();

        for s in iter {
            mask = mask & s.is_not_null();
        }
        self.filter(&mask)
    }

    /// Drop a column by name.
    /// This is a pure method and will return a new `DataFrame` instead of modifying
    /// the current one in place.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Ray type" => &["α", "β", "X", "γ"])?;
    /// let df2: DataFrame = df1.drop("Ray type")?;
    ///
    /// assert!(df2.is_empty());
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn drop(&self, name: &str) -> Result<Self> {
        let idx = self.check_name_to_idx(name)?;
        let mut new_cols = Vec::with_capacity(self.columns.len() - 1);

        self.columns.iter().enumerate().for_each(|(i, s)| {
            if i != idx {
                new_cols.push(s.clone())
            }
        });

        Ok(DataFrame::new_no_checks(new_cols))
    }

    fn insert_at_idx_no_name_check(&mut self, index: usize, series: Series) -> Result<&mut Self> {
        if series.len() == self.height() {
            self.columns.insert(index, series);
            Ok(self)
        } else {
            Err(PolarsError::ShapeMisMatch(
                format!(
                    "Could not add column. The Series length {} differs from the DataFrame height: {}",
                    series.len(),
                    self.height()
                )
                .into(),
            ))
        }
    }

    /// Insert a new column at a given index.
    pub fn insert_at_idx<S: IntoSeries>(&mut self, index: usize, column: S) -> Result<&mut Self> {
        let series = column.into_series();
        self.check_already_present(series.name())?;
        self.insert_at_idx_no_name_check(index, series)
    }

    /// Add a new column to this `DataFrame` or replace an existing one.
    pub fn with_column<S: IntoSeries>(&mut self, column: S) -> Result<&mut Self> {
        let series = column.into_series();
        if series.len() == self.height() || self.is_empty() {
            if let Some(idx) = self.find_idx_by_name(series.name()) {
                self.replace_at_idx(idx, series)?;
            } else {
                self.columns.push(series);
            }
            Ok(self)
        } else {
            Err(PolarsError::ShapeMisMatch(
                format!(
                    "Could not add column. The Series length {} differs from the DataFrame height: {}",
                    series.len(),
                    self.height()
                )
                .into(),
            ))
        }
    }

    /// Get a row in the `DataFrame`. Beware this is slow.
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// fn example(df: &mut DataFrame, idx: usize) -> Option<Vec<AnyValue>> {
    ///     df.get(idx)
    /// }
    /// ```
    pub fn get(&self, idx: usize) -> Option<Vec<AnyValue>> {
        match self.columns.get(0) {
            Some(s) => {
                if s.len() <= idx {
                    return None;
                }
            }
            None => return None,
        }
        Some(self.columns.iter().map(|s| s.get(idx)).collect())
    }

    /// Select a `Series` by index.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Star" => &["Sun", "Betelgeuse", "Sirius A", "Sirius B"],
    ///                         "Absolute magnitude" => &[4.83, -5.85, 1.42, 11.18])?;
    ///
    /// let s1: Option<&Series> = df.select_at_idx(0);
    /// let s2: Series = Series::new("Star", &["Sun", "Betelgeuse", "Sirius A", "Sirius B"]);
    ///
    /// assert_eq!(s1, Some(&s2));
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn select_at_idx(&self, idx: usize) -> Option<&Series> {
        self.columns.get(idx)
    }

    /// Select a mutable series by index.
    ///
    /// *Note: the length of the Series should remain the same otherwise the DataFrame is invalid.*
    /// For this reason the method is not public
    fn select_at_idx_mut(&mut self, idx: usize) -> Option<&mut Series> {
        self.columns.get_mut(idx)
    }

    /// Select column(s) from this `DataFrame` by range and return a new DataFrame
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df = df! {
    ///     "0" => &[0, 0, 0],
    ///     "1" => &[1, 1, 1],
    ///     "2" => &[2, 2, 2]
    /// }?;
    ///
    /// assert!(df.select(&["0", "1"])?.frame_equal(&df.select_by_range(0..=1)?));
    /// assert!(df.frame_equal(&df.select_by_range(..)?));
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn select_by_range<R>(&self, range: R) -> Result<Self>
    where
        R: ops::RangeBounds<usize>,
    {
        // This function is copied from std::slice::range (https://doc.rust-lang.org/std/slice/fn.range.html)
        // because it is the nightly feature. We should change here if this function were stable.
        fn get_range<R>(range: R, bounds: ops::RangeTo<usize>) -> ops::Range<usize>
        where
            R: ops::RangeBounds<usize>,
        {
            let len = bounds.end;

            let start: ops::Bound<&usize> = range.start_bound();
            let start = match start {
                ops::Bound::Included(&start) => start,
                ops::Bound::Excluded(start) => start.checked_add(1).unwrap_or_else(|| {
                    panic!("attempted to index slice from after maximum usize");
                }),
                ops::Bound::Unbounded => 0,
            };

            let end: ops::Bound<&usize> = range.end_bound();
            let end = match end {
                ops::Bound::Included(end) => end.checked_add(1).unwrap_or_else(|| {
                    panic!("attempted to index slice up to maximum usize");
                }),
                ops::Bound::Excluded(&end) => end,
                ops::Bound::Unbounded => len,
            };

            if start > end {
                panic!("slice index starts at {} but ends at {}", start, end);
            }
            if end > len {
                panic!(
                    "range end index {} out of range for slice of length {}",
                    end, len
                );
            }

            ops::Range { start, end }
        }

        let colnames = &self.get_column_names();
        let range = get_range(range, ..colnames.len());

        self.select(&&colnames[range])
    }

    /// Get column index of a `Series` by name.
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Name" => &["Player 1", "Player 2", "Player 3"],
    ///                         "Health" => &[100, 200, 500],
    ///                         "Mana" => &[250, 100, 0],
    ///                         "Strength" => &[30, 150, 300])?;
    ///
    /// assert_eq!(df.find_idx_by_name("Name"), Some(0));
    /// assert_eq!(df.find_idx_by_name("Health"), Some(1));
    /// assert_eq!(df.find_idx_by_name("Mana"), Some(2));
    /// assert_eq!(df.find_idx_by_name("Strength"), Some(3));
    /// assert_eq!(df.find_idx_by_name("Haste"), None);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn find_idx_by_name(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|s| s.name() == name)
    }

    /// Select a single column by name.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let s1: Series = Series::new("Password", &["123456", "[]B$u$g$s$B#u#n#n#y[]{}"]);
    /// let s2: Series = Series::new("Robustness", &["Weak", "Strong"]);
    /// let df: DataFrame = DataFrame::new(vec![s1.clone(), s2])?;
    ///
    /// assert_eq!(df.column("Password")?, &s1);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn column(&self, name: &str) -> Result<&Series> {
        let idx = self
            .find_idx_by_name(name)
            .ok_or_else(|| PolarsError::NotFound(name.into()))?;
        Ok(self.select_at_idx(idx).unwrap())
    }

    /// Selected multiple columns by name.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Latin name" => &["Oncorhynchus kisutch", "Salmo salar"],
    ///                         "Max weight (kg)" => &[16.0, 35.89])?;
    /// let sv: Vec<&Series> = df.columns(&["Latin name", "Max weight (kg)"])?;
    ///
    /// assert_eq!(&df[0], sv[0]);
    /// assert_eq!(&df[1], sv[1]);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn columns<I, S>(&self, names: I) -> Result<Vec<&Series>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        names
            .into_iter()
            .map(|name| self.column(name.as_ref()))
            .collect()
    }

    /// Select column(s) from this `DataFrame` and return a new `DataFrame`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// fn example(df: &DataFrame, possible: &str) -> Result<DataFrame> {
    ///     match possible {
    ///         "by_str" => df.select("my-column"),
    ///         "by_tuple" => df.select(("col_1", "col_2")),
    ///         "by_vec" => df.select(vec!["col_a", "col_b"]),
    ///          _ => unimplemented!()
    ///     }
    /// }
    /// ```
    pub fn select<'a, S, J>(&self, selection: S) -> Result<Self>
    where
        S: Selection<'a, J>,
    {
        let cols = selection.to_selection_vec();
        {
            let mut names = PlHashSet::with_capacity(cols.len());
            for name in &cols {
                if !names.insert(name) {
                    duplicate_err(name)?
                }
            }
        }
        let selected = self.select_series_impl(&cols)?;
        Ok(DataFrame::new_no_checks(selected))
    }

    /// Select column(s) from this `DataFrame` and return them into a `Vec`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Name" => &["Methane", "Ethane", "Propane"],
    ///                         "Carbon" => &[1, 2, 3],
    ///                         "Hydrogen" => &[4, 6, 8])?;
    /// let sv: Vec<Series> = df.select_series(&["Carbon", "Hydrogen"])?;
    ///
    /// assert_eq!(df["Carbon"], sv[0]);
    /// assert_eq!(df["Hydrogen"], sv[1]);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn select_series<'a, S, J>(&self, selection: S) -> Result<Vec<Series>>
    where
        S: Selection<'a, J>,
    {
        let cols = selection.to_selection_vec();
        self.select_series_impl(&cols)
    }

    /// A non generic implementation to reduce compiler bloat.
    fn select_series_impl(&self, cols: &[&str]) -> Result<Vec<Series>> {
        let selected = if cols.len() > 1 && self.columns.len() > 300 {
            // we hash, because there are user that having millions of columns.
            // # https://github.com/pola-rs/polars/issues/1023
            let name_to_idx: HashMap<&str, usize> = self
                .columns
                .iter()
                .enumerate()
                .map(|(i, s)| (s.name(), i))
                .collect();
            cols.iter()
                .map(|&name| {
                    let idx = *name_to_idx
                        .get(name)
                        .ok_or_else(|| PolarsError::NotFound(name.into()))?;
                    Ok(self.select_at_idx(idx).unwrap().clone())
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            cols.iter()
                .map(|c| self.column(c).map(|s| s.clone()))
                .collect::<Result<Vec<_>>>()?
        };

        Ok(selected)
    }

    /// Select a mutable series by name.
    /// *Note: the length of the Series should remain the same otherwise the DataFrame is invalid.*
    /// For this reason the method is not public
    fn select_mut(&mut self, name: &str) -> Option<&mut Series> {
        let opt_idx = self.find_idx_by_name(name);

        match opt_idx {
            Some(idx) => self.select_at_idx_mut(idx),
            None => None,
        }
    }

    /// Does a filter but splits thread chunks vertically instead of horizontally
    /// This yields a DataFrame with `n_chunks == n_threads`.
    fn filter_vertical(&self, mask: &BooleanChunked) -> Result<Self> {
        let n_threads = POOL.current_num_threads();

        let masks = split_ca(mask, n_threads).unwrap();
        let dfs = split_df(self, n_threads).unwrap();
        let dfs: Result<Vec<_>> = POOL.install(|| {
            masks
                .par_iter()
                .zip(dfs)
                .map(|(mask, df)| {
                    let cols = df
                        .columns
                        .iter()
                        .map(|s| s.filter(mask))
                        .collect::<Result<_>>()?;
                    Ok(DataFrame::new_no_checks(cols))
                })
                .collect()
        });

        let mut iter = dfs?.into_iter();
        let first = iter.next().unwrap();
        Ok(iter.fold(first, |mut acc, df| {
            acc.vstack_mut(&df).unwrap();
            acc
        }))
    }

    /// Take the `DataFrame` rows by a boolean mask.
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// fn example(df: &DataFrame) -> Result<DataFrame> {
    ///     let mask = df.column("sepal.width")?.is_not_null();
    ///     df.filter(&mask)
    /// }
    /// ```
    pub fn filter(&self, mask: &BooleanChunked) -> Result<Self> {
        if std::env::var("POLARS_VERT_PAR").is_ok() {
            return self.filter_vertical(mask);
        }

        let new_col = POOL.install(|| {
            self.columns
                .par_iter()
                .map(|s| match s.dtype() {
                    DataType::Utf8 => s.filter_threaded(mask, true),
                    _ => s.filter(mask),
                })
                .collect::<Result<Vec<_>>>()
        })?;
        Ok(DataFrame::new_no_checks(new_col))
    }

    /// Take `DataFrame` value by indexes from an iterator.
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// fn example(df: &DataFrame) -> Result<DataFrame> {
    ///     let iterator = (0..9).into_iter();
    ///     df.take_iter(iterator)
    /// }
    /// ```
    pub fn take_iter<I>(&self, iter: I) -> Result<Self>
    where
        I: Iterator<Item = usize> + Clone + Sync,
    {
        let new_col = POOL.install(|| {
            self.columns
                .par_iter()
                .map(|s| {
                    let mut i = iter.clone();
                    s.take_iter(&mut i)
                })
                .collect::<Result<_>>()
        })?;
        Ok(DataFrame::new_no_checks(new_col))
    }

    /// Take `DataFrame` values by indexes from an iterator.
    ///
    /// # Safety
    ///
    /// This doesn't do any bound checking but checks null validity.
    pub unsafe fn take_iter_unchecked<I>(&self, mut iter: I) -> Self
    where
        I: Iterator<Item = usize> + Clone + Sync,
    {
        if std::env::var("POLARS_VERT_PAR").is_ok() {
            let idx_ca: NoNull<UInt32Chunked> = iter.into_iter().map(|idx| idx as u32).collect();
            return self.take_unchecked_vectical(&idx_ca.into_inner());
        }

        let n_chunks = match self.n_chunks() {
            Err(_) => return self.clone(),
            Ok(n) => n,
        };
        let has_utf8 = self
            .columns
            .iter()
            .any(|s| matches!(s.dtype(), DataType::Utf8));

        if (n_chunks == 1 && self.width() > 1) || has_utf8 {
            let idx_ca: NoNull<UInt32Chunked> = iter.into_iter().map(|idx| idx as u32).collect();
            let idx_ca = idx_ca.into_inner();
            return self.take_unchecked(&idx_ca);
        }

        let new_col = if self.width() == 1 {
            self.columns
                .iter()
                .map(|s| s.take_iter_unchecked(&mut iter))
                .collect::<Vec<_>>()
        } else {
            POOL.install(|| {
                self.columns
                    .par_iter()
                    .map(|s| {
                        let mut i = iter.clone();
                        s.take_iter_unchecked(&mut i)
                    })
                    .collect::<Vec<_>>()
            })
        };
        DataFrame::new_no_checks(new_col)
    }

    /// Take `DataFrame` values by indexes from an iterator that may contain None values.
    ///
    /// # Safety
    ///
    /// This doesn't do any bound checking. Out of bounds may access uninitialized memory.
    /// Null validity is checked
    pub unsafe fn take_opt_iter_unchecked<I>(&self, mut iter: I) -> Self
    where
        I: Iterator<Item = Option<usize>> + Clone + Sync,
    {
        if std::env::var("POLARS_VERT_PAR").is_ok() {
            let idx_ca: UInt32Chunked = iter.into_iter().map(|opt| opt.map(|v| v as u32)).collect();
            return self.take_unchecked_vectical(&idx_ca);
        }

        let n_chunks = match self.n_chunks() {
            Err(_) => return self.clone(),
            Ok(n) => n,
        };

        let has_utf8 = self
            .columns
            .iter()
            .any(|s| matches!(s.dtype(), DataType::Utf8));

        if (n_chunks == 1 && self.width() > 1) || has_utf8 {
            let idx_ca: UInt32Chunked = iter.into_iter().map(|opt| opt.map(|v| v as u32)).collect();
            return self.take_unchecked(&idx_ca);
        }

        let new_col = if self.width() == 1 {
            self.columns
                .iter()
                .map(|s| s.take_opt_iter_unchecked(&mut iter))
                .collect::<Vec<_>>()
        } else {
            POOL.install(|| {
                self.columns
                    .par_iter()
                    .map(|s| {
                        let mut i = iter.clone();
                        s.take_opt_iter_unchecked(&mut i)
                    })
                    .collect::<Vec<_>>()
            })
        };

        DataFrame::new_no_checks(new_col)
    }

    /// Take `DataFrame` rows by index values.
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// fn example(df: &DataFrame) -> Result<DataFrame> {
    ///     let idx = UInt32Chunked::new("idx", &[0, 1, 9]);
    ///     df.take(&idx)
    /// }
    /// ```
    pub fn take(&self, indices: &UInt32Chunked) -> Result<Self> {
        let indices = if indices.chunks.len() > 1 {
            Cow::Owned(indices.rechunk())
        } else {
            Cow::Borrowed(indices)
        };
        let new_col = POOL.install(|| {
            self.columns
                .par_iter()
                .map(|s| match s.dtype() {
                    DataType::Utf8 => s.take_threaded(&indices, true),
                    _ => s.take(&indices),
                })
                .collect::<Result<_>>()
        })?;

        Ok(DataFrame::new_no_checks(new_col))
    }

    pub(crate) unsafe fn take_unchecked(&self, idx: &UInt32Chunked) -> Self {
        let cols = POOL.install(|| {
            self.columns
                .par_iter()
                .map(|s| match s.dtype() {
                    DataType::Utf8 => s.take_unchecked_threaded(idx, true).unwrap(),
                    _ => s.take_unchecked(idx).unwrap(),
                })
                .collect()
        });
        DataFrame::new_no_checks(cols)
    }

    unsafe fn take_unchecked_vectical(&self, indices: &UInt32Chunked) -> Self {
        let n_threads = POOL.current_num_threads();
        let idxs = split_ca(indices, n_threads).unwrap();

        let dfs: Vec<_> = POOL.install(|| {
            idxs.par_iter()
                .map(|idx| {
                    let cols = self
                        .columns
                        .iter()
                        .map(|s| s.take_unchecked(idx).unwrap())
                        .collect();
                    DataFrame::new_no_checks(cols)
                })
                .collect()
        });

        let mut iter = dfs.into_iter();
        let first = iter.next().unwrap();
        iter.fold(first, |mut acc, df| {
            acc.vstack_mut(&df).unwrap();
            acc
        })
    }

    /// Rename a column in the `DataFrame`.
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// fn example(df: &mut DataFrame) -> Result<&mut DataFrame> {
    ///     let original_name = "foo";
    ///     let new_name = "bar";
    ///     df.rename(original_name, new_name)
    /// }
    /// ```
    pub fn rename(&mut self, column: &str, name: &str) -> Result<&mut Self> {
        self.select_mut(column)
            .ok_or_else(|| PolarsError::NotFound(name.into()))
            .map(|s| s.rename(name))?;

        let unique_names: AHashSet<&str, ahash::RandomState> =
            AHashSet::from_iter(self.columns.iter().map(|s| s.name()));
        if unique_names.len() != self.columns.len() {
            return Err(PolarsError::SchemaMisMatch(
                "duplicate column names found".into(),
            ));
        }
        Ok(self)
    }

    /// Sort `DataFrame` in place by a column.
    pub fn sort_in_place(&mut self, by_column: &str, reverse: bool) -> Result<&mut Self> {
        self.columns = self.sort(by_column, reverse)?.columns;
        Ok(self)
    }

    /// This is the dispatch of Self::sort, and exists to reduce compile bloat by monomorphization.
    fn sort_impl(&self, by_column: Vec<&str>, reverse: Vec<bool>) -> Result<Self> {
        let first_reverse = reverse[0];
        let first_by_column = by_column[0];
        let take = match by_column.len() {
            1 => {
                let s = self.column(by_column[0])?;
                s.argsort(reverse[0])
            }
            _ => {
                #[cfg(feature = "sort_multiple")]
                {
                    let columns = self.select_series(by_column)?;

                    let (first, columns, reverse) = prepare_argsort(columns, reverse)?;
                    first.argsort_multiple(&columns, &reverse)?
                }
                #[cfg(not(feature = "sort_multiple"))]
                {
                    panic!("activate `sort_multiple` feature gate to enable this functionality");
                }
            }
        };
        // Safety:
        // the created indices are in bounds
        let mut df = if std::env::var("POLARS_VERT_PAR").is_ok() {
            unsafe { self.take_unchecked_vectical(&take) }
        } else {
            unsafe { self.take_unchecked(&take) }
        };
        // Mark the first sort column as sorted
        df.apply(first_by_column, |s| {
            let mut s = s.clone();
            let inner = s.get_inner_mut();
            inner.set_sorted(first_reverse);
            s
        })
        .expect("column is present");
        Ok(df)
    }

    /// Return a sorted clone of this `DataFrame`.
    ///
    /// # Example
    ///
    /// ```
    /// # use polars_core::prelude::*;
    /// fn sort_example(df: &DataFrame, reverse: bool) -> Result<DataFrame> {
    ///     df.sort("a", reverse)
    /// }
    ///
    /// fn sort_by_multiple_columns_example(df: &DataFrame) -> Result<DataFrame> {
    ///     df.sort(&["a", "b"], vec![false, true])
    /// }
    /// ```
    pub fn sort<'a, S, J>(&self, by_column: S, reverse: impl IntoVec<bool>) -> Result<Self>
    where
        S: Selection<'a, J>,
    {
        // we do this heap allocation and dispatch to reduce monomorphization bloat
        let by_column = by_column.to_selection_vec();
        let reverse = reverse.into_vec();
        self.sort_impl(by_column, reverse)
    }

    /// Replace a column with a `Series`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let mut df: DataFrame = df!("Country" => &["United States", "China"],
    ///                         "Area (km²)" => &[9_833_520, 9_596_961])?;
    /// let s: Series = Series::new("Country", &["USA", "PRC"]);
    ///
    /// assert!(df.replace("Nation", s.clone()).is_err());
    /// assert!(df.replace("Country", s).is_ok());
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn replace<S: IntoSeries>(&mut self, column: &str, new_col: S) -> Result<&mut Self> {
        self.apply(column, |_| new_col.into_series())
    }

    /// Replace or update a column. The difference between this method and [DataFrame::with_column]
    /// is that now the value of `column: &str` determines the name of the column and not the name
    /// of the `Series` passed to this method.
    pub fn replace_or_add<S: IntoSeries>(&mut self, column: &str, new_col: S) -> Result<&mut Self> {
        let mut new_col = new_col.into_series();
        new_col.rename(column);
        self.with_column(new_col)
    }

    /// Replace column at index `idx` with a `Series`.
    ///
    /// # Example
    ///
    /// ```ignored
    /// # use polars_core::prelude::*;
    /// let s0 = Series::new("foo", &["ham", "spam", "egg"]);
    /// let s1 = Series::new("ascii", &[70, 79, 79]);
    /// let mut df = DataFrame::new(vec![s0, s1])?;
    ///
    /// // Add 32 to get lowercase ascii values
    /// df.replace_at_idx(1, df.select_at_idx(1).unwrap() + 32);
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn replace_at_idx<S: IntoSeries>(&mut self, idx: usize, new_col: S) -> Result<&mut Self> {
        let mut new_column = new_col.into_series();
        if new_column.len() != self.height() {
            return Err(PolarsError::ShapeMisMatch(
                format!("Cannot replace Series at index {}. The shape of Series {} does not match that of the DataFrame {}",
                idx, new_column.len(), self.height()
                ).into()));
        };
        if idx >= self.width() {
            return Err(PolarsError::OutOfBounds(
                format!(
                    "Column index: {} outside of DataFrame with {} columns",
                    idx,
                    self.width()
                )
                .into(),
            ));
        }
        let old_col = &mut self.columns[idx];
        mem::swap(old_col, &mut new_column);
        Ok(self)
    }

    /// Apply a closure to a column. This is the recommended way to do in place modification.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let s0 = Series::new("foo", &["ham", "spam", "egg"]);
    /// let s1 = Series::new("names", &["Jean", "Claude", "van"]);
    /// let mut df = DataFrame::new(vec![s0, s1])?;
    ///
    /// fn str_to_len(str_val: &Series) -> Series {
    ///     str_val.utf8()
    ///         .unwrap()
    ///         .into_iter()
    ///         .map(|opt_name: Option<&str>| {
    ///             opt_name.map(|name: &str| name.len() as u32)
    ///          })
    ///         .collect::<UInt32Chunked>()
    ///         .into_series()
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
    pub fn apply<F, S>(&mut self, name: &str, f: F) -> Result<&mut Self>
    where
        F: FnOnce(&Series) -> S,
        S: IntoSeries,
    {
        let idx = self.check_name_to_idx(name)?;
        self.apply_at_idx(idx, f)
    }

    /// Apply a closure to a column at index `idx`. This is the recommended way to do in place
    /// modification.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let s0 = Series::new("foo", &["ham", "spam", "egg"]);
    /// let s1 = Series::new("ascii", &[70, 79, 79]);
    /// let mut df = DataFrame::new(vec![s0, s1])?;
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
    pub fn apply_at_idx<F, S>(&mut self, idx: usize, f: F) -> Result<&mut Self>
    where
        F: FnOnce(&Series) -> S,
        S: IntoSeries,
    {
        let df_height = self.height();
        let width = self.width();
        let col = self.columns.get_mut(idx).ok_or_else(|| {
            PolarsError::OutOfBounds(
                format!(
                    "Column index: {} outside of DataFrame with {} columns",
                    idx, width
                )
                .into(),
            )
        })?;
        let name = col.name().to_string();
        let new_col = f(col).into_series();
        match new_col.len() {
            1 => {
                let new_col = new_col.expand_at_index(0, df_height);
                let _ = mem::replace(col, new_col);
            }
            len if (len == df_height) => {
                let _ = mem::replace(col, new_col);
            }
            len => {
                return Err(PolarsError::ShapeMisMatch(
                    format!(
                        "Result Series has shape {} where the DataFrame has height {}",
                        len,
                        self.height()
                    )
                    .into(),
                ));
            }
        }

        // make sure the name remains the same after applying the closure
        unsafe {
            let col = self.columns.get_unchecked_mut(idx);
            col.rename(&name);
        }
        Ok(self)
    }

    /// Apply a closure that may fail to a column at index `idx`. This is the recommended way to do in place
    /// modification.
    ///
    /// # Example
    ///
    /// This is the idomatic way to replace some values a column of a `DataFrame` given range of indexes.
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let s0 = Series::new("foo", &["ham", "spam", "egg", "bacon", "quack"]);
    /// let s1 = Series::new("values", &[1, 2, 3, 4, 5]);
    /// let mut df = DataFrame::new(vec![s0, s1])?;
    ///
    /// let idx = vec![0, 1, 4];
    ///
    /// df.may_apply("foo", |s| {
    ///     s.utf8()?
    ///     .set_at_idx_with(idx, |opt_val| opt_val.map(|string| format!("{}-is-modified", string)))
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
    pub fn may_apply_at_idx<F, S>(&mut self, idx: usize, f: F) -> Result<&mut Self>
    where
        F: FnOnce(&Series) -> Result<S>,
        S: IntoSeries,
    {
        let width = self.width();
        let col = self.columns.get_mut(idx).ok_or_else(|| {
            PolarsError::OutOfBounds(
                format!(
                    "Column index: {} outside of DataFrame with {} columns",
                    idx, width
                )
                .into(),
            )
        })?;
        let name = col.name().to_string();

        let _ = mem::replace(col, f(col).map(|s| s.into_series())?);

        // make sure the name remains the same after applying the closure
        unsafe {
            let col = self.columns.get_unchecked_mut(idx);
            col.rename(&name);
        }
        Ok(self)
    }

    /// Apply a closure that may fail to a column. This is the recommended way to do in place
    /// modification.
    ///
    /// # Example
    ///
    /// This is the idomatic way to replace some values a column of a `DataFrame` given a boolean mask.
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let s0 = Series::new("foo", &["ham", "spam", "egg", "bacon", "quack"]);
    /// let s1 = Series::new("values", &[1, 2, 3, 4, 5]);
    /// let mut df = DataFrame::new(vec![s0, s1])?;
    ///
    /// // create a mask
    /// let values = df.column("values")?;
    /// let mask = values.lt_eq(1) | values.gt_eq(5_i32);
    ///
    /// df.may_apply("foo", |s| {
    ///     s.utf8()?
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
    pub fn may_apply<F, S>(&mut self, column: &str, f: F) -> Result<&mut Self>
    where
        F: FnOnce(&Series) -> Result<S>,
        S: IntoSeries,
    {
        let idx = self
            .find_idx_by_name(column)
            .ok_or_else(|| PolarsError::NotFound(column.to_string()))?;
        self.may_apply_at_idx(idx, f)
    }

    /// Slice the `DataFrame` along the rows.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Fruit" => &["Apple", "Grape", "Grape", "Fig", "Fig"],
    ///                         "Color" => &["Green", "Red", "White", "White", "Red"])?;
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
    pub fn slice(&self, offset: i64, length: usize) -> Self {
        let col = self
            .columns
            .iter()
            .map(|s| s.slice(offset, length))
            .collect::<Vec<_>>();
        DataFrame::new_no_checks(col)
    }

    /// Get the head of the `DataFrame`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let countries: DataFrame =
    ///     df!("Rank by GDP (2021)" => &[1, 2, 3, 4, 5],
    ///         "Continent" => &["North America", "Asia", "Asia", "Europe", "Europe"],
    ///         "Country" => &["United States", "China", "Japan", "Germany", "United Kingdom"],
    ///         "Capital" => &["Washington", "Beijing", "Tokyo", "Berlin", "London"])?;
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
    pub fn head(&self, length: Option<usize>) -> Self {
        let col = self
            .columns
            .iter()
            .map(|s| s.head(length))
            .collect::<Vec<_>>();
        DataFrame::new_no_checks(col)
    }

    /// Get the tail of the `DataFrame`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let countries: DataFrame =
    ///     df!("Rank (2021)" => &[105, 106, 107, 108, 109],
    ///         "Apple Price (€/kg)" => &[0.75, 0.70, 0.70, 0.65, 0.52],
    ///         "Country" => &["Kosovo", "Moldova", "North Macedonia", "Syria", "Turkey"])?;
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
    /// | 108         | 0.63               | Syria   |
    /// +-------------+--------------------+---------+
    /// | 109         | 0.63               | Turkey  |
    /// +-------------+--------------------+---------+
    /// ```
    pub fn tail(&self, length: Option<usize>) -> Self {
        let col = self
            .columns
            .iter()
            .map(|s| s.tail(length))
            .collect::<Vec<_>>();
        DataFrame::new_no_checks(col)
    }

    /// Transform the underlying chunks in the `DataFrame` to Arrow RecordBatches.
    pub fn as_record_batches(&self) -> Result<Vec<RecordBatch>> {
        self.n_chunks()?;
        Ok(self.iter_record_batches().collect())
    }

    /// Iterator over the rows in this `DataFrame` as Arrow RecordBatches.
    pub fn iter_record_batches(&self) -> impl Iterator<Item = RecordBatch> + '_ {
        RecordBatchIter {
            columns: &self.columns,
            schema: Arc::new(self.schema().to_arrow()),
            idx: 0,
            n_chunks: self.n_chunks().unwrap_or(0),
        }
    }

    /// Get a `DataFrame` with all the columns in reversed order.
    pub fn reverse(&self) -> Self {
        let col = self.columns.iter().map(|s| s.reverse()).collect::<Vec<_>>();
        DataFrame::new_no_checks(col)
    }

    /// Shift the values by a given period and fill the parts that will be empty due to this operation
    /// with `Nones`.
    ///
    /// See the method on [Series](../series/enum.Series.html#method.shift) for more info on the `shift` operation.
    pub fn shift(&self, periods: i64) -> Self {
        let col = POOL.install(|| self.columns.par_iter().map(|s| s.shift(periods)).collect());
        DataFrame::new_no_checks(col)
    }

    /// Replace None values with one of the following strategies:
    /// * Forward fill (replace None with the previous value)
    /// * Backward fill (replace None with the next value)
    /// * Mean fill (replace None with the mean of the whole array)
    /// * Min fill (replace None with the minimum of the whole array)
    /// * Max fill (replace None with the maximum of the whole array)
    ///
    /// See the method on [Series](../series/enum.Series.html#method.fill_null) for more info on the `fill_null` operation.
    pub fn fill_null(&self, strategy: FillNullStrategy) -> Result<Self> {
        let col = POOL.install(|| {
            self.columns
                .par_iter()
                .map(|s| s.fill_null(strategy))
                .collect::<Result<Vec<_>>>()
        })?;
        Ok(DataFrame::new_no_checks(col))
    }

    /// Aggregate the columns to their maximum values.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Die n°1" => &[1, 3, 1, 5, 6],
    ///                          "Die n°2" => &[3, 2, 3, 5, 3])?;
    /// assert_eq!(df1.shape(), (5, 2));
    ///
    /// let df2: DataFrame = df1.max();
    /// assert_eq!(df2.shape(), (1, 2));
    /// println!("{}", df2);
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (1, 2)
    /// +---------+---------+
    /// | Die n°1 | Die n°2 |
    /// | ---     | ---     |
    /// | i32     | i32     |
    /// +=========+=========+
    /// | 6       | 5       |
    /// +---------+---------+
    /// ```
    pub fn max(&self) -> Self {
        let columns = POOL.install(|| self.columns.par_iter().map(|s| s.max_as_series()).collect());
        DataFrame::new_no_checks(columns)
    }

    /// Aggregate the columns to their standard deviation values.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Die n°1" => &[1, 3, 1, 5, 6],
    ///                          "Die n°2" => &[3, 2, 3, 5, 3])?;
    /// assert_eq!(df1.shape(), (5, 2));
    ///
    /// let df2: DataFrame = df1.std();
    /// assert_eq!(df2.shape(), (1, 2));
    /// println!("{}", df2);
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (1, 2)
    /// +-------------------+--------------------+
    /// | Die n°1           | Die n°2            |
    /// | ---               | ---                |
    /// | f64               | f64                |
    /// +===================+====================+
    /// | 2.280350850198276 | 1.0954451150103321 |
    /// +-------------------+--------------------+
    /// ```
    pub fn std(&self) -> Self {
        let columns = POOL.install(|| self.columns.par_iter().map(|s| s.std_as_series()).collect());
        DataFrame::new_no_checks(columns)
    }
    /// Aggregate the columns to their variation values.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Die n°1" => &[1, 3, 1, 5, 6],
    ///                          "Die n°2" => &[3, 2, 3, 5, 3])?;
    /// assert_eq!(df1.shape(), (5, 2));
    ///
    /// let df2: DataFrame = df1.var();
    /// assert_eq!(df2.shape(), (1, 2));
    /// println!("{}", df2);
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (1, 2)
    /// +---------+---------+
    /// | Die n°1 | Die n°2 |
    /// | ---     | ---     |
    /// | f64     | f64     |
    /// +=========+=========+
    /// | 5.2     | 1.2     |
    /// +---------+---------+
    /// ```
    pub fn var(&self) -> Self {
        let columns = POOL.install(|| self.columns.par_iter().map(|s| s.var_as_series()).collect());
        DataFrame::new_no_checks(columns)
    }

    /// Aggregate the columns to their minimum values.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Die n°1" => &[1, 3, 1, 5, 6],
    ///                          "Die n°2" => &[3, 2, 3, 5, 3])?;
    /// assert_eq!(df1.shape(), (5, 2));
    ///
    /// let df2: DataFrame = df1.min();
    /// assert_eq!(df2.shape(), (1, 2));
    /// println!("{}", df2);
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (1, 2)
    /// +---------+---------+
    /// | Die n°1 | Die n°2 |
    /// | ---     | ---     |
    /// | i32     | i32     |
    /// +=========+=========+
    /// | 1       | 2       |
    /// +---------+---------+
    /// ```
    pub fn min(&self) -> Self {
        let columns = POOL.install(|| self.columns.par_iter().map(|s| s.min_as_series()).collect());
        DataFrame::new_no_checks(columns)
    }

    /// Aggregate the columns to their sum values.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Die n°1" => &[1, 3, 1, 5, 6],
    ///                          "Die n°2" => &[3, 2, 3, 5, 3])?;
    /// assert_eq!(df1.shape(), (5, 2));
    ///
    /// let df2: DataFrame = df1.sum();
    /// assert_eq!(df2.shape(), (1, 2));
    /// println!("{}", df2);
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (1, 2)
    /// +---------+---------+
    /// | Die n°1 | Die n°2 |
    /// | ---     | ---     |
    /// | i32     | i32     |
    /// +=========+=========+
    /// | 16      | 16      |
    /// +---------+---------+
    /// ```
    pub fn sum(&self) -> Self {
        let columns = POOL.install(|| self.columns.par_iter().map(|s| s.sum_as_series()).collect());
        DataFrame::new_no_checks(columns)
    }

    /// Aggregate the columns to their mean values.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Die n°1" => &[1, 3, 1, 5, 6],
    ///                          "Die n°2" => &[3, 2, 3, 5, 3])?;
    /// assert_eq!(df1.shape(), (5, 2));
    ///
    /// let df2: DataFrame = df1.mean();
    /// assert_eq!(df2.shape(), (1, 2));
    /// println!("{}", df2);
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (1, 2)
    /// +---------+---------+
    /// | Die n°1 | Die n°2 |
    /// | ---     | ---     |
    /// | f64     | f64     |
    /// +=========+=========+
    /// | 3.2     | 3.2     |
    /// +---------+---------+
    /// ```
    pub fn mean(&self) -> Self {
        let columns = POOL.install(|| {
            self.columns
                .par_iter()
                .map(|s| s.mean_as_series())
                .collect()
        });
        DataFrame::new_no_checks(columns)
    }

    /// Aggregate the columns to their median values.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Die n°1" => &[1, 3, 1, 5, 6],
    ///                          "Die n°2" => &[3, 2, 3, 5, 3])?;
    /// assert_eq!(df1.shape(), (5, 2));
    ///
    /// let df2: DataFrame = df1.median();
    /// assert_eq!(df2.shape(), (1, 2));
    /// println!("{}", df2);
    /// # Ok::<(), PolarsError>(())
    /// ```
    ///
    /// Output:
    ///
    /// ```text
    /// shape: (1, 2)
    /// +---------+---------+
    /// | Die n°1 | Die n°2 |
    /// | ---     | ---     |
    /// | i32     | i32     |
    /// +=========+=========+
    /// | 3       | 3       |
    /// +---------+---------+
    /// ```
    pub fn median(&self) -> Self {
        let columns = POOL.install(|| {
            self.columns
                .par_iter()
                .map(|s| s.median_as_series())
                .collect()
        });
        DataFrame::new_no_checks(columns)
    }

    /// Aggregate the columns to their quantile values.
    pub fn quantile(&self, quantile: f64, interpol: QuantileInterpolOptions) -> Result<Self> {
        let columns = POOL.install(|| {
            self.columns
                .par_iter()
                .map(|s| s.quantile_as_series(quantile, interpol))
                .collect::<Result<Vec<_>>>()
        })?;
        Ok(DataFrame::new_no_checks(columns))
    }

    /// Aggregate the column horizontally to their min values.
    #[cfg(feature = "zip_with")]
    #[cfg_attr(docsrs, doc(cfg(feature = "zip_with")))]
    pub fn hmin(&self) -> Result<Option<Series>> {
        match self.columns.len() {
            0 => Ok(None),
            1 => Ok(Some(self.columns[0].clone())),
            _ => {
                let first = Cow::Borrowed(&self.columns[0]);

                self.columns[1..]
                    .iter()
                    .try_fold(first, |acc, s| {
                        let mask = acc.lt(s) & acc.is_not_null() | s.is_null();
                        let min = acc.zip_with(&mask, s)?;
                        Ok(Cow::Owned(min))
                    })
                    .map(|s| Some(s.into_owned()))
            }
        }
    }

    /// Aggregate the column horizontally to their max values.
    #[cfg(feature = "zip_with")]
    #[cfg_attr(docsrs, doc(cfg(feature = "zip_with")))]
    pub fn hmax(&self) -> Result<Option<Series>> {
        match self.columns.len() {
            0 => Ok(None),
            1 => Ok(Some(self.columns[0].clone())),
            _ => {
                let first = Cow::Borrowed(&self.columns[0]);

                self.columns[1..]
                    .iter()
                    .try_fold(first, |acc, s| {
                        let mask = acc.gt(s) & acc.is_not_null() | s.is_null();
                        let max = acc.zip_with(&mask, s)?;

                        Ok(Cow::Owned(max))
                    })
                    .map(|s| Some(s.into_owned()))
            }
        }
    }

    /// Aggregate the column horizontally to their sum values.
    pub fn hsum(&self, none_strategy: NullStrategy) -> Result<Option<Series>> {
        match self.columns.len() {
            0 => Ok(None),
            1 => Ok(Some(self.columns[0].clone())),
            _ => {
                let first = Cow::Borrowed(&self.columns[0]);
                self.columns[1..]
                    .iter()
                    .map(Cow::Borrowed)
                    .try_fold(first, |acc, s| {
                        let mut acc = acc.as_ref().clone();
                        let mut s = s.as_ref().clone();

                        if let NullStrategy::Ignore = none_strategy {
                            // if has nulls
                            if acc.has_validity() {
                                acc = acc.fill_null(FillNullStrategy::Zero)?;
                            }
                            if s.has_validity() {
                                s = s.fill_null(FillNullStrategy::Zero)?;
                            }
                        }
                        Ok(Cow::Owned(&acc + &s))
                    })
                    .map(|s| Some(s.into_owned()))
            }
        }
    }

    /// Aggregate the column horizontally to their mean values.
    pub fn hmean(&self, none_strategy: NullStrategy) -> Result<Option<Series>> {
        match self.columns.len() {
            0 => Ok(None),
            1 => Ok(Some(self.columns[0].clone())),
            _ => {
                let sum = self.hsum(none_strategy)?;

                let first: Cow<Series> =
                    Cow::Owned(self.columns[0].is_null().cast(&DataType::UInt32).unwrap());
                let null_count = self.columns[1..]
                    .iter()
                    .map(Cow::Borrowed)
                    .fold(first, |acc, s| {
                        Cow::Owned(acc.as_ref() + &s.is_null().cast(&DataType::UInt32).unwrap())
                    })
                    .into_owned();

                // value lengths: len - null_count
                let value_length: UInt32Chunked =
                    (self.width().sub(&null_count)).u32().unwrap().clone();

                // make sure that we do not divide by zero
                // by replacing with None
                let value_length = value_length
                    .set(&value_length.equal(0), None)?
                    .into_series()
                    .cast(&DataType::Float64)?;

                Ok(sum.map(|sum| &sum / &value_length))
            }
        }
    }

    /// Pipe different functions/ closure operations that work on a DataFrame together.
    pub fn pipe<F, B>(self, f: F) -> Result<B>
    where
        F: Fn(DataFrame) -> Result<B>,
    {
        f(self)
    }

    /// Pipe different functions/ closure operations that work on a DataFrame together.
    pub fn pipe_mut<F, B>(&mut self, f: F) -> Result<B>
    where
        F: Fn(&mut DataFrame) -> Result<B>,
    {
        f(self)
    }

    /// Pipe different functions/ closure operations that work on a DataFrame together.
    pub fn pipe_with_args<F, B, Args>(self, f: F, args: Args) -> Result<B>
    where
        F: Fn(DataFrame, Args) -> Result<B>,
    {
        f(self, args)
    }

    /// Create dummy variables.
    ///
    /// # Example
    ///
    /// ```ignore
    ///
    /// # #[macro_use] extern crate polars_core;
    /// # fn main() {
    ///
    ///  use polars_core::prelude::*;
    ///
    ///  let df = df! {
    ///       "id" => &[1, 2, 3, 1, 2, 3, 1, 1],
    ///       "type" => &["A", "B", "B", "B", "C", "C", "C", "B"],
    ///       "code" => &["X1", "X2", "X3", "X3", "X2", "X2", "X1", "X1"]
    ///   }.unwrap();
    ///
    ///   let dummies = df.to_dummies().unwrap();
    ///   dbg!(dummies);
    /// # }
    /// ```
    /// Outputs:
    /// ```text
    ///  +------+------+------+--------+--------+--------+---------+---------+---------+
    ///  | id_1 | id_3 | id_2 | type_A | type_B | type_C | code_X1 | code_X2 | code_X3 |
    ///  | ---  | ---  | ---  | ---    | ---    | ---    | ---     | ---     | ---     |
    ///  | u8   | u8   | u8   | u8     | u8     | u8     | u8      | u8      | u8      |
    ///  +======+======+======+========+========+========+=========+=========+=========+
    ///  | 1    | 0    | 0    | 1      | 0      | 0      | 1       | 0       | 0       |
    ///  +------+------+------+--------+--------+--------+---------+---------+---------+
    ///  | 0    | 0    | 1    | 0      | 1      | 0      | 0       | 1       | 0       |
    ///  +------+------+------+--------+--------+--------+---------+---------+---------+
    ///  | 0    | 1    | 0    | 0      | 1      | 0      | 0       | 0       | 1       |
    ///  +------+------+------+--------+--------+--------+---------+---------+---------+
    ///  | 1    | 0    | 0    | 0      | 1      | 0      | 0       | 0       | 1       |
    ///  +------+------+------+--------+--------+--------+---------+---------+---------+
    ///  | 0    | 0    | 1    | 0      | 0      | 1      | 0       | 1       | 0       |
    ///  +------+------+------+--------+--------+--------+---------+---------+---------+
    ///  | 0    | 1    | 0    | 0      | 0      | 1      | 0       | 1       | 0       |
    ///  +------+------+------+--------+--------+--------+---------+---------+---------+
    ///  | 1    | 0    | 0    | 0      | 0      | 1      | 1       | 0       | 0       |
    ///  +------+------+------+--------+--------+--------+---------+---------+---------+
    ///  | 1    | 0    | 0    | 0      | 1      | 0      | 1       | 0       | 0       |
    ///  +------+------+------+--------+--------+--------+---------+---------+---------+
    /// ```
    pub fn to_dummies(&self) -> Result<Self> {
        let cols = POOL.install(|| {
            self.columns
                .par_iter()
                .map(|s| s.to_dummies())
                .collect::<Result<Vec<_>>>()
        })?;

        accumulate_dataframes_horizontal(cols)
    }

    /// Drop duplicate rows from a `DataFrame`.
    /// *This fails when there is a column of type List in DataFrame*
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
    /// println!("{}", df.drop_duplicates(true, None)?);
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
    pub fn drop_duplicates(&self, maintain_order: bool, subset: Option<&[String]>) -> Result<Self> {
        let names = match &subset {
            Some(s) => s.iter().map(|s| &**s).collect(),
            None => self.get_column_names(),
        };
        let gb = self.groupby(names)?;
        let groups = gb.get_groups().iter().map(|v| v.0);

        let df = if maintain_order {
            let mut groups = groups.collect::<Vec<_>>();
            groups.sort_unstable();
            unsafe { self.take_iter_unchecked(groups.into_iter().map(|i| i as usize)) }
        } else {
            unsafe { self.take_iter_unchecked(groups.into_iter().map(|i| i as usize)) }
        };

        Ok(df)
    }

    /// Get a mask of all the unique rows in the `DataFrame`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Company" => &["Apple", "Microsoft"],
    ///                         "ISIN" => &["US0378331005", "US5949181045"])?;
    /// let ca: ChunkedArray<BooleanType> = df.is_unique()?;
    ///
    /// assert!(ca.all_true());
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn is_unique(&self) -> Result<BooleanChunked> {
        let mut gb = self.groupby(self.get_column_names())?;
        let groups = std::mem::take(&mut gb.groups);
        Ok(is_unique_helper(groups, self.height() as u32, true, false))
    }

    /// Get a mask of all the duplicated rows in the `DataFrame`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use polars_core::prelude::*;
    /// let df: DataFrame = df!("Company" => &["Alphabet", "Alphabet"],
    ///                         "ISIN" => &["US02079K3059", "US02079K1079"])?;
    /// let ca: ChunkedArray<BooleanType> = df.is_duplicated()?;
    ///
    /// assert!(ca.all_false());
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn is_duplicated(&self) -> Result<BooleanChunked> {
        let mut gb = self.groupby(self.get_column_names())?;
        let groups = std::mem::take(&mut gb.groups);
        Ok(is_unique_helper(groups, self.height() as u32, false, true))
    }

    /// Create a new `DataFrame` that shows the null counts per column.
    pub fn null_count(&self) -> Self {
        let cols = self
            .columns
            .iter()
            .map(|s| Series::new(s.name(), &[s.null_count() as u32]))
            .collect();
        Self::new_no_checks(cols)
    }

    /// Hash and combine the row values
    #[cfg(feature = "row_hash")]
    pub fn hash_rows(&self, hasher_builder: Option<RandomState>) -> Result<UInt64Chunked> {
        let dfs = split_df(self, POOL.current_num_threads())?;
        let (cas, _) = df_rows_to_hashes_threaded(&dfs, hasher_builder);

        let mut iter = cas.into_iter();
        let mut acc_ca = iter.next().unwrap();
        for ca in iter {
            acc_ca.append(&ca);
        }
        Ok(acc_ca.rechunk())
    }
}

pub struct RecordBatchIter<'a> {
    columns: &'a Vec<Series>,
    schema: Arc<ArrowSchema>,
    idx: usize,
    n_chunks: usize,
}

impl<'a> Iterator for RecordBatchIter<'a> {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.n_chunks {
            None
        } else {
            // create a batch of the columns with the same chunk no.
            let batch_cols = self.columns.iter().map(|s| s.to_arrow(self.idx)).collect();
            self.idx += 1;

            Some(RecordBatch::try_new(self.schema.clone(), batch_cols).unwrap())
        }
    }
}

impl Default for DataFrame {
    fn default() -> Self {
        DataFrame::new_no_checks(vec![])
    }
}

impl From<DataFrame> for Vec<Series> {
    fn from(df: DataFrame) -> Self {
        df.columns
    }
}

/// Conversion from Vec<RecordBatch> into DataFrame
///
///
impl std::convert::TryFrom<RecordBatch> for DataFrame {
    type Error = PolarsError;

    fn try_from(batch: RecordBatch) -> Result<DataFrame> {
        let columns: Result<Vec<Series>> = batch
            .columns()
            .iter()
            .zip(batch.schema().fields())
            .map(|(arr, field)| Series::try_from((field.name().as_ref(), arr.clone())))
            .collect();

        DataFrame::new(columns?)
    }
}

/// Conversion from Vec<RecordBatch> into DataFrame
///
/// If batch-size is small it might be advisable to call rechunk
/// to ensure predictable performance
impl std::convert::TryFrom<Vec<RecordBatch>> for DataFrame {
    type Error = PolarsError;

    fn try_from(batches: Vec<RecordBatch>) -> Result<DataFrame> {
        let mut batch_iter = batches.iter();

        // Non empty array
        let first_batch = batch_iter.next().ok_or_else(|| {
            PolarsError::NoData("At least one record batch is needed to create a dataframe".into())
        })?;

        // Validate all record batches have the same schema
        let schema = first_batch.schema();
        for batch in batch_iter {
            if batch.schema() != schema {
                return Err(PolarsError::SchemaMisMatch(
                    "All record batches must have the same schema".into(),
                ));
            }
        }

        let dfs: Result<Vec<DataFrame>> = batches
            .iter()
            .map(|batch| DataFrame::try_from(batch.clone()))
            .collect();

        accumulate_dataframes_vertical(dfs?)
    }
}

#[cfg(test)]
mod test {
    use std::convert::TryFrom;

    use arrow::array::{Float64Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;

    use crate::frame::NullStrategy;
    use crate::prelude::*;

    fn create_frame() -> DataFrame {
        let s0 = Series::new("days", [0, 1, 2].as_ref());
        let s1 = Series::new("temp", [22.1, 19.9, 7.].as_ref());
        DataFrame::new(vec![s0, s1]).unwrap()
    }

    fn create_record_batches() -> Vec<RecordBatch> {
        // Creates a dataframe using 2 record-batches
        //
        // | foo    | bar    |
        // -------------------
        // | 1.0    | 1      |
        // | 2.0    | 2      |
        // | 3.0    | 3      |
        // | 4.0    | 4      |
        // | 5.0    | 5      |
        // -------------------
        let schema = Arc::new(Schema::new(vec![
            Field::new("foo", DataType::Float64, false),
            Field::new("bar", DataType::Int64, false),
        ]));

        let batch0 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float64Array::from_slice(&[1.0, 2.0, 3.0])),
                Arc::new(Int64Array::from_slice(&[1, 2, 3])),
            ],
        )
        .unwrap();

        let batch1 = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from_slice(&[4.0, 5.0])),
                Arc::new(Int64Array::from_slice(&[4, 5])),
            ],
        )
        .unwrap();

        return vec![batch0, batch1];
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_recordbatch_iterator() {
        let df = df!(
            "foo" => &[1, 2, 3, 4, 5]
        )
        .unwrap();
        let mut iter = df.iter_record_batches();
        assert_eq!(5, iter.next().unwrap().num_rows());
        assert!(iter.next().is_none());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_frame_from_recordbatch() {
        let record_batches: Vec<RecordBatch> = create_record_batches();

        let df = DataFrame::try_from(record_batches).expect("frame can be initialized");

        assert_eq!(
            Vec::from(df.column("bar").unwrap().i64().unwrap()),
            &[Some(1), Some(2), Some(3), Some(4), Some(5)]
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_select() {
        let df = create_frame();
        assert_eq!(df.column("days").unwrap().equal(1).sum(), Some(1));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_filter() {
        let df = create_frame();
        println!("{}", df.column("days").unwrap());
        println!("{:?}", df);
        println!("{:?}", df.filter(&df.column("days").unwrap().equal(0)))
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_filter_broadcast_on_utf8_col() {
        let col_name = "some_col";
        let v = vec!["test".to_string()];
        let s0 = Series::new(col_name, v);
        let mut df = DataFrame::new(vec![s0]).unwrap();
        println!("{}", df.column(col_name).unwrap());
        println!("{:?}", df);

        df = df.filter(&df.column(col_name).unwrap().equal("")).unwrap();
        assert_eq!(df.column(col_name).unwrap().n_chunks(), 1);
        println!("{:?}", df);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_filter_broadcast_on_list_col() {
        let s1 = Series::new("", &[true, false, true]);
        let ll: ListChunked = [&s1].iter().copied().collect();

        let mask = BooleanChunked::new_from_slice("", &[false]);
        let new = ll.filter(&mask).unwrap();

        assert_eq!(new.chunks.len(), 1);
        assert_eq!(new.len(), 0);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_sort() {
        let mut df = create_frame();
        df.sort_in_place("temp", false).unwrap();
        println!("{:?}", df);
    }

    #[test]
    fn slice() {
        let df = create_frame();
        let sliced_df = df.slice(0, 2);
        assert_eq!(sliced_df.shape(), (2, 2));
        println!("{:?}", df)
    }

    #[test]
    #[cfg(feature = "dtype-u8")]
    #[cfg_attr(miri, ignore)]
    fn get_dummies() {
        let df = df! {
            "id" => &[1, 2, 3, 1, 2, 3, 1, 1],
            "type" => &["A", "B", "B", "B", "C", "C", "C", "B"],
            "code" => &["X1", "X2", "X3", "X3", "X2", "X2", "X1", "X1"]
        }
        .unwrap();
        let dummies = df.to_dummies().unwrap();
        dbg!(&dummies);
        assert_eq!(
            Vec::from(dummies.column("id_1").unwrap().u8().unwrap()),
            &[
                Some(1),
                Some(0),
                Some(0),
                Some(1),
                Some(0),
                Some(0),
                Some(1),
                Some(1)
            ]
        );
        dbg!(dummies);
    }

    #[test]
    fn test_duplicate_column() {
        let mut df = df! {
            "foo" => &[1, 2, 3]
        }
        .unwrap();
        // check if column is replaced
        assert!(df.with_column(Series::new("foo", &[1, 2, 3])).is_ok());
        assert!(df.with_column(Series::new("bar", &[1, 2, 3])).is_ok());
        assert!(df.column("bar").is_ok())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn drop_duplicates() {
        let df = df! {
            "flt" => [1., 1., 2., 2., 3., 3.],
            "int" => [1, 1, 2, 2, 3, 3, ],
            "str" => ["a", "a", "b", "b", "c", "c"]
        }
        .unwrap();
        dbg!(&df);
        let df = df
            .drop_duplicates(true, None)
            .unwrap()
            .sort("flt", false)
            .unwrap();
        let valid = df! {
            "flt" => [1., 2., 3.],
            "int" => [1, 2, 3],
            "str" => ["a", "b", "c"]
        }
        .unwrap();
        dbg!(&df);
        assert!(df.frame_equal(&valid));
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
        assert_eq!(df.n_chunks().unwrap(), 2)
    }

    #[test]
    #[cfg(feature = "zip_with")]
    fn test_h_agg() {
        let a = Series::new("a", &[1, 2, 6]);
        let b = Series::new("b", &[Some(1), None, None]);
        let c = Series::new("c", &[Some(4), None, Some(3)]);

        let df = DataFrame::new(vec![a, b, c]).unwrap();
        assert_eq!(
            Vec::from(
                df.hmean(NullStrategy::Ignore)
                    .unwrap()
                    .unwrap()
                    .f64()
                    .unwrap()
            ),
            &[Some(2.0), Some(2.0), Some(4.5)]
        );
        assert_eq!(
            Vec::from(
                df.hsum(NullStrategy::Ignore)
                    .unwrap()
                    .unwrap()
                    .i32()
                    .unwrap()
            ),
            &[Some(6), Some(2), Some(9)]
        );
        assert_eq!(
            Vec::from(df.hmin().unwrap().unwrap().i32().unwrap()),
            &[Some(1), Some(2), Some(3)]
        );
        assert_eq!(
            Vec::from(df.hmax().unwrap().unwrap().i32().unwrap()),
            &[Some(4), Some(2), Some(6)]
        );
    }

    #[test]
    fn test_replace_or_add() -> Result<()> {
        let mut df = df!(
            "a" => [1, 2, 3],
            "b" => [1, 2, 3]
        )?;

        // check that the new column is "c" and not "bar".
        df.replace_or_add("c", Series::new("bar", [1, 2, 3]))?;

        assert_eq!(df.get_column_names(), &["a", "b", "c"]);
        Ok(())
    }
}
