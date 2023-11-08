//! Testing utilities.
use std::ops::Deref;

use crate::prelude::*;

impl Series {
    /// Check if series are equal. Note that `None == None` evaluates to `false`
    pub fn series_equal(&self, other: &Series) -> bool {
        if self.null_count() > 0 || other.null_count() > 0 || self.dtype() != other.dtype() {
            false
        } else {
            self.series_equal_missing(other)
        }
    }

    /// Check if all values in series are equal where `None == None` evaluates to `true`.
    /// Two [`Datetime`](DataType::Datetime) series are *not* equal if their timezones are different, regardless
    /// if they represent the same UTC time or not.
    pub fn series_equal_missing(&self, other: &Series) -> bool {
        match (self.dtype(), other.dtype()) {
            #[cfg(feature = "timezones")]
            (DataType::Datetime(_, tz_lhs), DataType::Datetime(_, tz_rhs)) => {
                if tz_lhs != tz_rhs {
                    return false;
                }
            },
            _ => {},
        }

        // differences from Partial::eq in that numerical dtype may be different
        self.len() == other.len()
            && self.name() == other.name()
            && self.null_count() == other.null_count()
            && {
                let eq = self.equal_missing(other);
                match eq {
                    Ok(b) => b.sum().map(|s| s as usize).unwrap_or(0) == self.len(),
                    Err(_) => false,
                }
            }
    }

    /// Get a pointer to the underlying data of this [`Series`].
    /// Can be useful for fast comparisons.
    pub fn get_data_ptr(&self) -> usize {
        let object = self.0.deref();

        // Safety:
        // A fat pointer consists of a data ptr and a ptr to the vtable.
        // we specifically check that we only transmute &dyn SeriesTrait e.g.
        // a trait object, therefore this is sound.
        #[allow(clippy::transmute_undefined_repr)]
        let (data_ptr, _vtable_ptr) =
            unsafe { std::mem::transmute::<&dyn SeriesTrait, (usize, usize)>(object) };
        data_ptr
    }
}

impl PartialEq for Series {
    fn eq(&self, other: &Self) -> bool {
        self.series_equal_missing(other)
    }
}

impl DataFrame {
    /// Check if [`DataFrame`]' schemas are equal.
    pub fn frame_equal_schema(&self, other: &DataFrame) -> PolarsResult<()> {
        for (lhs, rhs) in self.iter().zip(other.iter()) {
            polars_ensure!(
                lhs.name() == rhs.name(),
                SchemaMismatch: "column name mismatch: left-hand = '{}', right-hand = '{}'",
                lhs.name(), rhs.name()
            );
            polars_ensure!(
                lhs.dtype() == rhs.dtype(),
                SchemaMismatch: "column datatype mismatch: left-hand = '{}', right-hand = '{}'",
                lhs.dtype(), rhs.dtype()
            );
        }
        Ok(())
    }

    /// Check if [`DataFrame`]s are equal. Note that `None == None` evaluates to `false`
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Atomic number" => &[1, 51, 300],
    ///                         "Element" => &[Some("Hydrogen"), Some("Antimony"), None])?;
    /// let df2: DataFrame = df!("Atomic number" => &[1, 51, 300],
    ///                         "Element" => &[Some("Hydrogen"), Some("Antimony"), None])?;
    ///
    /// assert!(!df1.frame_equal(&df2));
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn frame_equal(&self, other: &DataFrame) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        for (left, right) in self.get_columns().iter().zip(other.get_columns()) {
            if !left.series_equal(right) {
                return false;
            }
        }
        true
    }

    /// Check if all values in [`DataFrame`]s are equal where `None == None` evaluates to `true`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Atomic number" => &[1, 51, 300],
    ///                         "Element" => &[Some("Hydrogen"), Some("Antimony"), None])?;
    /// let df2: DataFrame = df!("Atomic number" => &[1, 51, 300],
    ///                         "Element" => &[Some("Hydrogen"), Some("Antimony"), None])?;
    ///
    /// assert!(df1.frame_equal_missing(&df2));
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn frame_equal_missing(&self, other: &DataFrame) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        for (left, right) in self.get_columns().iter().zip(other.get_columns()) {
            if !left.series_equal_missing(right) {
                return false;
            }
        }
        true
    }

    /// Checks if the Arc ptrs of the [`Series`] are equal
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let df1: DataFrame = df!("Atomic number" => &[1, 51, 300],
    ///                         "Element" => &[Some("Hydrogen"), Some("Antimony"), None])?;
    /// let df2: &DataFrame = &df1;
    ///
    /// assert!(df1.ptr_equal(df2));
    /// # Ok::<(), PolarsError>(())
    /// ```
    pub fn ptr_equal(&self, other: &DataFrame) -> bool {
        self.columns
            .iter()
            .zip(other.columns.iter())
            .all(|(a, b)| a.get_data_ptr() == b.get_data_ptr())
    }
}

impl PartialEq for DataFrame {
    fn eq(&self, other: &Self) -> bool {
        self.shape() == other.shape()
            && self
                .columns
                .iter()
                .zip(other.columns.iter())
                .all(|(s1, s2)| s1.series_equal_missing(s2))
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_series_equal() {
        let a = Series::new("a", &[1_u32, 2, 3]);
        let b = Series::new("a", &[1_u32, 2, 3]);
        assert!(a.series_equal(&b));

        let s = Series::new("foo", &[None, Some(1i64)]);
        assert!(s.series_equal_missing(&s));
    }

    #[test]
    fn test_series_dtype_noteq() {
        let s_i32 = Series::new("a", &[1_i32, 2_i32]);
        let s_i64 = Series::new("a", &[1_i64, 2_i64]);
        assert!(!s_i32.series_equal(&s_i64));
    }

    #[test]
    fn test_df_equal() {
        let a = Series::new("a", [1, 2, 3].as_ref());
        let b = Series::new("b", [1, 2, 3].as_ref());

        let df1 = DataFrame::new(vec![a, b]).unwrap();
        assert!(df1.frame_equal(&df1))
    }

    #[test]
    fn test_df_partialeq() {
        let df1 = df!("a" => &[1, 2, 3],
                      "b" => &[4, 5, 6])
        .unwrap();
        let df2 = df!("b" => &[4, 5, 6],
                      "a" => &[1, 2, 3])
        .unwrap();
        let df3 = df!("" => &[Some(1), None]).unwrap();
        let df4 = df!("" => &[f32::NAN]).unwrap();

        assert_eq!(df1, df1);
        assert_ne!(df1, df2);
        assert_eq!(df2, df2);
        assert_ne!(df2, df3);
        assert_eq!(df3, df3);
        assert_ne!(df4, df4);
    }
}
