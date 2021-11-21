//! Testing utilities.
use crate::prelude::*;
use std::ops::Deref;

impl Series {
    /// Check if series are equal. Note that `None == None` evaluates to `false`
    pub fn series_equal(&self, other: &Series) -> bool {
        if self.get_data_ptr() == other.get_data_ptr() {
            return true;
        }
        if self.len() != other.len() || self.null_count() != other.null_count() {
            return false;
        }
        if self.dtype() != other.dtype()
            && !(matches!(self.dtype(), DataType::Utf8 | DataType::Categorical)
                || matches!(other.dtype(), DataType::Utf8 | DataType::Categorical))
            && !(self.is_numeric() && other.is_numeric())
        {
            return false;
        }
        match self.equal(other).sum() {
            None => false,
            Some(sum) => sum as usize == self.len(),
        }
    }

    /// Check if all values in series are equal where `None == None` evaluates to `true`.
    pub fn series_equal_missing(&self, other: &Series) -> bool {
        if self.get_data_ptr() == other.get_data_ptr() {
            return true;
        }
        let null_count_left = self.null_count();
        if self.len() != other.len() || null_count_left != other.null_count() {
            return false;
        }
        if self.dtype() != other.dtype()
            && !(matches!(self.dtype(), DataType::Utf8 | DataType::Categorical)
                || matches!(other.dtype(), DataType::Utf8 | DataType::Categorical))
            && !(self.is_numeric() && other.is_numeric())
        {
            return false;
        }
        // if all null and previous check did not return (so other is also all null)
        if null_count_left == self.len() {
            return true;
        }
        match self.eq_missing(other).sum() {
            None => false,
            Some(sum) => sum as usize == self.len(),
        }
    }

    /// Get a pointer to the underlying data of this Series.
    /// Can be useful for fast comparisons.
    pub fn get_data_ptr(&self) -> usize {
        let object = self.0.deref();

        // Safety:
        // A fat pointer consists of a data ptr and a ptr to the vtable.
        // we specifically check that we only transmute &dyn SeriesTrait e.g.
        // a trait object, therefore this is sound.
        let (data_ptr, _vtable_ptr) =
            unsafe { std::mem::transmute::<&dyn SeriesTrait, (usize, usize)>(object) };
        data_ptr
    }
}

impl PartialEq for Series {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self.field() == other.field()
            && self.null_count() == other.null_count()
            && self.eq_missing(other).sum().map(|s| s as usize) == Some(self.len())
    }
}

impl DataFrame {
    /// Check if `DataFrames` are equal. Note that `None == None` evaluates to `false`
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

    /// Check if all values in `DataFrames` are equal where `None == None` evaluates to `true`.
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

    /// Checks if the Arc ptrs of the Series are equal
    pub fn ptr_equal(&self, other: &DataFrame) -> bool {
        self.columns
            .iter()
            .zip(other.columns.iter())
            .all(|(a, b)| a.get_data_ptr() == b.get_data_ptr())
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_series_equal() {
        let a = Series::new("a", &[1, 2, 3]);
        let b = Series::new("b", &[1, 2, 3]);
        assert!(a.series_equal(&b));

        let s = Series::new("foo", &[None, Some(1i64)]);
        assert!(s.series_equal_missing(&s));
    }

    #[test]
    fn test_df_equal() {
        let a = Series::new("a", [1, 2, 3].as_ref());
        let b = Series::new("b", [1, 2, 3].as_ref());

        let df1 = DataFrame::new(vec![a, b]).unwrap();
        let df2 = df1.clone();
        assert!(df1.frame_equal(&df2))
    }

    #[test]
    fn test_series_partialeq() {
        let s1 = Series::new("a", &[1_i32, 2_i32, 3_i32]);
        let s1_bis = Series::new("b", &[1_i32, 2_i32, 3_i32]);
        let s1_ter = Series::new("a", &[1.0_f64, 2.0_f64, 3.0_f64]);
        let s2 = Series::new("", &[Some(1), Some(0)]);
        let s3 = Series::new("", &[Some(1), None]);
        let s4 = Series::new("", &[1.0, f64::NAN]);

        assert_eq!(s1, s1);
        assert_ne!(s1, s1_bis);
        assert_ne!(s1, s1_ter);
        assert_eq!(s2, s2);
        assert_ne!(s2, s3);
        assert_ne!(s4, s4);
    }
}
