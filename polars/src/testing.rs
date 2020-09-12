//! Testing utilities.
use crate::prelude::*;

impl Series {
    /// Check if series are equal. Note that `None == None` evaluates to `false`
    pub fn series_equal(&self, other: &Series) -> bool {
        if self.len() != other.len() {
            return false;
        }
        if self.null_count() != other.null_count() {
            return false;
        }
        match self.eq(other).sum() {
            None => false,
            Some(sum) => sum as usize == self.len(),
        }
    }

    /// Check if all values in series are equal where `None == None` evaluates to `true`.
    pub fn series_equal_missing(&self, other: &Series) -> bool {
        if self.len() != other.len() {
            return false;
        }
        if self.null_count() != other.null_count() {
            return false;
        }
        // if all null and previous check did not return (so other is also all null)
        if self.null_count() == self.len() {
            return true;
        }
        match self.eq_missing(other).sum() {
            None => false,
            Some(sum) => sum as usize == self.len(),
        }
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
        return true;
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
        return true;
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
}
