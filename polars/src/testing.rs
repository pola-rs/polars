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
    pub fn series_all_equal(&self, other: &Series) -> bool {
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
        // Fill all None values with the minimum. We cannot do a backfill, because it can take
        // multiple iterations to fill all None's.
        // TODO: speedup, by implementing typed comparison logic in ChunkedArray.
        let left;
        let left_ref;
        let right;
        let right_ref;
        if self.null_count() > 0 {
            left = self
                .fill_none(FillNoneStrategy::Min)
                .expect("fill none operation not implemented");
            left_ref = &left;
        } else {
            left_ref = self;
        }
        if other.null_count() > 0 {
            right = other
                .fill_none(FillNoneStrategy::Min)
                .expect("fill none operation not implemented");
            right_ref = &right;
        } else {
            right_ref = other;
        }

        match left_ref.eq(right_ref).sum() {
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
        // todo: early return
        self.get_columns()
            .iter()
            .zip(other.get_columns().iter())
            .map(|(a, b)| a.series_equal(b))
            .all(|v| v)
    }

    /// Check if all values in `DataFrames` are equal where `None == None` evaluates to `true`.
    pub fn frame_all_equal(&self, other: &DataFrame) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        // todo: early return
        self.get_columns()
            .iter()
            .zip(other.get_columns().iter())
            .map(|(a, b)| a.series_all_equal(b))
            .all(|v| v)
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
        assert!(s.series_all_equal(&s));
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
