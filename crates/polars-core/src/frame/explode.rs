use arrow::offset::OffsetsBuffer;
use rayon::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use smartstring::alias::String as SmartString;

use crate::chunked_array::ops::explode::offsets_to_indexes;
use crate::prelude::*;
use crate::series::IsSorted;
use crate::POOL;

fn get_exploded(series: &Series) -> PolarsResult<(Series, OffsetsBuffer<i64>)> {
    match series.dtype() {
        DataType::List(_) => series.list().unwrap().explode_and_offsets(),
        #[cfg(feature = "dtype-array")]
        DataType::Array(_, _) => series.array().unwrap().explode_and_offsets(),
        _ => polars_bail!(opq = explode, series.dtype()),
    }
}

/// Arguments for `[DataFrame::unpivot]` function
#[derive(Clone, Default, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UnpivotArgsIR {
    pub on: Vec<SmartString>,
    pub index: Vec<SmartString>,
    pub variable_name: Option<SmartString>,
    pub value_name: Option<SmartString>,
}

impl DataFrame {
    pub fn explode_impl(&self, mut columns: Vec<Series>) -> PolarsResult<DataFrame> {
        polars_ensure!(!columns.is_empty(), InvalidOperation: "no columns provided in explode");
        let mut df = self.clone();
        if self.is_empty() {
            for s in &columns {
                df.with_column(s.explode()?)?;
            }
            return Ok(df);
        }
        columns.sort_by(|sa, sb| {
            self.check_name_to_idx(sa.name())
                .expect("checked above")
                .partial_cmp(&self.check_name_to_idx(sb.name()).expect("checked above"))
                .expect("cmp usize -> Ordering")
        });

        // first remove all the exploded columns
        for s in &columns {
            df = df.drop(s.name())?;
        }

        let exploded_columns = POOL.install(|| {
            columns
                .par_iter()
                .map(get_exploded)
                .collect::<PolarsResult<Vec<_>>>()
        })?;

        fn process_column(
            original_df: &DataFrame,
            df: &mut DataFrame,
            exploded: Series,
        ) -> PolarsResult<()> {
            if exploded.len() == df.height() || df.width() == 0 {
                let col_idx = original_df.check_name_to_idx(exploded.name())?;
                df.columns.insert(col_idx, exploded);
            } else {
                polars_bail!(
                    ShapeMismatch: "exploded column(s) {:?} doesn't have the same length: {} \
                    as the dataframe: {}", exploded.name(), exploded.name(), df.height(),
                );
            }
            Ok(())
        }

        let check_offsets = || {
            let first_offsets = exploded_columns[0].1.as_slice();
            for (_, offsets) in &exploded_columns[1..] {
                let offsets = offsets.as_slice();

                let offset_l = first_offsets[0];
                let offset_r = offsets[0];
                let all_equal_len = first_offsets.len() != offsets.len() || {
                    first_offsets
                        .iter()
                        .zip(offsets.iter())
                        .all(|(l, r)| (*l - offset_l) == (*r - offset_r))
                };

                polars_ensure!(all_equal_len,
                    ShapeMismatch: "exploded columns must have matching element counts"
                )
            }
            Ok(())
        };
        let process_first = || {
            let (exploded, offsets) = &exploded_columns[0];

            let row_idx = offsets_to_indexes(offsets.as_slice(), exploded.len());
            let mut row_idx = IdxCa::from_vec("", row_idx);
            row_idx.set_sorted_flag(IsSorted::Ascending);

            // SAFETY:
            // We just created indices that are in bounds.
            let mut df = unsafe { df.take_unchecked(&row_idx) };
            process_column(self, &mut df, exploded.clone())?;
            PolarsResult::Ok(df)
        };
        let (df, result) = POOL.join(process_first, check_offsets);
        let mut df = df?;
        result?;

        for (exploded, _) in exploded_columns.into_iter().skip(1) {
            process_column(self, &mut df, exploded)?
        }

        Ok(df)
    }
    /// Explode `DataFrame` to long format by exploding a column with Lists.
    ///
    /// # Example
    ///
    /// ```ignore
    /// # use polars_core::prelude::*;
    /// let s0 = Series::new("a", &[1i64, 2, 3]);
    /// let s1 = Series::new("b", &[1i64, 1, 1]);
    /// let s2 = Series::new("c", &[2i64, 2, 2]);
    /// let list = Series::new("foo", &[s0, s1, s2]);
    ///
    /// let s0 = Series::new("B", [1, 2, 3]);
    /// let s1 = Series::new("C", [1, 1, 1]);
    /// let df = DataFrame::new(vec![list, s0, s1])?;
    /// let exploded = df.explode(["foo"])?;
    ///
    /// println!("{:?}", df);
    /// println!("{:?}", exploded);
    /// # Ok::<(), PolarsError>(())
    /// ```
    /// Outputs:
    ///
    /// ```text
    ///  +-------------+-----+-----+
    ///  | foo         | B   | C   |
    ///  | ---         | --- | --- |
    ///  | list [i64]  | i32 | i32 |
    ///  +=============+=====+=====+
    ///  | "[1, 2, 3]" | 1   | 1   |
    ///  +-------------+-----+-----+
    ///  | "[1, 1, 1]" | 2   | 1   |
    ///  +-------------+-----+-----+
    ///  | "[2, 2, 2]" | 3   | 1   |
    ///  +-------------+-----+-----+
    ///
    ///  +-----+-----+-----+
    ///  | foo | B   | C   |
    ///  | --- | --- | --- |
    ///  | i64 | i32 | i32 |
    ///  +=====+=====+=====+
    ///  | 1   | 1   | 1   |
    ///  +-----+-----+-----+
    ///  | 2   | 1   | 1   |
    ///  +-----+-----+-----+
    ///  | 3   | 1   | 1   |
    ///  +-----+-----+-----+
    ///  | 1   | 2   | 1   |
    ///  +-----+-----+-----+
    ///  | 1   | 2   | 1   |
    ///  +-----+-----+-----+
    ///  | 1   | 2   | 1   |
    ///  +-----+-----+-----+
    ///  | 2   | 3   | 1   |
    ///  +-----+-----+-----+
    ///  | 2   | 3   | 1   |
    ///  +-----+-----+-----+
    ///  | 2   | 3   | 1   |
    ///  +-----+-----+-----+
    /// ```
    pub fn explode<I, S>(&self, columns: I) -> PolarsResult<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        // We need to sort the column by order of original occurrence. Otherwise the insert by index
        // below will panic
        let columns = self.select_series(columns)?;
        self.explode_impl(columns)
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    #[cfg(feature = "dtype-i8")]
    #[cfg_attr(miri, ignore)]
    fn test_explode() {
        let s0 = Series::new("a", &[1i8, 2, 3]);
        let s1 = Series::new("b", &[1i8, 1, 1]);
        let s2 = Series::new("c", &[2i8, 2, 2]);
        let list = Series::new("foo", &[s0, s1, s2]);

        let s0 = Series::new("B", [1, 2, 3]);
        let s1 = Series::new("C", [1, 1, 1]);
        let df = DataFrame::new(vec![list, s0.clone(), s1.clone()]).unwrap();
        let exploded = df.explode(["foo"]).unwrap();
        assert_eq!(exploded.shape(), (9, 3));
        assert_eq!(exploded.column("C").unwrap().i32().unwrap().get(8), Some(1));
        assert_eq!(exploded.column("B").unwrap().i32().unwrap().get(8), Some(3));
        assert_eq!(
            exploded.column("foo").unwrap().i8().unwrap().get(8),
            Some(2)
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_explode_df_empty_list() -> PolarsResult<()> {
        let s0 = Series::new("a", &[1, 2, 3]);
        let s1 = Series::new("b", &[1, 1, 1]);
        let list = Series::new("foo", &[s0, s1.clone(), s1.clear()]);
        let s0 = Series::new("B", [1, 2, 3]);
        let s1 = Series::new("C", [1, 1, 1]);
        let df = DataFrame::new(vec![list, s0.clone(), s1.clone()])?;

        let out = df.explode(["foo"])?;
        let expected = df![
            "foo" => [Some(1), Some(2), Some(3), Some(1), Some(1), Some(1), None],
            "B" => [1, 1, 1, 2, 2, 2, 3],
            "C" => [1, 1, 1, 1, 1, 1, 1],
        ]?;

        assert!(out.equals_missing(&expected));

        let list = Series::new("foo", [s0.clone(), s1.clear(), s1.clone()]);
        let df = DataFrame::new(vec![list, s0, s1])?;
        let out = df.explode(["foo"])?;
        let expected = df![
            "foo" => [Some(1), Some(2), Some(3), None, Some(1), Some(1), Some(1)],
            "B" => [1, 1, 1, 2, 3, 3, 3],
            "C" => [1, 1, 1, 1, 1, 1, 1],
        ]?;

        assert!(out.equals_missing(&expected));
        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_explode_single_col() -> PolarsResult<()> {
        let s0 = Series::new("a", &[1i32, 2, 3]);
        let s1 = Series::new("b", &[1i32, 1, 1]);
        let list = Series::new("foo", &[s0, s1]);
        let df = DataFrame::new(vec![list])?;

        let out = df.explode(["foo"])?;
        let out = out
            .column("foo")?
            .i32()?
            .into_no_null_iter()
            .collect::<Vec<_>>();
        assert_eq!(out, &[1i32, 2, 3, 1, 1, 1]);

        Ok(())
    }
}
