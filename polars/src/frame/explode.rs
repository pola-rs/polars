use crate::frame::select::Selection;
use crate::prelude::*;
use arrow::array::{Array, ListArray};
use std::collections::VecDeque;
use std::convert::TryFrom;

/// Convert Arrow array offsets to indexes of the original list
fn offsets_to_indexes(offsets: &[i32], capacity: usize) -> Vec<usize> {
    let mut idx = Vec::with_capacity(capacity);

    let mut count = 0;
    let mut last_idx = 0;
    for &offset in offsets.iter().skip(1) {
        while count < offset {
            count += 1;
            idx.push(last_idx)
        }
        last_idx += 1;
    }
    for _ in 0..(capacity - count as usize) {
        idx.push(last_idx);
    }
    idx
}

impl ListChunked {
    pub fn explode(&self) -> Result<(Series, &[i32])> {
        // A list array's memory layout is actually already 'exploded', so we can just take the values array
        // of the list. And we also return a slice of the offsets. This slice can be used to find the old
        // list layout or indexes to expand the DataFrame in the same manner as the 'explode' operation
        let ca = self.rechunk(Some(&[1])).unwrap();
        let listarr: &ListArray = ca.downcast_chunks()[0];
        let list_data = listarr.data();
        let values = listarr.values();
        let offset_ptr = list_data.buffers()[0].raw_data() as *const i32;
        // offsets in the list array. These indicate where a new list starts
        let offsets = unsafe { std::slice::from_raw_parts(offset_ptr, self.len()) };

        let s = Series::try_from((self.name(), values)).unwrap();
        Ok((s, offsets))
    }
}

impl DataFrame {
    /// Explode `DataFrame` to long format by exploding a column with Lists.
    ///
    /// # Example
    ///
    /// ```rust
    ///  use polars::prelude::*;
    ///  let s0 = Series::new("a", &[1i8, 2, 3]);
    ///  let s1 = Series::new("b", &[1i8, 1, 1]);
    ///  let s2 = Series::new("c", &[2i8, 2, 2]);
    ///  let list = Series::new("foo", &[s0, s1, s2]);
    ///
    ///  let s0 = Series::new("B", [1, 2, 3]);
    ///  let s1 = Series::new("C", [1, 1, 1]);
    ///  let df = DataFrame::new(vec![list, s0, s1]).unwrap();
    ///  let exploded = df.explode("foo").unwrap();
    ///
    ///  println!("{:?}", df);
    ///  println!("{:?}", exploded);
    /// ```
    /// Outputs:
    ///
    /// ```text
    ///  +-------------+-----+-----+
    ///  | foo         | B   | C   |
    ///  | ---         | --- | --- |
    ///  | list [i8]   | i32 | i32 |
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
    ///  | i8  | i32 | i32 |
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
    pub fn explode<'a, J, S: Selection<'a, J>>(&self, columns: S) -> Result<DataFrame> {
        let columns = self.select_series(columns)?;

        // first remove all the exploded columns
        let mut df = self.clone();
        for s in &columns {
            df = df.drop(s.name())?;
        }

        for (i, s) in columns.iter().enumerate() {
            if let Ok(ca) = s.list() {
                let (exploded, offsets) = ca.explode()?;
                let col_idx = self.name_to_idx(s.name())?;

                // expand all the other columns based the exploded first column
                if i == 0 {
                    let row_idx = offsets_to_indexes(offsets, exploded.len());
                    df = unsafe { df.take_iter_unchecked(row_idx.into_iter(), None) };
                }
                if exploded.len() == df.height() {
                    df.columns.insert(col_idx, exploded);
                } else {
                    return Err(PolarsError::ShapeMisMatch(
                        format!("The exploded columns don't have the same length. Length DataFrame: {}. Length exploded column {}: {}", df.height(), exploded.name(), exploded.len()).into(),
                    ));
                }
            } else {
                return Err(PolarsError::InvalidOperation(
                    format!("cannot explode dtype: {:?}", s.dtype()).into(),
                ));
            }
        }
        Ok(df)
    }

    ///
    /// Unpivot a `DataFrame` from wide to long format.
    ///
    /// # Example
    ///
    /// # Arguments
    ///
    /// * `id_vars` - String slice that represent the columns to use as id variables.
    /// * `value_vars` - String slice that represent the columns to use as value variables.
    ///
    /// ```rust
    ///
    ///  # #[macro_use] extern crate polars;
    /// use polars::prelude::*;
    /// let df = df!("A" => &["a", "b", "a"],
    ///              "B" => &[1, 3, 5],
    ///              "C" => &[10, 11, 12],
    ///              "D" => &[2, 4, 6]
    ///     )
    /// .unwrap();
    ///
    /// let melted = df.melt(&["A", "B"], &["C", "D"]).unwrap();
    /// println!("{:?}", df);
    /// println!("{:?}", melted);
    /// ```
    /// Outputs:
    /// ```text
    ///  +-----+-----+-----+-----+
    ///  | A   | B   | C   | D   |
    ///  | --- | --- | --- | --- |
    ///  | str | i32 | i32 | i32 |
    ///  +=====+=====+=====+=====+
    ///  | "a" | 1   | 10  | 2   |
    ///  +-----+-----+-----+-----+
    ///  | "b" | 3   | 11  | 4   |
    ///  +-----+-----+-----+-----+
    ///  | "a" | 5   | 12  | 6   |
    ///  +-----+-----+-----+-----+
    ///
    ///  +-----+-----+----------+-------+
    ///  | A   | B   | variable | value |
    ///  | --- | --- | ---      | ---   |
    ///  | str | i32 | str      | i32   |
    ///  +=====+=====+==========+=======+
    ///  | "a" | 1   | "C"      | 10    |
    ///  +-----+-----+----------+-------+
    ///  | "b" | 3   | "C"      | 11    |
    ///  +-----+-----+----------+-------+
    ///  | "a" | 5   | "C"      | 12    |
    ///  +-----+-----+----------+-------+
    ///  | "a" | 1   | "D"      | 2     |
    ///  +-----+-----+----------+-------+
    ///  | "b" | 3   | "D"      | 4     |
    ///  +-----+-----+----------+-------+
    ///  | "a" | 5   | "D"      | 6     |
    ///  +-----+-----+----------+-------+
    /// ```
    pub fn melt<'a, 'b, J, K, SelId: Selection<'a, J>, SelValue: Selection<'b, K>>(
        &self,
        id_vars: SelId,
        value_vars: SelValue,
    ) -> Result<Self> {
        let ids = self.select(id_vars)?;
        let value_vars = value_vars.to_selection_vec();
        let len = self.height();

        let mut dataframe_chunks = VecDeque::with_capacity(value_vars.len());

        for value_column_name in value_vars {
            let variable_col = Utf8Chunked::full("variable", value_column_name, len).into_series();
            let mut value_col = self.column(value_column_name)?.clone();
            value_col.rename("value");

            let mut df_chunk = ids.clone();
            df_chunk.hstack_mut(&[variable_col, value_col])?;
            dataframe_chunks.push_back(df_chunk)
        }

        let mut main_df = dataframe_chunks
            .pop_front()
            .ok_or_else(|| PolarsError::NoData("No data in melt operation".into()))?;

        while let Some(df) = dataframe_chunks.pop_front() {
            main_df.vstack_mut(&df)?;
        }
        Ok(main_df)
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_explode() {
        let s0 = Series::new("a", &[1i8, 2, 3]);
        let s1 = Series::new("b", &[1i8, 1, 1]);
        let s2 = Series::new("c", &[2i8, 2, 2]);
        let list = Series::new("foo", &[s0, s1, s2]);

        let s0 = Series::new("B", [1, 2, 3]);
        let s1 = Series::new("C", [1, 1, 1]);
        let df = DataFrame::new(vec![list, s0, s1]).unwrap();
        let exploded = df.explode("foo").unwrap();
        println!("{:?}", df);
        println!("{:?}", exploded);
        assert_eq!(exploded.shape(), (9, 3));
    }

    #[test]
    fn test_melt() {
        let df = df!("A" => &["a", "b", "a"],
         "B" => &[1, 3, 5],
         "C" => &[10, 11, 12],
         "D" => &[2, 4, 6]
        )
        .unwrap();

        let melted = df.melt(&["A", "B"], &["C", "D"]).unwrap();
        assert_eq!(
            Vec::from(melted.column("value").unwrap().i32().unwrap()),
            &[Some(10), Some(11), Some(12), Some(2), Some(4), Some(6)]
        )
    }
}
