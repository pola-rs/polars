use crate::frame::select::Selection;
use crate::prelude::*;
use std::collections::VecDeque;

impl ListChunked {
    pub fn explode(&self) -> Result<(Series, Vec<usize>)> {
        macro_rules! impl_with_builder {
            ($self:expr, $builder:expr, $dtype:ty) => {{
                let mut row_idx = Vec::with_capacity($self.len() * 10);

                for i in 0..$self.len() {
                    match $self.get(i) {
                        Some(series) => {
                            let ca = series.unpack::<$dtype>()?;
                            if ca.null_count() == 0 {
                                ca.into_no_null_iter().for_each(|v| {
                                    $builder.append_value(v);
                                    row_idx.push(i)
                                })
                            } else {
                                ca.into_iter().for_each(|opt_v| {
                                    $builder.append_option(opt_v);
                                    row_idx.push(i)
                                })
                            }
                        }
                        None => {
                            $builder.append_null();
                            row_idx.push(i)
                        }
                    }
                }
                let exploded = $builder.finish().into_series();
                Ok((exploded, row_idx))
            }};
        }

        macro_rules! impl_primitive {
            ($dtype:ty, $self:expr) => {{
                // the 10 is an avg length of 10 elements in every Series.
                // A better alternative?
                let mut builder =
                    PrimitiveChunkedBuilder::<$dtype>::new($self.name(), $self.len() * 10);
                impl_with_builder!(self, builder, $dtype)
            }};
        }
        macro_rules! impl_utf8 {
            ($self:expr) => {{
                let mut builder = Utf8ChunkedBuilder::new($self.name(), $self.len() * 10);
                impl_with_builder!(self, builder, Utf8Type)
            }};
        }

        match_arrow_data_type_apply_macro!(
            **self.get_inner_dtype(),
            impl_primitive,
            impl_utf8,
            self
        )
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
    pub fn explode(&self, column: &str) -> Result<DataFrame> {
        let s = self.column(column)?;
        if let Series::List(ca) = s {
            let (exploded, row_idx) = ca.explode()?;
            let col_idx = self.name_to_idx(column)?;
            let df = self.drop(column)?;
            let mut df = unsafe { df.take_iter_unchecked(row_idx.into_iter(), None) };
            df.columns.insert(col_idx, exploded);
            Ok(df)
        } else {
            Ok(self.clone())
        }
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
            df_chunk.hstack(&[variable_col, value_col])?;
            dataframe_chunks.push_back(df_chunk)
        }

        let mut main_df = dataframe_chunks
            .pop_front()
            .ok_or(PolarsError::NoData("No data in melt operation".into()))?;

        while let Some(df) = dataframe_chunks.pop_front() {
            main_df.vstack(&df)?;
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
