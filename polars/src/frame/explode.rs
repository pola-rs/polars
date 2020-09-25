use crate::prelude::*;

impl LargeListChunked {
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
    ///  use polars::chunked_array::builder::get_large_list_builder;
    ///
    ///  let mut builder = get_large_list_builder(&ArrowDataType::Int8, 3, "foo");
    ///  builder.append_series(&Series::new("a", &[1i8, 2, 3]));
    ///  builder.append_series(&Series::new("b", &[1i8, 1, 1]));
    ///  builder.append_series(&Series::new("c", &[2i8, 2, 2]));
    ///  let list = builder.finish().into_series();
    ///
    ///  let s = Series::new("B", [1, 2, 3]);
    ///  let s1 = Series::new("C", [1, 1, 1]);
    ///  let df = DataFrame::new(vec![list, s, s1]).unwrap();
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
        if let Series::LargeList(ca) = s {
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
}

#[cfg(test)]
mod test {
    use crate::chunked_array::builder::get_large_list_builder;
    use crate::prelude::*;

    #[test]
    fn test_explode() {
        let mut builder = get_large_list_builder(&ArrowDataType::Int8, 3, "foo");
        builder.append_series(&Series::new("a", &[1i8, 2, 3]));
        builder.append_series(&Series::new("b", &[1i8, 1, 1]));
        builder.append_series(&Series::new("c", &[2i8, 2, 2]));
        let list = builder.finish().into_series();

        let s = Series::new("B", [1, 2, 3]);
        let s1 = Series::new("C", [1, 1, 1]);
        let df = DataFrame::new(vec![list, s, s1]).unwrap();
        let exploded = df.explode("foo").unwrap();
        println!("{:?}", df);
        println!("{:?}", exploded);
        assert_eq!(exploded.shape(), (9, 3));
    }
}
