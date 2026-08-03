use super::*;

impl DataFrame {
    /// Get a row from a [`DataFrame`]. Use of this is discouraged as it will likely be slow.
    pub fn get_row(&self, idx: usize) -> PolarsResult<Row<'_>> {
        let values = self
            .materialized_column_iter()
            .map(|s| s.get(idx))
            .collect::<PolarsResult<Vec<_>>>()?;
        Ok(Row(values))
    }

    /// Amortize allocations by reusing a row.
    /// The caller is responsible to make sure that the row has at least the capacity for the number
    /// of columns in the [`DataFrame`]
    pub fn get_row_amortized<'a>(&'a self, idx: usize, row: &mut Row<'a>) -> PolarsResult<()> {
        for (s, any_val) in self.materialized_column_iter().zip(&mut row.0) {
            *any_val = s.get(idx)?;
        }
        Ok(())
    }

    /// Amortize allocations by reusing a row.
    /// The caller is responsible to make sure that the row has at least the capacity for the number
    /// of columns in the [`DataFrame`]
    ///
    /// # Safety
    /// Does not do any bounds checking.
    #[inline]
    pub unsafe fn get_row_amortized_unchecked<'a>(&'a self, idx: usize, row: &mut Row<'a>) {
        self.materialized_column_iter()
            .zip(&mut row.0)
            .for_each(|(s, any_val)| {
                *any_val = s.get_unchecked(idx);
            });
    }

    /// Create a new [`DataFrame`] from rows.
    ///
    /// This should only be used when you have row wise data, as this is a lot slower
    /// than creating the [`Series`] in a columnar fashion
    pub fn from_rows_and_schema(rows: &[Row], schema: &Schema) -> PolarsResult<Self> {
        Self::from_rows_iter_and_schema(rows.iter(), schema)
    }

    /// Like [`Self::from_rows_and_schema`] but rejects values that do not match
    /// the schema dtype. If `strict_columns` is given, only columns marked `true`
    /// are validated; the rest are coerced.
    pub fn from_rows_and_schema_strict(
        rows: &[Row],
        schema: &Schema,
        strict_columns: Option<&[bool]>,
    ) -> PolarsResult<Self> {
        let n_cols = schema.len();
        if let Some(sc) = strict_columns {
            polars_ensure!(
                sc.len() == n_cols,
                ShapeMismatch:
                "length of `strict_columns` ({}) does not match the schema width ({})",
                sc.len(),
                n_cols,
            );
        }

        let n_rows = rows.len();
        let mut columns: Vec<Vec<AnyValue>> = Vec::with_capacity(n_cols);
        for _ in 0..n_cols {
            columns.push(Vec::with_capacity(n_rows));
        }
        for row in rows {
            for (i, col) in columns.iter_mut().enumerate() {
                let val = row.0.get(i).cloned().unwrap_or(AnyValue::Null);
                col.push(val);
            }
        }

        let v: Vec<Column> = schema
            .iter()
            .enumerate()
            .map(|(i, (name, dtype))| {
                // Null columns must always be strict to avoid silently dropping values.
                let strict = dtype == &DataType::Null || strict_columns.is_none_or(|cols| cols[i]);
                let s =
                    Series::from_any_values_and_dtype(name.clone(), &columns[i], dtype, strict)?;
                Ok(s.into_column())
            })
            .collect::<PolarsResult<_>>()?;

        DataFrame::new(n_rows, v)
    }

    /// Create a new [`DataFrame`] from an iterator over rows.
    ///
    /// This should only be used when you have row wise data, as this is a lot slower
    /// than creating the [`Series`] in a columnar fashion.
    pub fn from_rows_iter_and_schema<'a, I>(mut rows: I, schema: &Schema) -> PolarsResult<Self>
    where
        I: Iterator<Item = &'a Row<'a>>,
    {
        if schema.is_empty() {
            let height = rows.count();
            let columns = Vec::new();
            return Ok(unsafe { DataFrame::new_unchecked(height, columns) });
        }

        let capacity = rows.size_hint().0;

        let mut buffers: Vec<_> = schema
            .iter_values()
            .map(|dtype| {
                let buf: AnyValueBuffer = (dtype, capacity).into();
                buf
            })
            .collect();

        let mut expected_len = 0;
        rows.try_for_each::<_, PolarsResult<()>>(|row| {
            expected_len += 1;
            for (value, buf) in row.0.iter().zip(&mut buffers) {
                buf.add_fallible(value)?
            }
            Ok(())
        })?;

        let v = buffers
            .into_iter()
            .zip(schema.iter_names())
            .map(|(b, name)| {
                let mut c = b.into_series()?.into_column();
                // if the schema adds a column not in the rows, we
                // fill it with nulls
                if c.is_empty() {
                    Ok(Column::full_null(name.clone(), expected_len, c.dtype()))
                } else {
                    c.rename(name.clone());
                    Ok(c)
                }
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        DataFrame::new(expected_len, v)
    }

    /// Create a new [`DataFrame`] from an iterator over rows. This should only be used when you have row wise data,
    /// as this is a lot slower than creating the [`Series`] in a columnar fashion
    pub fn try_from_rows_iter_and_schema<'a, I>(mut rows: I, schema: &Schema) -> PolarsResult<Self>
    where
        I: Iterator<Item = PolarsResult<&'a Row<'a>>>,
    {
        let capacity = rows.size_hint().0;

        let mut buffers: Vec<_> = schema
            .iter_values()
            .map(|dtype| {
                let buf: AnyValueBuffer = (dtype, capacity).into();
                buf
            })
            .collect();

        let mut expected_len = 0;
        rows.try_for_each::<_, PolarsResult<()>>(|row| {
            expected_len += 1;
            for (value, buf) in row?.0.iter().zip(&mut buffers) {
                buf.add_fallible(value)?
            }
            Ok(())
        })?;
        let v = buffers
            .into_iter()
            .zip(schema.iter_names())
            .map(|(b, name)| {
                let mut c = b.into_series()?.into_column();
                // if the schema adds a column not in the rows, we
                // fill it with nulls
                if c.is_empty() {
                    Ok(Column::full_null(name.clone(), expected_len, c.dtype()))
                } else {
                    c.rename(name.clone());
                    Ok(c)
                }
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        DataFrame::new(expected_len, v)
    }

    /// Create a new [`DataFrame`] from rows. This should only be used when you have row wise data,
    /// as this is a lot slower than creating the [`Series`] in a columnar fashion
    pub fn from_rows(rows: &[Row]) -> PolarsResult<Self> {
        let schema = rows_to_schema_first_non_null(rows, Some(50))?;
        let has_nulls = schema
            .iter_values()
            .any(|dtype| matches!(dtype, DataType::Null));
        polars_ensure!(
            !has_nulls, ComputeError: "unable to infer row types because of null values"
        );
        Self::from_rows_and_schema(rows, &schema)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strict_row_construction_rejects_mismatch() {
        let schema = Schema::from_iter([Field::new("a".into(), DataType::Int64)]);
        let rows = vec![Row(vec![AnyValue::Float64(1.5)])];
        assert!(DataFrame::from_rows_and_schema_strict(&rows, &schema, None).is_err());
    }

    #[test]
    fn test_strict_row_construction_accepts_match() {
        let schema = Schema::from_iter([Field::new("a".into(), DataType::Int64)]);
        let rows = vec![Row(vec![AnyValue::Int64(1)])];
        assert!(DataFrame::from_rows_and_schema_strict(&rows, &schema, None).is_ok());
    }

    #[test]
    fn test_strict_row_per_column_strictness() {
        let schema = Schema::from_iter([
            Field::new("a".into(), DataType::Float64),
            Field::new("b".into(), DataType::Float64),
        ]);
        let rows = vec![Row(vec![AnyValue::Int64(1), AnyValue::Int64(1)])];

        assert!(DataFrame::from_rows_and_schema_strict(&rows, &schema, None).is_err());
        let df =
            DataFrame::from_rows_and_schema_strict(&rows, &schema, Some(&[false, false])).unwrap();
        assert_eq!(df.column("a").unwrap().dtype(), &DataType::Float64);
        assert!(
            DataFrame::from_rows_and_schema_strict(&rows, &schema, Some(&[false, true])).is_err()
        );
        assert!(DataFrame::from_rows_and_schema_strict(&rows, &schema, Some(&[false])).is_err());
    }

    #[test]
    fn test_strict_row_short_rows_padded_with_null() {
        let schema = Schema::from_iter([
            Field::new("a".into(), DataType::Int64),
            Field::new("b".into(), DataType::Int64),
        ]);
        let rows = vec![Row(vec![AnyValue::Int64(1)])];
        let df =
            DataFrame::from_rows_and_schema_strict(&rows, &schema, Some(&[false, false])).unwrap();
        assert_eq!(df.height(), 1);
        assert_eq!(df.width(), 2);
        assert!(df.column("b").unwrap().get(0).unwrap().is_null());
    }

    #[test]
    fn test_strict_row_empty_rows() {
        let schema = Schema::from_iter([Field::new("a".into(), DataType::Int64)]);
        let rows: Vec<Row> = vec![];
        let df = DataFrame::from_rows_and_schema_strict(&rows, &schema, Some(&[true])).unwrap();
        assert_eq!(df.height(), 0);
        assert_eq!(df.width(), 1);
    }
}
