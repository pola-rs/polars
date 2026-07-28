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

    /// Create a new [`DataFrame`] from rows, rejecting values that do not match the schema.
    ///
    /// A value is accepted exactly when strict [`Series::from_any_values_and_dtype`]
    /// accepts it for the column's dtype. If `strict_columns` is given, only the
    /// columns where it is true are validated; values in the remaining columns are
    /// coerced as in [`DataFrame::from_rows_and_schema`].
    ///
    /// This should only be used when you have row wise data, as this is a lot slower
    /// than creating the [`Series`] in a columnar fashion.
    pub fn from_rows_and_schema_strict(
        rows: &[Row],
        schema: &Schema,
        strict_columns: Option<&[bool]>,
    ) -> PolarsResult<Self> {
        if let Some(strict_columns) = strict_columns {
            polars_ensure!(
                strict_columns.len() == schema.len(),
                ShapeMismatch:
                "length of `strict_columns` ({}) does not match the schema width ({})",
                strict_columns.len(),
                schema.len(),
            );
        }
        Self::from_rows_iter_and_schema_impl::<_, true>(rows.iter(), schema, strict_columns)
    }

    /// Create a new [`DataFrame`] from an iterator over rows.
    ///
    /// This should only be used when you have row wise data, as this is a lot slower
    /// than creating the [`Series`] in a columnar fashion.
    pub fn from_rows_iter_and_schema<'a, I>(rows: I, schema: &Schema) -> PolarsResult<Self>
    where
        I: Iterator<Item = &'a Row<'a>>,
    {
        Self::from_rows_iter_and_schema_impl::<_, false>(rows, schema, None)
    }

    // `STRICT` is a const generic so that the all-lenient instantiation folds the
    // per-column flags away; a runtime flag measures ~3-4% slower in that hot loop.
    fn from_rows_iter_and_schema_impl<'a, I, const STRICT: bool>(
        mut rows: I,
        schema: &Schema,
        strict_columns: Option<&[bool]>,
    ) -> PolarsResult<Self>
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
            .enumerate()
            .map(|(i, dtype)| {
                let buf: AnyValueBuffer = (dtype, capacity).into();
                let strict = STRICT && strict_columns.is_none_or(|cols| cols[i]);
                (buf, strict)
            })
            .collect();

        let mut expected_len = 0;
        rows.try_for_each::<_, PolarsResult<()>>(|row| {
            expected_len += 1;
            for (value, (buf, strict)) in row.0.iter().zip(&mut buffers) {
                if *strict {
                    buf.add_strict(value)?
                } else {
                    buf.add_fallible(value)?
                }
            }
            Ok(())
        })?;

        let v = buffers
            .into_iter()
            .zip(schema.iter_names())
            .map(|((mut b, strict), name)| {
                let mut c = b.reset(0, strict)?.into_column();
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

    fn build_strict(dtype: DataType, values: Vec<AnyValue<'static>>) -> PolarsResult<DataFrame> {
        let rows = values
            .into_iter()
            .map(|value| Row(vec![value]))
            .collect::<Vec<_>>();
        let schema = Schema::from_iter([Field::new("a".into(), dtype)]);
        DataFrame::from_rows_and_schema_strict(&rows, &schema, None)
    }

    #[test]
    fn test_add_strict_per_buffer_arm() {
        // One matching and one mismatching value per buffer arm; the value-level
        // semantics themselves are shared with `Series::from_any_values_and_dtype`
        // and tested through the Python API.
        #[allow(unused_mut)]
        let mut cases: Vec<(DataType, AnyValue, AnyValue)> = vec![
            (
                DataType::Boolean,
                AnyValue::Boolean(true),
                AnyValue::Int64(1),
            ),
            (DataType::Int32, AnyValue::Int64(1), AnyValue::Float64(1.0)),
            // integers are range-checked: a value outside the target width errors
            (
                DataType::Int32,
                AnyValue::Int64(i32::MAX as i64),
                AnyValue::Int64(i32::MAX as i64 + 1),
            ),
            (DataType::Int64, AnyValue::UInt64(1), AnyValue::Float64(1.0)),
            (DataType::UInt32, AnyValue::Int64(1), AnyValue::Int64(-1)),
            (DataType::UInt64, AnyValue::Int64(1), AnyValue::Int64(-1)),
            (
                DataType::Float32,
                AnyValue::Float32(1.0),
                AnyValue::Float64(1.0),
            ),
            (
                DataType::Float64,
                AnyValue::Float32(1.0),
                AnyValue::Int64(1),
            ),
            (
                DataType::String,
                AnyValue::StringOwned("a".into()),
                AnyValue::BinaryOwned(b"a".to_vec()),
            ),
            (DataType::Null, AnyValue::Null, AnyValue::Int64(1)),
        ];
        #[cfg(feature = "dtype-f16")]
        {
            use polars_utils::float16::pf16;
            let f16 = AnyValue::Float16(pf16::from(1.0f32));
            cases.push((DataType::Float32, f16.clone(), AnyValue::Int64(1)));
            cases.push((DataType::Float64, f16, AnyValue::Int64(1)));
        }
        #[cfg(feature = "dtype-date")]
        cases.push((
            DataType::Date,
            AnyValue::Date(19000),
            AnyValue::Int32(19000),
        ));
        #[cfg(feature = "dtype-time")]
        cases.push((
            DataType::Time,
            AnyValue::Time(1_000),
            AnyValue::Int64(1_000),
        ));
        #[cfg(feature = "dtype-datetime")]
        cases.push((
            DataType::Datetime(TimeUnit::Microseconds, None),
            AnyValue::Datetime(1_000, TimeUnit::Microseconds, None),
            // a mismatched time unit is not rescaled under strict construction
            AnyValue::Datetime(1_000, TimeUnit::Nanoseconds, None),
        ));
        #[cfg(feature = "dtype-duration")]
        cases.push((
            DataType::Duration(TimeUnit::Milliseconds),
            AnyValue::Duration(1_000, TimeUnit::Milliseconds),
            AnyValue::Duration(1_000, TimeUnit::Microseconds),
        ));

        for (dtype, valid, invalid) in cases {
            assert!(
                build_strict(dtype.clone(), vec![valid, AnyValue::Null]).is_ok(),
                "valid value rejected for {dtype:?}"
            );
            assert!(
                build_strict(dtype.clone(), vec![invalid]).is_err(),
                "invalid value accepted for {dtype:?}"
            );
        }
    }

    #[test]
    fn test_strict_row_per_column_strictness() {
        let schema = Schema::from_iter([
            Field::new("a".into(), DataType::Float64),
            Field::new("b".into(), DataType::Float64),
        ]);
        let rows = vec![Row(vec![AnyValue::Int64(1), AnyValue::Int64(1)])];

        // Int64 into a strict Float64 column errors, a lenient column coerces.
        assert!(DataFrame::from_rows_and_schema_strict(&rows, &schema, None).is_err());
        let df =
            DataFrame::from_rows_and_schema_strict(&rows, &schema, Some(&[false, false])).unwrap();
        assert_eq!(df.column("a").unwrap().dtype(), &DataType::Float64);
        assert!(
            DataFrame::from_rows_and_schema_strict(&rows, &schema, Some(&[false, true])).is_err()
        );

        // The mask length must match the schema width.
        assert!(DataFrame::from_rows_and_schema_strict(&rows, &schema, Some(&[false])).is_err());
    }
}
