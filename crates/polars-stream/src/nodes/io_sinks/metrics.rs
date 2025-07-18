use arrow::array::builder::ShareStrategy;
use polars_core::frame::DataFrame;
use polars_core::prelude::{
    AnyValue, ChunkedBuilder, DataType, IntoColumn, PrimitiveChunkedBuilder, StringChunkedBuilder,
    StructChunked, UInt64Type,
};
use polars_core::schema::Schema;
use polars_core::series::builder::SeriesBuilder;
use polars_core::series::{IntoSeries, Series};
use polars_error::PolarsResult;
use polars_expr::reduce::{GroupedReduction, new_max_reduction, new_min_reduction};
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;

/// Metrics that relate to a written file.
pub struct WriteMetrics {
    /// Stringified path to the file.
    pub path: String,
    /// Number of rows in the file.
    pub num_rows: u64,
    /// Size of written file in bytes.
    pub file_size: u64,
    /// Keys of the partition.
    pub keys: Option<Vec<AnyValue<'static>>>,
    /// Metrics for each column.
    pub columns: Vec<WriteMetricsColumn>,
}

/// Metrics in a written file for a specific column.
pub struct WriteMetricsColumn {
    /// Number of missing values in the column.
    pub null_count: u64,
    /// Number of NaN values in the column.
    pub nan_count: u64,
    /// The minimum value in the column.
    ///
    /// `NaN`s are always ignored and `None` is the default value.
    pub lower_bound: Option<Box<dyn GroupedReduction>>,
    /// The maximum value in the column.
    ///
    /// `NaN`s are always ignored and `None` is the default value.
    pub upper_bound: Option<Box<dyn GroupedReduction>>,
}

impl WriteMetrics {
    pub fn new(path: String, schema: &Schema) -> Self {
        Self {
            path,
            file_size: 0,
            num_rows: 0,
            keys: None,
            columns: schema
                .iter_values()
                .cloned()
                .map(WriteMetricsColumn::new)
                .collect(),
        }
    }

    pub fn append(&mut self, df: &DataFrame) -> PolarsResult<()> {
        assert_eq!(self.columns.len(), df.width());
        self.num_rows += df.height() as u64;
        for (w, c) in self.columns.iter_mut().zip(df.get_columns()) {
            let null_count = c.null_count();
            w.null_count += c.null_count() as u64;

            let mut has_non_null_non_nan_values = df.height() != null_count;
            if c.dtype().is_float() {
                let nan_count = c.is_nan()?.sum().unwrap_or_default();
                has_non_null_non_nan_values = nan_count as usize + null_count < df.height();
                #[allow(clippy::useless_conversion)]
                {
                    w.nan_count += u64::from(nan_count);
                }
            }

            if has_non_null_non_nan_values {
                if let Some(lb) = &mut w.lower_bound {
                    lb.update_group(c, 0, 0)?;
                }
                if let Some(ub) = &mut w.upper_bound {
                    ub.update_group(c, 0, 0)?;
                }
            }
        }
        Ok(())
    }

    pub fn collapse_to_df(
        metrics: Vec<Self>,
        input_schema: &Schema,
        key_schema: Option<&Schema>,
    ) -> DataFrame {
        let num_metrics = metrics.len();

        let mut path = StringChunkedBuilder::new(PlSmallStr::from_static("path"), num_metrics);
        let mut num_rows = PrimitiveChunkedBuilder::<UInt64Type>::new(
            PlSmallStr::from_static("num_rows"),
            num_metrics,
        );
        let mut file_size = PrimitiveChunkedBuilder::<UInt64Type>::new(
            PlSmallStr::from_static("file_size"),
            num_metrics,
        );
        let mut keys = key_schema.map(|s| {
            (0..s.len())
                .map(|_| Vec::with_capacity(metrics.len()))
                .collect::<Vec<_>>()
        });
        let mut columns = input_schema
            .iter_values()
            .map(|dtype| {
                let null_count = PrimitiveChunkedBuilder::<UInt64Type>::new(
                    PlSmallStr::from_static("null_count"),
                    num_metrics,
                );
                let nan_count = PrimitiveChunkedBuilder::<UInt64Type>::new(
                    PlSmallStr::from_static("nan_count"),
                    num_metrics,
                );
                let mut lower_bound = SeriesBuilder::new(dtype.clone());
                let mut upper_bound = SeriesBuilder::new(dtype.clone());
                lower_bound.reserve(num_metrics);
                upper_bound.reserve(num_metrics);

                (null_count, nan_count, lower_bound, upper_bound)
            })
            .collect::<Vec<_>>();

        for m in metrics {
            path.append_value(m.path);
            num_rows.append_value(m.num_rows);
            file_size.append_value(m.file_size);
            match (&mut keys, m.keys) {
                (None, None) => {},
                (Some(keys), Some(m_keys)) => {
                    for (key, m_key) in keys.iter_mut().zip(m_keys) {
                        key.push(m_key);
                    }
                },
                _ => unreachable!(),
            }

            for (mut w, c) in m.columns.into_iter().zip(columns.iter_mut()) {
                c.0.append_value(w.null_count);
                c.1.append_value(w.nan_count);
                match &mut w.lower_bound {
                    None => c.2.extend_nulls(1),
                    Some(lb) => c.2.extend(&lb.finalize().unwrap(), ShareStrategy::Always),
                }
                match &mut w.upper_bound {
                    None => c.3.extend_nulls(1),
                    Some(ub) => c.3.extend(&ub.finalize().unwrap(), ShareStrategy::Always),
                }
            }
        }

        let mut df_columns = Vec::with_capacity(4 + input_schema.len());
        df_columns.push(path.finish().into_column());
        df_columns.push(num_rows.finish().into_column());
        df_columns.push(file_size.finish().into_column());
        match (keys, key_schema) {
            (None, None) => df_columns.push(
                StructChunked::from_series(
                    PlSmallStr::from_static("keys"),
                    num_metrics,
                    [].into_iter(),
                )
                .unwrap()
                .into_column(),
            ),
            (Some(keys), Some(key_schema)) => {
                let keys = keys
                    .into_iter()
                    .zip(key_schema.iter())
                    .map(|(key, (name, dtype))| {
                        Series::from_any_values_and_dtype(name.clone(), key.as_slice(), dtype, true)
                            .unwrap()
                    })
                    .collect::<Vec<Series>>();
                df_columns.push(
                    StructChunked::from_series(
                        PlSmallStr::from_static("keys"),
                        num_metrics,
                        keys.iter(),
                    )
                    .unwrap()
                    .into_column(),
                );
            },
            _ => unreachable!(),
        }
        for (name, column) in input_schema.iter_names().zip(columns) {
            let struct_ca = StructChunked::from_series(
                format_pl_smallstr!("{name}_stats"),
                num_metrics,
                [
                    column.0.finish().into_series(),
                    column.1.finish().into_series(),
                    column.2.freeze(PlSmallStr::from_static("lower_bound")),
                    column.3.freeze(PlSmallStr::from_static("upper_bound")),
                ]
                .iter(),
            )
            .unwrap();
            df_columns.push(struct_ca.into_column());
        }

        DataFrame::new_with_height(num_metrics, df_columns).unwrap()
    }
}

impl WriteMetricsColumn {
    pub fn new(dtype: DataType) -> Self {
        let (lower_bound, upper_bound) = if dtype.is_nested() {
            (None, None)
        } else {
            let mut lower_bound = new_min_reduction(dtype.clone(), false);
            let mut upper_bound = new_max_reduction(dtype, false);

            lower_bound.resize(1);
            upper_bound.resize(1);

            (Some(lower_bound), Some(upper_bound))
        };

        Self {
            null_count: 0,
            nan_count: 0,
            lower_bound,
            upper_bound,
        }
    }
}
