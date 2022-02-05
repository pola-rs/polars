use super::*;
use polars_arrow::array::ValueSize;

/// Specialized expressions for [`Series`] of [`DataType::Utf8`].
pub struct StringNameSpace(pub(crate) Expr);

impl StringNameSpace {
    pub fn extract(self, pat: &str, group_index: usize) -> Expr {
        let pat = pat.to_string();
        let function = move |s: Series| {
            let ca = s.utf8()?;
            ca.extract(&pat, group_index).map(|ca| ca.into_series())
        };
        self.0
            .map(function, GetOutput::from_type(DataType::Utf8))
            .with_fmt("str.extract")
    }

    #[cfg(feature = "temporal")]
    pub fn strptime(self, options: StrpTimeOptions) -> Expr {
        let out_type = options.date_dtype.clone();
        let function = move |s: Series| {
            let ca = s.utf8()?;

            let out = match &options.date_dtype {
                DataType::Date => {
                    if options.exact {
                        ca.as_date(options.fmt.as_deref())?.into_series()
                    } else {
                        ca.as_date_not_exact(options.fmt.as_deref())?.into_series()
                    }
                }
                DataType::Datetime(tu, _) => {
                    if options.exact {
                        ca.as_datetime(options.fmt.as_deref(), *tu)?.into_series()
                    } else {
                        ca.as_datetime_not_exact(options.fmt.as_deref(), *tu)?
                            .into_series()
                    }
                }
                dt => {
                    return Err(PolarsError::ComputeError(
                        format!("not implemented for dtype {:?}", dt).into(),
                    ))
                }
            };
            if options.strict {
                if out.null_count() != ca.null_count() {
                    Err(PolarsError::ComputeError(
                        "strict conversion to dates failed, maybe set strict=False".into(),
                    ))
                } else {
                    Ok(out.into_series())
                }
            } else {
                Ok(out.into_series())
            }
        };
        self.0
            .map(function, GetOutput::from_type(out_type))
            .with_fmt("str.strptime")
    }

    #[cfg(feature = "concat_str")]
    /// Concat the values into a string array.
    /// # Arguments
    ///
    /// * `delimiter` - A string that will act as delimiter between values.
    pub fn concat(self, delimiter: &str) -> Expr {
        let delimiter = delimiter.to_owned();
        let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
            Ok(s[0].str_concat(&delimiter).into_series())
        }) as Arc<dyn SeriesUdf>);
        Expr::Function {
            input: vec![self.0],
            function,
            output_type: GetOutput::from_type(DataType::Utf8),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyGroups,
                input_wildcard_expansion: false,
                auto_explode: true,
                fmt_str: "str.concat",
            },
        }
    }

    /// Split the string by a substring.
    pub fn split(self, by: &str) -> Expr {
        let by = by.to_string();

        let function = move |s: Series| {
            let ca = s.utf8()?;

            let mut builder = ListUtf8ChunkedBuilder::new(s.name(), s.len(), ca.get_values_size());
            ca.into_iter().for_each(|opt_s| match opt_s {
                None => builder.append_null(),
                Some(s) => {
                    let iter = s.split(&by);
                    builder.append_values_iter(iter);
                }
            });
            Ok(builder.finish().into_series())
        };
        self.0
            .map(
                function,
                GetOutput::from_type(DataType::List(Box::new(DataType::Utf8))),
            )
            .with_fmt("str.split")
    }

    /// Split the string by a substring.
    pub fn split_inclusive(self, by: &str) -> Expr {
        let by = by.to_string();

        let function = move |s: Series| {
            let ca = s.utf8()?;

            let mut builder = ListUtf8ChunkedBuilder::new(s.name(), s.len(), ca.get_values_size());
            ca.into_iter().for_each(|opt_s| match opt_s {
                None => builder.append_null(),
                Some(s) => {
                    let iter = s.split_inclusive(&by);
                    builder.append_values_iter(iter);
                }
            });
            Ok(builder.finish().into_series())
        };
        self.0
            .map(
                function,
                GetOutput::from_type(DataType::List(Box::new(DataType::Utf8))),
            )
            .with_fmt("str.split_inclusive")
    }
}
