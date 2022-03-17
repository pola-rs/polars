use super::*;
use polars_arrow::array::ValueSize;
use polars_arrow::export::arrow::array::{MutableArray, MutableUtf8Array};
use polars_time::prelude::*;

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
    // Split exactly `n` times by a given substring. The resulting dtype is `List<Utf8>`.
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

    #[cfg(feature = "dtype-struct")]
    // Split exactly `n` times by a given substring. The resulting dtype is [`DataType::Struct`].
    pub fn split_exact(self, by: &str, n: usize) -> Expr {
        let by = by.to_string();

        let function = move |s: Series| {
            let ca = s.utf8()?;

            let mut arrs = (0..n + 1)
                .map(|_| MutableUtf8Array::<i64>::with_capacity(ca.len()))
                .collect::<Vec<_>>();

            ca.into_iter().for_each(|opt_s| match opt_s {
                None => {
                    for arr in &mut arrs {
                        arr.push_null()
                    }
                }
                Some(s) => {
                    let mut arr_iter = arrs.iter_mut();
                    let split_iter = s.split(&by);
                    (split_iter)
                        .zip(&mut arr_iter)
                        .for_each(|(splitted, arr)| arr.push(Some(splitted)));
                    // fill the remaining with null
                    for arr in arr_iter {
                        arr.push_null()
                    }
                }
            });
            let fields = arrs
                .into_iter()
                .enumerate()
                .map(|(i, arr)| {
                    Series::try_from((format!("field_{i}").as_str(), arr.into_arc())).unwrap()
                })
                .collect::<Vec<_>>();
            Ok(StructChunked::new(ca.name(), &fields)?.into_series())
        };
        self.0
            .map(
                function,
                GetOutput::from_type(DataType::Struct(
                    (0..n + 1)
                        .map(|i| Field::new(&format!("field_{i}"), DataType::Utf8))
                        .collect(),
                )),
            )
            .with_fmt("str.split_exact")
    }

    #[cfg(feature = "dtype-struct")]
    // Split exactly `n` times by a given substring and keep the substring.
    // The resulting dtype is [`DataType::Struct`].
    pub fn split_exact_inclusive(self, by: &str, n: usize) -> Expr {
        let by = by.to_string();

        let function = move |s: Series| {
            let ca = s.utf8()?;

            let mut arrs = (0..n + 1)
                .map(|_| MutableUtf8Array::<i64>::with_capacity(ca.len()))
                .collect::<Vec<_>>();

            ca.into_iter().for_each(|opt_s| match opt_s {
                None => {
                    for arr in &mut arrs {
                        arr.push_null()
                    }
                }
                Some(s) => {
                    let mut arr_iter = arrs.iter_mut();
                    let split_iter = s.split_inclusive(&by);
                    (split_iter)
                        .zip(&mut arr_iter)
                        .for_each(|(splitted, arr)| arr.push(Some(splitted)));
                    // fill the remaining with null
                    for arr in arr_iter {
                        arr.push_null()
                    }
                }
            });
            let fields = arrs
                .into_iter()
                .enumerate()
                .map(|(i, arr)| {
                    Series::try_from((format!("field_{i}").as_str(), arr.into_arc())).unwrap()
                })
                .collect::<Vec<_>>();
            Ok(StructChunked::new(ca.name(), &fields)?.into_series())
        };
        self.0
            .map(
                function,
                GetOutput::from_type(DataType::Struct(
                    (0..n + 1)
                        .map(|i| Field::new(&format!("field_{i}"), DataType::Utf8))
                        .collect(),
                )),
            )
            .with_fmt("str.split_exact")
    }

    /// Split the string by a substring and keep the substring.
    // Split exactly `n` times by a given substring. The resulting dtype is `List<Utf8>`.
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
