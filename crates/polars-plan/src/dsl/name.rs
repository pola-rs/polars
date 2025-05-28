#[cfg(feature = "dtype-struct")]
use polars_utils::pl_str::PlSmallStr;

use super::*;

/// Specialized expressions for modifying the name of existing expressions.
pub struct ExprNameNameSpace(pub(crate) Expr);

impl ExprNameNameSpace {
    /// Keep the original root name
    ///
    /// ```rust,no_run
    /// # use polars_core::prelude::*;
    /// # use polars_plan::prelude::*;
    /// fn example(df: LazyFrame) -> LazyFrame {
    ///     df.select([
    /// // even thought the alias yields a different column name,
    /// // `keep` will make sure that the original column name is used
    ///         col("*").alias("foo").name().keep()
    /// ])
    /// }
    /// ```
    pub fn keep(self) -> Expr {
        Expr::KeepName(Arc::new(self.0))
    }

    /// Define an alias by mapping a function over the original root column name.
    pub fn map<F>(self, function: F) -> Expr
    where
        F: Fn(&PlSmallStr) -> PolarsResult<PlSmallStr> + 'static + Send + Sync,
    {
        let function = SpecialEq::new(Arc::new(function) as Arc<RenameAliasRustFn>);
        Expr::RenameAlias {
            expr: Arc::new(self.0),
            function: RenameAliasFn::Rust(function),
        }
    }

    /// Define an alias by mapping a python lambda over the original root column name.
    #[cfg(feature = "python")]
    pub fn map_udf(self, function: polars_utils::python_function::PythonObject) -> Expr {
        Expr::RenameAlias {
            expr: Arc::new(self.0),
            function: RenameAliasFn::Python(Arc::new(function)),
        }
    }

    /// Add a prefix to the root column name.
    pub fn prefix(self, prefix: &str) -> Expr {
        Expr::RenameAlias {
            expr: Arc::new(self.0),
            function: RenameAliasFn::Prefix(prefix.into()),
        }
    }

    /// Add a suffix to the root column name.
    pub fn suffix(self, suffix: &str) -> Expr {
        Expr::RenameAlias {
            expr: Arc::new(self.0),
            function: RenameAliasFn::Suffix(suffix.into()),
        }
    }

    /// Update the root column name to use lowercase characters.
    #[allow(clippy::wrong_self_convention)]
    pub fn to_lowercase(self) -> Expr {
        Expr::RenameAlias {
            expr: Arc::new(self.0),
            function: RenameAliasFn::ToLowercase,
        }
    }

    /// Update the root column name to use uppercase characters.
    #[allow(clippy::wrong_self_convention)]
    pub fn to_uppercase(self) -> Expr {
        Expr::RenameAlias {
            expr: Arc::new(self.0),
            function: RenameAliasFn::ToUppercase,
        }
    }

    #[cfg(feature = "dtype-struct")]
    pub fn map_fields(self, function: FieldsNameMapper) -> Expr {
        let f = function.clone();
        self.0.map(
            move |s| {
                let s = s.struct_()?;
                let fields = s
                    .fields_as_series()
                    .iter()
                    .map(|fd| {
                        let mut fd = fd.clone();
                        fd.rename(function(fd.name()));
                        fd
                    })
                    .collect::<Vec<_>>();
                let mut out = StructChunked::from_series(s.name().clone(), s.len(), fields.iter())?;
                out.zip_outer_validity(s);
                Ok(Some(out.into_column()))
            },
            GetOutput::map_dtype(move |dt| match dt {
                DataType::Struct(fds) => {
                    let fields = fds
                        .iter()
                        .map(|fd| Field::new(f(fd.name()), fd.dtype().clone()))
                        .collect();
                    Ok(DataType::Struct(fields))
                },
                _ => panic!("Only struct dtype is supported for `map_fields`."),
            }),
        )
    }

    #[cfg(feature = "dtype-struct")]
    pub fn prefix_fields(self, prefix: &str) -> Expr {
        self.0
            .map_unary(FunctionExpr::StructExpr(StructFunction::PrefixFields(
                PlSmallStr::from_str(prefix),
            )))
    }

    #[cfg(feature = "dtype-struct")]
    pub fn suffix_fields(self, suffix: &str) -> Expr {
        self.0
            .map_unary(FunctionExpr::StructExpr(StructFunction::SuffixFields(
                PlSmallStr::from_str(suffix),
            )))
    }
}

#[cfg(feature = "dtype-struct")]
pub type FieldsNameMapper = Arc<dyn Fn(&str) -> PlSmallStr + Send + Sync>;
