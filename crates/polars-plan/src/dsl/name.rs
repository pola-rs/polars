#[cfg(feature = "dtype-struct")]
use smartstring::alias::String as SmartString;

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
        Expr::KeepName(Box::new(self.0))
    }

    /// Define an alias by mapping a function over the original root column name.
    pub fn map<F>(self, function: F) -> Expr
    where
        F: Fn(&str) -> PolarsResult<String> + 'static + Send + Sync,
    {
        let function = SpecialEq::new(Arc::new(function) as Arc<dyn RenameAliasFn>);
        Expr::RenameAlias {
            expr: Box::new(self.0),
            function,
        }
    }

    /// Add a prefix to the root column name.
    pub fn prefix(self, prefix: &str) -> Expr {
        let prefix = prefix.to_string();
        self.map(move |name| Ok(format!("{prefix}{name}")))
    }

    /// Add a suffix to the root column name.
    pub fn suffix(self, suffix: &str) -> Expr {
        let suffix = suffix.to_string();
        self.map(move |name| Ok(format!("{name}{suffix}")))
    }

    /// Update the root column name to use lowercase characters.
    #[allow(clippy::wrong_self_convention)]
    pub fn to_lowercase(self) -> Expr {
        self.map(move |name| Ok(name.to_lowercase()))
    }

    /// Update the root column name to use uppercase characters.
    #[allow(clippy::wrong_self_convention)]
    pub fn to_uppercase(self) -> Expr {
        self.map(move |name| Ok(name.to_uppercase()))
    }

    #[cfg(feature = "dtype-struct")]
    pub fn map_fields(self, function: FieldsNameMapper) -> Expr {
        let f = function.clone();
        self.0.map(
            move |s| {
                let s = s.struct_()?;
                let fields = s
                    .fields()
                    .iter()
                    .map(|fd| {
                        let mut fd = fd.clone();
                        fd.rename(&function(fd.name()));
                        fd
                    })
                    .collect::<Vec<_>>();
                StructChunked::new(s.name(), &fields).map(|ca| Some(ca.into_series()))
            },
            GetOutput::map_dtype(move |dt| match dt {
                DataType::Struct(fds) => {
                    let fields = fds
                        .iter()
                        .map(|fd| Field::new(&f(fd.name()), fd.data_type().clone()))
                        .collect();
                    DataType::Struct(fields)
                },
                _ => panic!("Only struct dtype is supported for `map_fields`."),
            }),
        )
    }

    #[cfg(feature = "dtype-struct")]
    pub fn prefix_fields(self, prefix: &str) -> Expr {
        self.0
            .map_private(FunctionExpr::StructExpr(StructFunction::PrefixFields(
                Arc::from(prefix),
            )))
    }

    #[cfg(feature = "dtype-struct")]
    pub fn suffix_fields(self, suffix: &str) -> Expr {
        self.0
            .map_private(FunctionExpr::StructExpr(StructFunction::SuffixFields(
                Arc::from(suffix),
            )))
    }
}

#[cfg(feature = "dtype-struct")]
pub type FieldsNameMapper = Arc<dyn Fn(&str) -> SmartString + Send + Sync>;
