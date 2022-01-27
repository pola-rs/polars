use super::*;

pub struct StringNameSpace(pub(crate) Expr);

impl StringNameSpace {
    pub fn extract(self, pat: &str, group_index: usize) -> Expr {
        let pat = pat.to_string();
        let function = move |s: Series| {
            let ca = s.utf8()?;
            match ca.extract(&pat, group_index) {
                Ok(ca) => Ok(ca.into_series()),
                Err(e) => Err(PolarsError::ComputeError(format!("{:?}", e).into())),
            }
        };
        self.0
            .map(function, GetOutput::from_type(DataType::Utf8))
            .with_fmt("str.extract")
    }

    #[cfg(feature = "temporal")]
    pub fn strftime(self, fmt: &str) -> Expr {
        let fmt = fmt.to_string();
        let function = move |s: Series| s.strftime(&fmt);
        self.0
            .map(function, GetOutput::from_type(DataType::Utf8))
            .with_fmt("strftime")
    }

    #[cfg(feature = "concat_str")]
    /// Concat the values into a string array.
    /// # Arguments
    ///
    /// * `delimiter` - A string that will act as delimiter between values.
    pub fn concat(self, delimiter: &str) -> Expr {
        let delimiter = delimiter.to_owned();
        self.0
            .apply(
                move |s| Ok(s.str_concat(&delimiter).into_series()),
                GetOutput::from_type(DataType::Utf8),
            )
            .with_fmt("str_concat")
    }
}
