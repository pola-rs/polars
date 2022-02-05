use super::*;

/// Specialized expressions for [`Series`] with dates/datetimes.
pub struct DateLikeNameSpace(pub(crate) Expr);

impl DateLikeNameSpace {
    pub fn strftime(self, fmt: &str) -> Expr {
        let fmt = fmt.to_string();
        let function = move |s: Series| s.strftime(&fmt);
        self.0
            .map(function, GetOutput::from_type(DataType::Utf8))
            .with_fmt("strftime")
    }
}
