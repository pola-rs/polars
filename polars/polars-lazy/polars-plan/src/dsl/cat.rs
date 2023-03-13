use super::*;

/// Specialized expressions for Categorical dtypes.
pub struct CategoricalNameSpace(pub(crate) Expr);

#[derive(Copy, Clone, Debug)]
pub enum CategoricalOrdering {
    /// Use the physical categories for sorting
    Physical,
    /// Use the string value for sorting
    Lexical,
}

impl CategoricalNameSpace {
    pub fn set_ordering(self, ordering: CategoricalOrdering) -> Expr {
        self.0
            .map(
                move |s| {
                    let mut ca = s.categorical()?.clone();
                    let set_lexical = match ordering {
                        CategoricalOrdering::Lexical => true,
                        CategoricalOrdering::Physical => false,
                    };
                    ca.set_lexical_sorted(set_lexical);
                    Ok(Some(ca.into_series()))
                },
                GetOutput::from_type(DataType::Categorical(None)),
            )
            .with_fmt("set_ordering")
    }
}
