use super::*;

/// Specialized expressions for [`DataType::Union`].
pub struct UnionNameSpace(pub(crate) Expr);

impl UnionNameSpace {
    /// The variant of every row, as an `Enum` of the variant names.
    pub fn tag(self) -> Expr {
        self.0.map_unary(UnionFunction::Tag)
    }

    /// The values of the variant `name`, null in rows of other variants.
    pub fn field(self, name: PlSmallStr) -> Expr {
        self.0.map_unary(UnionFunction::Field(name))
    }

    /// A union of `dtype` in which every row holds this expression as the variant `variant`.
    pub fn value(self, dtype: DataTypeExpr, variant: PlSmallStr) -> Expr {
        self.0.map_unary(UnionFunction::Value { dtype, variant })
    }

    /// Evaluates the arm of each row's variant, where `pl.element()` is the variant's value.
    ///
    /// Variants without an arm give `default`, which is required unless every variant has one.
    pub fn match_(self, arms: Vec<(PlSmallStr, Expr)>, default: Option<Expr>) -> Expr {
        let (names, arms): (Vec<_>, Vec<_>) = arms.into_iter().unzip();
        let has_default = default.is_some();
        let input = std::iter::once(self.0).chain(arms).chain(default).collect();
        Expr::Function {
            input,
            function: UnionFunction::Match { names, has_default }.into(),
        }
    }
}
