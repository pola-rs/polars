use super::*;
use crate::dsl::functions::nth;
use crate::plans::conversion::is_regex_projection;

/// Specialized expressions for Struct dtypes.
pub struct StructNameSpace(pub(crate) Expr);

impl StructNameSpace {
    pub fn field_by_index(self, index: i64) -> Expr {
        self.0
            .map_unary(FunctionExpr::StructExpr(StructFunction::SelectFields(nth(
                index,
            ))))
    }

    /// Retrieve one or multiple of the fields of this [`StructChunked`] as a new Series.
    /// This expression also expands the `"*"` wildcard column.
    pub fn field_by_names<I, S>(self, names: I) -> Expr
    where
        I: IntoIterator<Item = S>,
        S: Into<PlSmallStr>,
    {
        self.field_by_names_impl(names.into_iter().map(|x| x.into()).collect())
    }

    fn field_by_names_impl(self, names: Arc<[PlSmallStr]>) -> Expr {
        let mut selector = Selector::Empty;
        let _s = &mut selector;
        let names = names
            .iter()
            .filter(|n| {
                match n.as_str() {
                    _ if is_regex_projection(n.as_str()) => *_s |= Selector::Matches((*n).clone()),
                    "*" => *_s |= Selector::Wildcard,
                    _ => return true,
                }

                false
            })
            .cloned()
            .collect();
        selector |= Selector::ByName {
            names,
            strict: true,
        };

        self.0
            .map_unary(FunctionExpr::StructExpr(StructFunction::SelectFields(
                selector,
            )))
    }

    /// Retrieve one of the fields of this [`StructChunked`] as a new Series.
    /// This expression also supports wildcard "*" and regex expansion.
    pub fn field_by_name(self, name: &str) -> Expr {
        if name == "*" || is_regex_projection(name) {
            return self.field_by_names([name]);
        }
        self.0
            .map_unary(FunctionExpr::StructExpr(StructFunction::FieldByName(
                name.into(),
            )))
    }

    /// Rename the fields of the [`StructChunked`].
    pub fn rename_fields<I, S>(self, names: I) -> Expr
    where
        I: IntoIterator<Item = S>,
        S: Into<PlSmallStr>,
    {
        self._rename_fields_impl(names.into_iter().map(|x| x.into()).collect())
    }

    pub fn _rename_fields_impl(self, names: Arc<[PlSmallStr]>) -> Expr {
        self.0
            .map_unary(FunctionExpr::StructExpr(StructFunction::RenameFields(
                names,
            )))
    }

    #[cfg(feature = "json")]
    pub fn json_encode(self) -> Expr {
        self.0
            .map_unary(FunctionExpr::StructExpr(StructFunction::JsonEncode))
    }

    pub fn with_fields(self, fields: Vec<Expr>) -> Expr {
        Expr::StructEval {
            expr: Arc::new(self.0),
            evaluation: fields,
        }
    }
}
