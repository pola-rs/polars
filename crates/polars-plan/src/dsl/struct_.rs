use super::*;
use crate::plans::conversion::is_regex_projection;

/// Specialized expressions for Struct dtypes.
pub struct StructNameSpace(pub(crate) Expr);

impl StructNameSpace {
    pub fn field_by_index(self, index: i64) -> Expr {
        self.0
            .map_unary(FunctionExpr::StructExpr(StructFunction::FieldByIndex(
                index,
            )))
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
        self.0
            .map_unary(FunctionExpr::StructExpr(StructFunction::MultipleFields(
                names,
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

    pub fn with_fields(self, fields: Vec<Expr>) -> PolarsResult<Expr> {
        fn materialize_field(this: &Expr, field: Expr) -> PolarsResult<Expr> {
            field.try_map_expr(|e| match e {
                Expr::Field(names) => {
                    let this = this.clone().struct_();
                    Ok(if names.len() == 1 {
                        this.field_by_name(names[0].as_ref())
                    } else {
                        this.field_by_names_impl(names)
                    })
                },
                Expr::Exclude(_, _) => {
                    polars_bail!(InvalidOperation: "'exclude' not allowed in 'field'")
                },
                _ => Ok(e),
            })
        }

        let s = self.0.clone();
        self.0.try_map_n_ary(
            FunctionExpr::StructExpr(StructFunction::WithFields),
            fields.into_iter().map(|e| materialize_field(&s, e)),
        )
    }
}
