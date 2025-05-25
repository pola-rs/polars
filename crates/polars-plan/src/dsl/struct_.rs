use super::*;
use crate::plans::conversion::is_regex_projection;

/// Specialized expressions for Struct dtypes.
pub struct StructNameSpace(pub Expr);

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
        fn materialize_field(base: Expr, field: Expr) -> PolarsResult<Expr> {
            match &field {
                Expr::Alias(_, _) | Expr::Literal(_) | Expr::Column(_) | Expr::Field(_) => Ok(field),
                Expr::Exclude(_, _) => {
                    polars_bail!(InvalidOperation: "'exclude' not allowed in 'field'")
                },
                _ => field.try_map_expr(|e| match e {
                    Expr::Field(names) => {
                        let ns = StructNameSpace(base.clone());
                        Ok(if names.len() == 1 {
                            ns.field_by_name(names[0].as_ref())
                        } else {
                            ns.field_by_names_impl(names)
                        })
                    },
                    Expr::Exclude(_, _) => {
                        polars_bail!(InvalidOperation: "'exclude' not allowed in 'field'")
                    },
                    _ => Ok(e),
                }),
            }
        }

        fn flatten_structs(exprs: Vec<Expr>) -> Vec<Expr> {
            let mut flat = Vec::with_capacity(exprs.len());
            let mut stack = exprs;

            while let Some(e) = stack.pop() {
                match e {
                    Expr::Function {
                        function:
                        FunctionExpr::AsStruct
                        | FunctionExpr::StructExpr(StructFunction::WithFields),
                        input,
                        ..
                    } => {
                        stack.extend(input.into_iter().rev());
                    }
                    other => flat.push(other),
                }
            }
            flat
        }

        let base = self.0;
        let flattened_fields = flatten_structs(fields);
        debug_assert!(
            {
                Self::assert_no_struct_wrappers(&flattened_fields);
                true
            },
            "Struct flattening failed"
        );
        let base_clone = base.clone();

        base.try_map_n_ary(
            FunctionExpr::StructExpr(StructFunction::WithFields),
            flattened_fields.into_iter().map(move |e| materialize_field(base_clone.clone(), e)),
        )
    }

    fn assert_no_struct_wrappers(exprs: &[Expr]) {
        for e in exprs {
            match e {
                Expr::Function {
                    function:
                    FunctionExpr::AsStruct
                    | FunctionExpr::StructExpr(StructFunction::WithFields),
                    ..
                } => panic!("Unexpected nested struct expression: {:?}", e),
                Expr::Function { input, .. } => Self::assert_no_struct_wrappers(input),
                Expr::Ternary { truthy, falsy, predicate } => {
                    Self::assert_no_struct_wrappers(&[
                        truthy.as_ref().clone(),
                        falsy.as_ref().clone(),
                        predicate.as_ref().clone(),
                    ]);
                }
                _ => {}
            }
        }
    }
    
}
