use super::*;
use crate::logical_plan::conversion::is_regex_projection;

/// Specialized expressions for Struct dtypes.
pub struct StructNameSpace(pub(crate) Expr);

impl StructNameSpace {
    pub fn field_by_index(self, index: i64) -> Expr {
        self.0
            .map_private(FunctionExpr::StructExpr(StructFunction::FieldByIndex(
                index,
            )))
            .with_function_options(|mut options| {
                options.allow_rename = true;
                options
            })
    }

    /// Retrieve one or multiple of the fields of this [`StructChunked`] as a new Series.
    /// This expression also expands the `"*"` wildcard column.
    pub fn field_by_names<S: AsRef<str>>(self, names: &[S]) -> Expr {
        self.field_by_names_impl(
            names
                .iter()
                .map(|name| ColumnName::from(name.as_ref()))
                .collect(),
        )
    }

    fn field_by_names_impl(self, names: Arc<[ColumnName]>) -> Expr {
        self.0
            .map_private(FunctionExpr::StructExpr(StructFunction::MultipleFields(
                names,
            )))
            .with_function_options(|mut options| {
                options.allow_rename = true;
                options
            })
    }

    /// Retrieve one of the fields of this [`StructChunked`] as a new Series.
    /// This expression also supports wildcard "*" and regex expansion.
    pub fn field_by_name(self, name: &str) -> Expr {
        if name == "*" || is_regex_projection(name) {
            return self.field_by_names(&[name]);
        }
        self.0
            .map_private(FunctionExpr::StructExpr(StructFunction::FieldByName(
                ColumnName::from(name),
            )))
            .with_function_options(|mut options| {
                options.allow_rename = true;
                options
            })
    }

    /// Rename the fields of the [`StructChunked`].
    pub fn rename_fields(self, names: Vec<String>) -> Expr {
        self.0
            .map_private(FunctionExpr::StructExpr(StructFunction::RenameFields(
                Arc::from(names),
            )))
    }

    #[cfg(feature = "json")]
    pub fn json_encode(self) -> Expr {
        self.0
            .map_private(FunctionExpr::StructExpr(StructFunction::JsonEncode))
    }

    pub fn with_fields(self, fields: Vec<Expr>) -> Expr {
        fn materialize_field(this: &Expr, field: Expr) -> Expr {
            field.map_expr(|e| match e {
                Expr::Field(names) => {
                    let this = this.clone().struct_();
                    if names.len() == 1 {
                        this.field_by_name(names[0].as_ref())
                    } else {
                        this.field_by_names_impl(names)
                    }
                },
                _ => e,
            })
        }

        let mut new_fields = Vec::with_capacity(fields.len());
        new_fields.push(Default::default());

        new_fields.extend(fields.into_iter().map(|e| materialize_field(&self.0, e)));
        new_fields[0] = self.0;
        Expr::Function {
            input: new_fields,
            function: FunctionExpr::StructExpr(StructFunction::WithFields),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ElementWise,
                pass_name_to_apply: true,
                allow_group_aware: false,
                ..Default::default()
            },
        }
    }
}
