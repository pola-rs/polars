use super::*;
use crate::dsl::function_expr::StructFunction;

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

    /// Retrieve one of the fields of this [`StructChunked`] as a new Series.
    pub fn field_by_name(self, name: &str) -> Expr {
        self.0
            .map_private(FunctionExpr::StructExpr(StructFunction::FieldByName(
                Arc::from(name),
            )))
            .with_function_options(|mut options| {
                options.allow_rename = true;
                options
            })
    }

    /// Rename the fields of the [`StructChunked`].
    pub fn rename_fields(self, names: Vec<String>) -> Expr {
        let names = Arc::new(names);
        let names2 = names.clone();
        self.0
            .map(
                move |s| {
                    let ca = s.struct_()?;
                    let fields = ca
                        .fields()
                        .iter()
                        .zip(names.as_ref())
                        .map(|(s, name)| {
                            let mut s = s.clone();
                            s.rename(name);
                            s
                        })
                        .collect::<Vec<_>>();
                    StructChunked::new(ca.name(), &fields).map(|ca| Some(ca.into_series()))
                },
                GetOutput::map_dtype(move |dt| match dt {
                    DataType::Struct(fields) => {
                        let fields = fields
                            .iter()
                            .zip(names2.as_ref())
                            .map(|(fld, name)| Field::new(name, fld.data_type().clone()))
                            .collect();
                        DataType::Struct(fields)
                    }
                    // The types will be incorrect, but its better than nothing
                    // we can get an incorrect type with python lambdas, because we only know return type when running
                    // the query
                    dt => DataType::Struct(
                        names2
                            .iter()
                            .map(|name| Field::new(name, dt.clone()))
                            .collect(),
                    ),
                }),
            )
            .with_fmt("struct.rename_fields")
    }
}
