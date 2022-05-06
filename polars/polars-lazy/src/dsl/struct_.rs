use super::*;

/// Specialized expressions for Struct dtypes.
pub struct StructNameSpace(pub(crate) Expr);

impl StructNameSpace {
    /// Retrieve one of the fields of this [`StructChunked`] as a new Series.
    pub fn field_by_name(self, name: &str) -> Expr {
        let name1 = name.to_string();
        let name2 = name.to_string();

        self.0
            .map(
                move |s| {
                    let ca = s.struct_()?;
                    ca.field_by_name(&name1)
                },
                GetOutput::map_dtype(move |dtype| {
                    if let DataType::Struct(flds) = dtype {
                        let fld = flds
                            .iter()
                            .find(|fld| fld.name() == &name2)
                            .unwrap_or_else(|| panic!("{} not found", name2));
                        fld.data_type().clone()
                    } else {
                        unreachable!()
                    }
                }),
            )
            .with_fmt("struct.field_by_name")
            .alias(name)
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
                    StructChunked::new(ca.name(), &fields).map(|ca| ca.into_series())
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
