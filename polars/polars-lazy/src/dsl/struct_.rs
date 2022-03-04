use super::*;

/// Specialized expressions for Struct dtypes.
pub struct StructNameSpace(pub(crate) Expr);

impl StructNameSpace {
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
}
