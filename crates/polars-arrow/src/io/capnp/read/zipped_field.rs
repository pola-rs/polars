use capnp::introspect::TypeVariant;
use capnp::schema::{Field as CapnpField, StructSchema};
use polars_arrow::datatypes::{ArrowDataType, Field as ArrowField};

// Zip the capnp field and corresponding arrow fields into a single ZippedField struct.
// This helps significantly improve performance (getting capnp fields is slow) as well as
// making it easier to reference field metadata in recursive deserialization
#[derive(Clone)]
pub struct ZippedField {
    arrow_field: ArrowField,
    capnp_field: Option<CapnpField>,
    nested_inner_fields: Option<Vec<ZippedField>>,
}

impl ZippedField {
    pub fn arrow_field(&self) -> &ArrowField {
        &self.arrow_field
    }

    pub fn capnp_field(&self) -> &CapnpField {
        match &self.capnp_field {
            Some(f) => f,
            None => panic!("Expected field '{}' to have a capnp field. Only list items should not have a capnp field.", self.arrow_field.name)
        }
    }

    pub fn inner_fields(&self) -> &Vec<ZippedField> {
        match &self.nested_inner_fields {
            Some(f) => f,
            None => panic!(
                "Expected field '{}' to have inner fields. Structs have nested inner fields.",
                self.arrow_field.name
            ),
        }
    }

    pub fn inner_field(&self) -> &ZippedField {
        match &self.nested_inner_fields {
            Some(f) => {
                match f.len() {
                    1 => &f[0],
                    _ => panic!("Expected field '{}' to have a single inner field. Lists have a single nested inner field.", self.arrow_field.name)
                }
            },
            None => panic!("Expected field '{}' to have inner fields. Nested types (struct and list) have inner fields.", self.arrow_field.name)
        }
    }
}

pub fn zip_fields(
    schema: StructSchema,
    arrow_fields: &[ArrowField],
) -> ::capnp::Result<Vec<ZippedField>> {
    let fields = arrow_fields
        .iter()
        .map(|arrow_field| {
            let capnp_field = schema.get_field_by_name(&arrow_field.name).unwrap();
            match arrow_field.data_type() {
                ArrowDataType::Struct(inner_arrow_fields) => {
                    let mut inner_fields = Vec::<ZippedField>::new();
                    if let TypeVariant::Struct(st) = capnp_field.get_type().which() {
                        let inner_schema: StructSchema = st.into();
                        inner_fields.extend(zip_fields(inner_schema, inner_arrow_fields).unwrap());
                    }
                    ZippedField {
                        arrow_field: ArrowField::new(
                            arrow_field.name.to_string(),
                            arrow_field.data_type().clone(),
                            true,
                        ),
                        capnp_field: Some(capnp_field),
                        nested_inner_fields: Some(inner_fields),
                    }
                }
                ArrowDataType::List(inner) => {
                    let mut inner_fields = Vec::<ZippedField>::new();
                    if let TypeVariant::List(l) = capnp_field.get_type().which() {
                        inner_fields.push(
                            zip_list_field(
                                l.which(),
                                ArrowField::new(
                                    inner.name.to_string(),
                                    inner.data_type().clone(),
                                    true,
                                ),
                            )
                            .unwrap(),
                        );
                    }
                    ZippedField {
                        arrow_field: ArrowField::new(
                            arrow_field.name.to_string(),
                            arrow_field.data_type().clone(),
                            true,
                        ),
                        capnp_field: Some(capnp_field),
                        nested_inner_fields: Some(inner_fields),
                    }
                }
                _ => ZippedField {
                    arrow_field: ArrowField::new(
                        arrow_field.name.to_string(),
                        arrow_field.data_type().clone(),
                        true,
                    ),
                    capnp_field: Some(capnp_field),
                    nested_inner_fields: None,
                },
            }
        })
        .collect();
    Ok(fields)
}

fn zip_list_field(
    capnp_dtype: TypeVariant,
    arrow_field: ArrowField,
) -> ::capnp::Result<ZippedField> {
    match arrow_field.data_type() {
        ArrowDataType::Struct(inner_arrow_fields) => match capnp_dtype {
            TypeVariant::Struct(st) => {
                let schema: StructSchema = st.into();
                let inner_fields = zip_fields(schema, inner_arrow_fields)?;
                Ok(ZippedField {
                    arrow_field: ArrowField::new(
                        arrow_field.name.to_string(),
                        arrow_field.data_type().clone(),
                        true,
                    ),
                    capnp_field: None,
                    nested_inner_fields: Some(inner_fields),
                })
            }
            _ => panic!(
                "Expected arrow struct type to match capnp field type for {}",
                arrow_field.name
            ),
        },
        ArrowDataType::List(inner) => match capnp_dtype {
            TypeVariant::List(l) => Ok(zip_list_field(
                l.which(),
                ArrowField::new(inner.name.to_string(), inner.data_type().clone(), true),
            )?),
            _ => panic!(
                "Expected arrow list type to match capnp field type for {}",
                arrow_field.name
            ),
        },
        _ => Ok(ZippedField {
            arrow_field,
            capnp_field: None,
            nested_inner_fields: None,
        }),
    }
}
