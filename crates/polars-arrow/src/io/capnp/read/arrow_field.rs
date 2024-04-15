use crate::reader::get_schema;
use capnp::dynamic_value;
use capnp::introspect::{RawEnumSchema, TypeVariant};
use capnp::schema::{Field as CapnpField, StructSchema};
use capnp::Error as CapnpError;
use polars_arrow::datatypes::{ArrowDataType, Field as ArrowField, IntegerType};
use std::collections::BTreeMap;
use std::collections::HashMap;

type Result<T> = std::result::Result<T, Error>;

enum Error {
    Capnp(CapnpError),
    EmptyStruct,
    VoidType,
    RecursionLimitExceeded,
}

pub const ENUMERANTS_METADATA_KEY: &str = "enumerants";
pub const ENUMERANTS_METADATA_SEPARATOR: &str = " ";
const MAX_RECURSIVE_DEPTH: i8 = 3;

// Infer the arrow fields from the capnp messages
pub fn infer_fields(messages: &[dynamic_value::Reader]) -> ::capnp::Result<Vec<ArrowField>> {
    let schema = get_schema(messages);
    capnp_schema_to_arrow_fields(schema)
}

fn capnp_schema_to_arrow_fields(capnp_schema: StructSchema) -> ::capnp::Result<Vec<ArrowField>> {
    let mut recursion_depth: HashMap<String, i8> = HashMap::new();
    make_arrow_fields(capnp_schema, &mut recursion_depth)
}

fn make_arrow_fields(
    schema: StructSchema,
    recursion_depth: &mut HashMap<String, i8>,
) -> ::capnp::Result<Vec<ArrowField>> {
    let mut arrow_fields = Vec::<ArrowField>::new();
    let capnp_fields = schema.get_fields()?;
    for field in capnp_fields {
        if field.get_type().is_pointer_type() {
            // TODO: Determine how to handle
        }
        match capnp_field_to_arrow_field(field, recursion_depth) {
            Ok(field) => arrow_fields.push(field),
            Err(e) => match e {
                Error::EmptyStruct => eprintln!(
                    "Cannot convert empty struct '{}' from capnp to arrow. Arrow requires at least one field.", 
                    field_name(field).unwrap()
                ),
                Error::VoidType => eprintln!(
                    "Serialized columnar data (e.g. delta lake, parquet readers) do not support Null type '{}'.", 
                    field_name(field).unwrap()
                ),
                Error::RecursionLimitExceeded => eprintln!(
                    "Exceeded recursion limit for struct '{}' from capnp to arrow after {} previously created fields. Arrow cannot use infinitely recursive schema.", 
                    field_name(field).unwrap(),
                    MAX_RECURSIVE_DEPTH,
                ),
                Error::Capnp(capnp_error) => return Err(capnp_error),
            },
        }
    }
    Ok(arrow_fields)
}

fn capnp_field_to_arrow_field(
    field: CapnpField,
    recursion_depth: &mut HashMap<String, i8>,
) -> Result<ArrowField> {
    let name = match field_name(field) {
        Ok(name) => name,
        Err(e) => return Err(Error::Capnp(e)),
    };
    // Union fields must be nullable
    let is_nullable = true; // TODO: Determine nullable or remove comment
    let capnp_dtype = field.get_type().which();
    let arrow_dtype = match limit_recursion(&name, capnp_dtype, recursion_depth) {
        false => capnp_dtype_to_arrow_dtype(capnp_dtype, recursion_depth)?,
        true => match capnp_dtype {
            TypeVariant::Struct(_) => return Err(Error::RecursionLimitExceeded),
            TypeVariant::List(_) => return Err(Error::RecursionLimitExceeded),
            _ => panic!("Expected recursion limiting to only affect struct and list"),
        },
    };
    if let TypeVariant::Enum(e) = capnp_dtype {
        let metadata = get_enum_metadata(&e);
        return Ok(ArrowField {
            name,
            data_type: arrow_dtype,
            is_nullable,
            metadata,
        });
    }
    Ok(ArrowField::new(name, arrow_dtype, is_nullable))
}

fn capnp_dtype_to_arrow_dtype(
    capnp_dtype: TypeVariant,
    recursion_depth: &mut HashMap<String, i8>,
) -> Result<ArrowDataType> {
    let arrow_dtype = match capnp_dtype {
        TypeVariant::Void => return Err(Error::VoidType),
        TypeVariant::Bool => ArrowDataType::Boolean,
        TypeVariant::Int8 => ArrowDataType::Int8,
        TypeVariant::Int16 => ArrowDataType::Int16,
        TypeVariant::Int32 => ArrowDataType::Int32,
        TypeVariant::Int64 => ArrowDataType::Int64,
        TypeVariant::UInt8 => ArrowDataType::UInt8,
        TypeVariant::UInt16 => ArrowDataType::UInt16,
        TypeVariant::UInt32 => ArrowDataType::UInt32,
        TypeVariant::UInt64 => ArrowDataType::UInt64,
        TypeVariant::Float32 => ArrowDataType::Float32,
        TypeVariant::Float64 => ArrowDataType::Float64,
        TypeVariant::Text => ArrowDataType::Utf8,
        TypeVariant::Data => ArrowDataType::Binary,
        TypeVariant::Struct(st) => {
            let schema: StructSchema = st.into();
            let fields = match &schema.get_fields() {
                Ok(capnp_fields) => {
                    if capnp_fields.is_empty() {
                        return Err(Error::EmptyStruct);
                    } else {
                        match make_arrow_fields(schema, recursion_depth) {
                            Ok(fields) => fields,
                            Err(e) => return Err(Error::Capnp(e)),
                        }
                    }
                }
                Err(e) => {
                    panic!("{}", e)
                }
            };
            if fields.is_empty() {
                return Err(Error::EmptyStruct);
            }
            ArrowDataType::Struct(fields)
        }
        TypeVariant::List(l) => {
            let inner_capnp_dtype = l.which();
            let inner_dtype = capnp_dtype_to_arrow_dtype(inner_capnp_dtype, recursion_depth)?;
            let inner_field = match inner_capnp_dtype {
                TypeVariant::Enum(e) => {
                    let metadata = get_enum_metadata(&e);
                    ArrowField {
                        name: String::from("item"),
                        data_type: inner_dtype,
                        is_nullable: true,
                        metadata,
                    }
                }
                _ => ArrowField::new("item", inner_dtype, true), // TODO: Determine nullable
            };
            ArrowDataType::List(Box::new(inner_field))
        }
        TypeVariant::Enum(_e) => {
            ArrowDataType::Dictionary(IntegerType::UInt16, Box::new(ArrowDataType::Utf8), false)
        }
        TypeVariant::AnyPointer => panic!("unsupported"),
        TypeVariant::Capability => panic!("unsupported"),
    };
    Ok(arrow_dtype)
}

// For recusive structs we need to break recursion of the schema at a reasonable level
// We will ignore any nested lists or structs beyond this limit
fn limit_recursion(
    name: &String,
    capnp_dtype: TypeVariant,
    recursion_depth: &mut HashMap<String, i8>,
) -> bool {
    match capnp_dtype {
        TypeVariant::Struct(_) => {
            update_recusion_depth(name, recursion_depth);
            if *recursion_depth.get(name).unwrap() > MAX_RECURSIVE_DEPTH {
                return true;
            }
        }
        TypeVariant::List(l) => {
            if let TypeVariant::Struct(_) = l.which() {
                update_recusion_depth(name, recursion_depth);
                if *recursion_depth.get(name).unwrap() > MAX_RECURSIVE_DEPTH {
                    return true;
                }
            }
        }
        _ => (),
    }
    false
}

fn update_recusion_depth(name: &String, recursion_depth: &mut HashMap<String, i8>) {
    if let Some(count) = recursion_depth.get(name) {
        recursion_depth.insert(name.to_string(), count + 1);
    } else {
        recursion_depth.insert(name.to_string(), 1);
    }
}

fn get_enum_metadata(enum_schema: &RawEnumSchema) -> BTreeMap<String, String> {
    let mut metadata: BTreeMap<String, String> = BTreeMap::new();
    let mut enumerants = Vec::<String>::new();
    capnp::schema::EnumSchema::from(*enum_schema)
        .get_enumerants()
        .unwrap()
        .iter()
        .for_each(|e| enumerants.push(e.get_proto().get_name().unwrap().to_string().unwrap()));
    metadata.insert(
        String::from(ENUMERANTS_METADATA_KEY),
        enumerants.join(ENUMERANTS_METADATA_SEPARATOR),
    );
    metadata
}

fn field_name(field: CapnpField) -> ::capnp::Result<String> {
    Ok(field.get_proto().get_name()?.to_string()?)
}
