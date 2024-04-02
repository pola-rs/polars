use capnp::dynamic_value;
use crate::io::capnp::read::arrow_field::infer_fields;
use crate::io::capnp::read::deserialize::deserialize;
use crate::io::capnp::read::reader::capnp_messages_from_data;
use polars::datatypes;
use polars::prelude::*;
use std::fs;
use std::path::Path;

include! {"../mod.rs"}

fn get_test_df() -> DataFrame {
    let data_path = Path::new("tests/test_data/all_types.bin");
    let data = fs::read(data_path).unwrap();

    let readers = capnp_messages_from_data(data);
    let messages: Vec<dynamic_value::Reader> = readers
        .iter()
        .map(|r| {
            r.get_root::<test_all_types_capnp::test_all_types::Reader>()
                .unwrap()
                .into()
        })
        .collect();
    let fields = infer_fields(messages.as_slice()).unwrap();
    let chunk = deserialize(messages.as_slice(), fields.as_slice()).unwrap();
    let df = DataFrame::try_from((chunk, fields.as_slice())).unwrap();
    df
}

#[test]
fn test_primitives() {
    let df = get_test_df();
    let actual_primitives = df
        .select([
            "boolField",
            "int8Field",
            "int16Field",
            "int32Field",
            "int64Field",
            "uInt8Field",
            "uInt16Field",
            "uInt32Field",
            "uInt64Field",
            "float32Field",
            "float64Field",
            "textField",
        ])
        .unwrap();
    let expected_primitives = df!(
        "boolField"=> &[true],
        "int8Field"=> {
            let field: [i8; 1] = [-123];
            field
        },
        "int16Field"=> {
            let field: [i32; 1] = [-12345];
            field
        },
        "int32Field"=> {
            let field: [i32; 1] = [-12345678];
            field
        },
        "int64Field"=> {
            let field: [i64; 1] = [-123456789012345];
            field
        },
        "uInt8Field"=> {
            let field: [u8; 1] = [234];
            field
        },
        "uInt16Field"=> {
            let field: [u16; 1] = [45678];
            field
        },
        "uInt32Field"=> {
            let field: [u32; 1] = [3456789012];
            field
        },
        "uInt64Field"=> {
            let field: [u64; 1] = [12345678901234567890];
            field
        },
        "float32Field"=> {
            let field: [f32; 1] = [1234.5];
            field
        },
        "float64Field"=> {
            let field: [f64; 1] = [-1.23e47];
            field
        },
        "textField"=> &["foo"],
    )
    .unwrap();
    assert_eq!(actual_primitives, expected_primitives);
}

#[test]
fn test_nested_struct() {
    let df = get_test_df();
    let actual_nested = df
        .select(["structField"])
        .unwrap()
        .unnest(["structField"])
        .unwrap()
        .select(["structField"])
        .unwrap()
        .unnest(["structField"])
        .unwrap()
        .select(["structField"])
        .unwrap()
        .unnest(["structField"])
        .unwrap()
        .select(["textField"])
        .unwrap();
    let expected_nested = df!(
        "textField" => &["really nested"]
    )
    .unwrap();
    assert_eq!(actual_nested, expected_nested);
}

#[test]
fn test_nested_list() {
    let df = get_test_df();
    let actual_nested_list = df
        .select(["structField"])
        .unwrap()
        .unnest(["structField"])
        .unwrap()
        .select(["textList"])
        .unwrap()
        .explode(["textList"])
        .unwrap();
    let expected_nested_list = df!(
        "textList" => &["quux", "corge", "grault"]
    )
    .unwrap();
    assert_eq!(actual_nested_list, expected_nested_list);
}

#[test]
fn test_list() {
    let df = get_test_df();
    let actual_list = df
        .select(["int64List"])
        .unwrap()
        .explode(["int64List"])
        .unwrap();
    let expected_list = df!(
        "int64List" => {
            let vals: [i64; 2] = [1111111111111111111, -1111111111111111111];
            vals
        }
    )
    .unwrap();
    assert_eq!(actual_list, expected_list);
}

#[test]
fn test_enum() {
    let df = get_test_df();
    let actual_enum = df.select(["enumField"]).unwrap();
    let actual_field = actual_enum.schema().get_field("enumField").unwrap();
    let actual_dtype = actual_field.data_type();
    let expected_categories = &[
        "foo", "bar", "baz", "qux", "quux", "corge", "grault", "garply",
    ];

    // Assert the type and categories are equal
    match actual_dtype {
        datatypes::DataType::Categorical(Some(mapping), _) => {
            let categories = mapping.get_categories();
            let actual_categories: Vec<_> =
                categories.values_iter().map(|category| category).collect();
            assert_eq!(actual_categories, expected_categories);
        }
        _ => panic!("Expected datatype for this field to be Categorical"),
    }

    // Comparing the dataframes gives an unexpected not equal error
    // Instead we assert that the values are equal
    assert_eq!(actual_enum.height(), 1);
    let actual_val = actual_enum.get(0).unwrap();
    let actual_val = actual_val[0].get_str().unwrap();
    assert_eq!(actual_val, "corge");
}

#[test]
fn test_empty_struct() {
    let data_path = Path::new("tests/test_data/empty_struct.bin");
    let data = fs::read(data_path).unwrap();

    let readers = capnp_messages_from_data(data);
    let messages: Vec<dynamic_value::Reader> = readers
        .iter()
        .map(|r| {
            r.get_root::<test_all_types_capnp::test_empty_struct::Reader>()
                .unwrap()
                .into()
        })
        .collect();
    let fields = infer_fields(messages.as_slice()).unwrap();
    assert_eq!(fields.len(), 0);
}
