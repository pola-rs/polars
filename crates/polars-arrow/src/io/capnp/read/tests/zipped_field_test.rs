use capnp::{dynamic_struct, dynamic_value};
use crate::io::capnp::read::arrow_field::infer_fields;
use crate::io::capnp::read::reader::{capnp_messages_from_data, get_schema};
use crate::io::capnp::read::zipped_field::{zip_fields, ZippedField};
use std::fs;
use std::path::Path;

include! {"../mod.rs"}

fn get_fields() -> Vec<ZippedField> {
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
    let capnp_schema = get_schema(messages.as_slice());
    zip_fields(capnp_schema, fields.as_slice()).unwrap()
}

fn get_union_fields() -> Vec<ZippedField> {
    let data_path = Path::new("tests/test_data/union.bin");
    let data = fs::read(data_path).unwrap();

    let readers = capnp_messages_from_data(data);
    let messages: Vec<dynamic_value::Reader> = readers
        .iter()
        .map(|r| {
            r.get_root::<test_all_types_capnp::test_union::Reader>()
                .unwrap()
                .into()
        })
        .collect();
    let fields = infer_fields(messages.as_slice()).unwrap();
    let capnp_schema = messages[0]
        .downcast::<dynamic_struct::Reader>()
        .get_schema();
    zip_fields(capnp_schema, fields.as_slice()).unwrap()
}

#[test]
fn test_primitives() {
    let fields = get_fields();
    let field_names = vec![
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
    ];
    for (i, field) in fields[0..field_names.len()].iter().enumerate() {
        assert_eq!(field.arrow_field().name, field_names[i]);
    }
}

#[test]
fn test_no_void_field() {
    let fields = get_fields();
    let field_name = "voidField";
    for field in fields {
        if field.arrow_field().name == field_name {
            panic!("Expect to drop voidField")
        }
    }
}

#[test]
#[should_panic]
fn test_primitives_inner_fields() {
    let fields = get_fields();
    let field_names = vec![
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
    ];
    for field in fields[0..field_names.len()].iter() {
        let _ = field.inner_fields();
    }
}

#[test]
fn test_struct_field() {
    let fields = get_fields();
    assert_eq!(&fields[13].arrow_field().name, "structField");
    let children = &fields[13].inner_fields();
    assert_eq!(children.len(), 29);
    let bool_field = &children[0];
    assert_eq!(bool_field.arrow_field().name, "boolField");
    assert_eq!(
        bool_field
            .capnp_field()
            .get_proto()
            .get_name()
            .unwrap()
            .to_string()
            .unwrap(),
        "boolField"
    );
}

#[test]
fn test_list_field() {
    let fields = get_fields();
    let children = &fields[15].inner_fields();
    assert_eq!(children.len(), 1);
    let inner_field = &fields[15].inner_field();
    assert_eq!(inner_field.arrow_field().name, "item");
    assert_eq!(
        fields[15]
            .capnp_field()
            .get_proto()
            .get_name()
            .unwrap()
            .to_string()
            .unwrap(),
        "boolList"
    );
}

#[test]
#[should_panic]
fn test_list_field_capnp_panic() {
    let fields = get_fields();
    let inner_field = &fields[16].inner_field();
    inner_field.capnp_field(); // No capnp equivalent field to list item, should panic
}

#[test]
#[should_panic]
fn test_struct_field_singular_inner_field_panic() {
    let fields = get_fields();
    let _ = &fields[14].inner_field(); // more than one inner field, should panic
}

#[test]
fn test_nested_struct_field() {
    let fields = get_fields();
    let struct0 = &fields[13].inner_fields();
    assert_eq!(&struct0[13].arrow_field().name, "structField");
    let struct1 = &struct0[13].inner_fields();
    assert_eq!(&struct1[13].arrow_field().name, "structField");
    let struct2 = &struct1[13].inner_fields();
    // We cut off `structField` to limit the depth of recursion
    assert_eq!(struct2[13].arrow_field().name, "enumField");
    assert_eq!(struct2.len(), 29);
    let primitive_child = &struct2[1];
    assert_eq!(primitive_child.arrow_field().name, "int8Field");
    assert_eq!(
        primitive_child
            .capnp_field()
            .get_proto()
            .get_name()
            .unwrap()
            .to_string()
            .unwrap(),
        "int8Field"
    );
    // The nested structs continue in the structList field
    let list_child = &struct2[27];
    assert_eq!(list_child.arrow_field().name, "structList");
    let list_struct_item0 = struct2[27].inner_field();
    let list_struct0 = list_struct_item0.inner_fields();
    assert_eq!(&list_struct0[0].arrow_field().name, "boolField");
}

#[test]
fn test_union_fields() {
    let fields: Vec<ZippedField> = get_union_fields();
    assert_eq!(fields.len(), 4);
    assert_eq!(fields[0].arrow_field().name, "union0");
    assert_eq!(fields[1].arrow_field().name, "listOuter");
    assert_eq!(fields[2].arrow_field().name, "grault");
    assert_eq!(fields[3].arrow_field().name, "garply");
}

#[test]
fn test_nested_union_list_field() {
    let fields: Vec<ZippedField> = get_union_fields();
    let list_children = &fields[1].inner_fields();
    assert_eq!(&list_children[0].arrow_field().name, "item");
    let outer_struct_children = &list_children[0].inner_fields();
    assert_eq!(&outer_struct_children[0].arrow_field().name, "union1");
    let union1_children = &outer_struct_children[0].inner_fields();
    assert_eq!(&union1_children[0].arrow_field().name, "listInner");
    let list_inner_children = &union1_children[0].inner_fields();
    assert_eq!(&list_inner_children[0].arrow_field().name, "item");
    let inner_struct_children = &list_inner_children[0].inner_fields();
    let baz = &inner_struct_children[0];
    assert_eq!(inner_struct_children.len(), 1);
    assert_eq!(baz.arrow_field().name, "baz");
    assert_eq!(
        baz.capnp_field()
            .get_proto()
            .get_name()
            .unwrap()
            .to_string()
            .unwrap(),
        "baz"
    );
}
