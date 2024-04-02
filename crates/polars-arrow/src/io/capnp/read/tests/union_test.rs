use capnp::dynamic_value;
use capnp2arrow::arrow_field::infer_fields;
use capnp2arrow::deserialize::deserialize;
use capnp2arrow::reader::capnp_messages_from_data;
use polars::prelude::*;
use polars_arrow::array::{Array, BooleanArray, Float32Array, Int32Array, StructArray, UnionArray};
use polars_arrow::buffer::Buffer;
use polars_arrow::chunk::Chunk;
use polars_arrow::datatypes::UnionMode;
use polars_arrow::datatypes::{ArrowDataType, Field as ArrowField};
use smartstring::alias::String as SmartString;
use std::fs;
use std::path::Path;

include! {"../src/test.rs"}

fn get_test_df() -> DataFrame {
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
    let chunk = deserialize(messages.as_slice(), fields.as_slice()).unwrap();
    let df = DataFrame::try_from((chunk, fields.as_slice())).unwrap();
    df
}

#[test]
fn test_union_values() {
    let df = get_test_df();
    assert_eq!(df.height(), 2);
    let actual_union0 = df.select(["union0"]).unwrap().unnest(["union0"]).unwrap();
    let expected_union0 =
        df!("foo" => {let val: [Option<u16>; 2] = [None, Some(9)]; val}, "bar" => &[Some(0), None])
            .unwrap();
    assert_eq!(actual_union0, expected_union0);
    let actual_grault = df.select(["grault"]).unwrap();
    let expected_grault = df!("grault" => {let val: [Option<u16>; 2] = [None, None]; val}).unwrap();
    assert_eq!(actual_grault, expected_grault);
    let actual_garply = df.select(["garply"]).unwrap();
    let expected_garply = df!("garply" => &["first", "second"]).unwrap();
    assert_eq!(actual_garply, expected_garply);
    let df_list_outer = df
        .select(["listOuter"])
        .unwrap()
        .explode(["listOuter"])
        .unwrap()
        .unnest(["listOuter"])
        .unwrap();
    let actual_primitive_list = df_list_outer
        .select(["primitiveList"])
        .unwrap()
        .explode(["primitiveList"])
        .unwrap();
    let expected_primitive_list = df!("primitiveList" => {let val: [Option<u16>; 6] = [Some(2), Some(3), None, Some(8), Some(11), Some(12)]; val}).unwrap();
    assert_eq!(actual_primitive_list, expected_primitive_list);
    let actual_corge = df_list_outer
        .select(["corge"])
        .unwrap()
        .unnest(["corge"])
        .unwrap();
    let expected_corge =
        df!("baz" => {let val: [Option<u16>; 4] = [None, Some(5), None, None]; val}).unwrap();
    assert_eq!(actual_corge, expected_corge);
    let df_union1 = df_list_outer
        .select(["union1"])
        .unwrap()
        .unnest(["union1"])
        .unwrap();
    let actual_list_inner = df_union1
        .select(["listInner"])
        .unwrap()
        .explode(["listInner"])
        .unwrap()
        .unnest(["listInner"])
        .unwrap();
    let expected_list_inner =
        df!("baz" => {let val: [Option<u16>; 5] = [Some(1), None, Some(6), Some(7), None]; val})
            .unwrap();
    assert_eq!(actual_list_inner, expected_list_inner);
    let actual_qux = df_union1.select(["qux"]).unwrap();
    let expected_qux =
        df!("qux" => {let val: [Option<u32>; 4] = [None, Some(4), None, Some(10)]; val}).unwrap();
    assert_eq!(actual_qux, expected_qux);
}

#[test]
fn test_union_schema() {
    let df = get_test_df();
    let mut schema = Schema::new();
    let union0 = DataType::Struct(vec![
        Field::new("foo", DataType::UInt16),
        Field::new("bar", DataType::UInt32),
    ]);
    let mut union0_name = SmartString::new();
    union0_name.push_str("union0");
    schema.with_column(union0_name, union0);
    let struct_in_union = DataType::Struct(vec![Field::new("baz", DataType::UInt16)]);
    let list_inner = DataType::List(Box::new(struct_in_union.clone()));
    let union1 = DataType::Struct(vec![
        Field::new("listInner", list_inner),
        Field::new("qux", DataType::UInt32),
    ]);
    let union_in_struct = DataType::Struct(vec![
        Field::new("union1", union1),
        Field::new("primitiveList", DataType::List(Box::new(DataType::UInt16))),
        Field::new("corge", struct_in_union.clone()),
    ]);
    let list_outer = DataType::List(Box::new(union_in_struct));
    let mut list_outer_name = SmartString::new();
    list_outer_name.push_str("listOuter");
    schema.with_column(list_outer_name, list_outer);
    let mut grault = SmartString::new();
    grault.push_str("grault");
    schema.with_column(grault, DataType::UInt16);
    let mut garply = SmartString::new();
    garply.push_str("garply");
    schema.with_column(garply, DataType::String);

    assert_eq!(df.schema(), schema);
}

// This test creates two arrow types for unnamed union in a struct: a struct with struct fields
// and a union with union fields.
// Currently the test produces an error because polars does not support unions in dataframes
// we panic if it produces a different error and we panic when it succeeds
// (because we have work to do).
#[test]
fn test_struct_and_union_dataframe() {
    let boolean = BooleanArray::from_slice(&[false, true]).boxed();
    let struct_fields = vec![ArrowField::new("b", ArrowDataType::Boolean, false)];
    let struct_array = StructArray::new(ArrowDataType::Struct(struct_fields), vec![boolean], None);
    let struct_type = ArrowDataType::Struct(struct_array.fields().to_vec());

    let mut union_fields: Vec<ArrowField> = Vec::new();
    union_fields.push(ArrowField::new("union0", ArrowDataType::Int32, true));
    union_fields.push(ArrowField::new("union1", ArrowDataType::Float32, true));
    let union_type = ArrowDataType::Union(union_fields, None, UnionMode::Sparse);

    let union0 = Int32Array::from(&[Some(1), None]).boxed();
    let union1 = Float32Array::from(&[None, Some(2.0)]).boxed();
    let union_values = vec![union0, union1];

    let types: Buffer<i8> = vec![0, 1].into();
    let union_array = UnionArray::new(union_type.clone(), types, union_values, None);

    let fields = vec![
        ArrowField::new("name", struct_type, true),
        ArrowField::new("name", union_type, true),
    ];
    let arrays: Vec<Box<dyn Array>> = vec![struct_array.to_boxed(), union_array.to_boxed()];
    let chunk = Chunk::new(arrays);
    let try_df = DataFrame::try_from((chunk, fields.as_slice()));
    match try_df {
        Ok(_df) => {
            panic!("Polars dataframes now support unions! We should support unions as well.")
        }
        Err(e) => {
            if !e.to_string().contains("cannot create series from Union") {
                panic!("unknown error: {}", e)
            }
        }
    }
}
