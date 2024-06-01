use arrow::array::*;
use arrow::datatypes::{ArrowDataType, Field};

fn dt() -> ArrowDataType {
    ArrowDataType::Struct(vec![
        Field::new("a", ArrowDataType::Utf8, true),
        Field::new("b", ArrowDataType::Utf8, true),
    ])
}

fn array() -> MapArray {
    let data_type = ArrowDataType::Map(Box::new(Field::new("a", dt(), true)), false);

    let field = StructArray::new(
        dt(),
        vec![
            Box::new(Utf8Array::<i32>::from_slice(["a", "aa", "aaa"])) as _,
            Box::new(Utf8Array::<i32>::from_slice(["b", "bb", "bbb"])),
        ],
        None,
    );

    MapArray::new(
        data_type,
        vec![0, 1, 2, 3].try_into().unwrap(),
        Box::new(field),
        None,
    )
}

#[test]
fn basics() {
    let array = array();

    assert_eq!(
        array.value(0),
        Box::new(StructArray::new(
            dt(),
            vec![
                Box::new(Utf8Array::<i32>::from_slice(["a"])) as _,
                Box::new(Utf8Array::<i32>::from_slice(["b"])),
            ],
            None,
        )) as Box<dyn Array>
    );

    let sliced = array.sliced(1, 1);
    assert_eq!(
        sliced.value(0),
        Box::new(StructArray::new(
            dt(),
            vec![
                Box::new(Utf8Array::<i32>::from_slice(["aa"])) as _,
                Box::new(Utf8Array::<i32>::from_slice(["bb"])),
            ],
            None,
        )) as Box<dyn Array>
    );
}

#[test]
fn split_at() {
    let (lhs, rhs) = array().split_at(1);

    assert_eq!(
        lhs.value(0),
        Box::new(StructArray::new(
            dt(),
            vec![
                Box::new(Utf8Array::<i32>::from_slice(["a"])) as _,
                Box::new(Utf8Array::<i32>::from_slice(["b"])),
            ],
            None,
        )) as Box<dyn Array>
    );
    assert_eq!(
        rhs.value(0),
        Box::new(StructArray::new(
            dt(),
            vec![
                Box::new(Utf8Array::<i32>::from_slice(["aa"])) as _,
                Box::new(Utf8Array::<i32>::from_slice(["bb"])),
            ],
            None,
        )) as Box<dyn Array>
    );
    assert_eq!(
        rhs.value(1),
        Box::new(StructArray::new(
            dt(),
            vec![
                Box::new(Utf8Array::<i32>::from_slice(["aaa"])) as _,
                Box::new(Utf8Array::<i32>::from_slice(["bbb"])),
            ],
            None,
        )) as Box<dyn Array>
    );
}
