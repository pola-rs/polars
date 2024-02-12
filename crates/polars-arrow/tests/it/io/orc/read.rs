use polars_arrow::array::*;
use polars_arrow::error::Error;
use polars_arrow::io::orc::{format, read};

#[test]
fn infer() -> Result<(), Error> {
    let mut reader = std::fs::File::open("fixtures/pyorc/test.orc").unwrap();
    let metadata = format::read::read_metadata(&mut reader)?;
    let schema = read::infer_schema(&metadata.footer)?;

    assert_eq!(schema.fields.len(), 12);
    Ok(())
}

fn deserialize_column(column_name: &str) -> Result<Box<dyn Array>, Error> {
    let mut reader = std::fs::File::open("fixtures/pyorc/test.orc").unwrap();
    let metadata = format::read::read_metadata(&mut reader)?;
    let schema = read::infer_schema(&metadata.footer)?;

    let footer = format::read::read_stripe_footer(&mut reader, &metadata, 0, &mut vec![])?;

    let (pos, field) = schema
        .fields
        .iter()
        .enumerate()
        .find(|f| f.1.name == column_name)
        .unwrap();

    let data_type = field.data_type.clone();
    let column = format::read::read_stripe_column(
        &mut reader,
        &metadata,
        0,
        footer,
        1 + pos as u32,
        vec![],
    )?;

    read::deserialize(data_type, &column)
}

#[test]
fn float32() -> Result<(), Error> {
    assert_eq!(
        deserialize_column("float_nullable")?,
        Float32Array::from([Some(1.0), Some(2.0), None, Some(4.0), Some(5.0)]).boxed()
    );

    assert_eq!(
        deserialize_column("float_required")?,
        Float32Array::from([Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)]).boxed()
    );
    Ok(())
}

#[test]
fn float64() -> Result<(), Error> {
    assert_eq!(
        deserialize_column("double_nullable")?,
        Float64Array::from([Some(1.0), Some(2.0), None, Some(4.0), Some(5.0)]).boxed()
    );

    assert_eq!(
        deserialize_column("double_required")?,
        Float64Array::from([Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)]).boxed()
    );
    Ok(())
}

#[test]
fn boolean() -> Result<(), Error> {
    assert_eq!(
        deserialize_column("bool_nullable")?,
        BooleanArray::from([Some(true), Some(false), None, Some(true), Some(false)]).boxed()
    );

    assert_eq!(
        deserialize_column("bool_required")?,
        BooleanArray::from([Some(true), Some(false), Some(true), Some(true), Some(false)]).boxed()
    );
    Ok(())
}

#[test]
fn int() -> Result<(), Error> {
    assert_eq!(
        deserialize_column("int_required")?,
        Int32Array::from([Some(5), Some(-5), Some(1), Some(5), Some(5)]).boxed()
    );

    assert_eq!(
        deserialize_column("int_nullable")?,
        Int32Array::from([Some(5), Some(-5), None, Some(5), Some(5)]).boxed()
    );
    Ok(())
}

#[test]
fn bigint() -> Result<(), Error> {
    assert_eq!(
        deserialize_column("bigint_required")?,
        Int64Array::from([Some(5), Some(-5), Some(1), Some(5), Some(5)]).boxed()
    );

    assert_eq!(
        deserialize_column("bigint_nullable")?,
        Int64Array::from([Some(5), Some(-5), None, Some(5), Some(5)]).boxed()
    );
    Ok(())
}

#[test]
fn utf8() -> Result<(), Error> {
    assert_eq!(
        deserialize_column("utf8_required")?,
        Utf8Array::<i32>::from_slice(["a", "bb", "ccc", "dddd", "eeeee"]).boxed()
    );

    assert_eq!(
        deserialize_column("utf8_nullable")?,
        Utf8Array::<i32>::from([Some("a"), Some("bb"), None, Some("dddd"), Some("eeeee")]).boxed()
    );
    Ok(())
}
