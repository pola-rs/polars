use futures::io::Cursor;
use polars_arrow::array::*;
use polars_arrow::error::Result;
use polars_arrow::io::csv::read_async::*;

#[tokio::test]
async fn read() -> Result<()> {
    let data = r#"city,lat,lng
"Elgin, Scotland, the UK",57.653484,-3.335724
"Stoke-on-Trent, Staffordshire, the UK",53.002666,-2.179404
"Solihull, Birmingham, UK",52.412811,-1.778197
"Cardiff, Cardiff county, UK",51.481583,-3.179090
"Eastbourne, East Sussex, UK",50.768036,0.290472
"Oxford, Oxfordshire, UK",51.752022,-1.257677
"London, UK",51.509865,-0.118092
"Swindon, Swindon, UK",51.568535,-1.772232
"Gravesend, Kent, UK",51.441883,0.370759
"Northampton, Northamptonshire, UK",52.240479,-0.902656
"Rugby, Warwickshire, UK",52.370876,-1.265032
"Sutton Coldfield, West Midlands, UK",52.570385,-1.824042
"Harlow, Essex, UK",51.772938,0.102310
"Aberdeen, Aberdeen City, UK",57.149651,-2.099075"#;
    let mut reader = AsyncReaderBuilder::new().create_reader(Cursor::new(data.as_bytes()));

    let (fields, _) = infer_schema(&mut reader, None, true, &infer).await?;

    let mut rows = vec![ByteRecord::default(); 100];
    let rows_read = read_rows(&mut reader, 0, &mut rows).await?;

    let columns = deserialize_batch(&rows[..rows_read], &fields, None, 0, deserialize_column)?;

    assert_eq!(14, columns.len());
    assert_eq!(3, columns.arrays().len());

    let lat = columns.arrays()[1]
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert!((57.653484 - lat.value(0)).abs() < f64::EPSILON);

    let city = columns.arrays()[0]
        .as_any()
        .downcast_ref::<Utf8Array<i32>>()
        .unwrap();

    assert_eq!("Elgin, Scotland, the UK", city.value(0));
    assert_eq!("Aberdeen, Aberdeen City, UK", city.value(13));
    Ok(())
}
