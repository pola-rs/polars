// --8<-- [start:setup]
use chrono::prelude::*;
use polars::prelude::*;
// --8<-- [end:setup]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:df]
    let time = polars::time::date_range(
        "time",
        NaiveDate::from_ymd_opt(2021, 12, 16)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap(),
        NaiveDate::from_ymd_opt(2021, 12, 16)
            .unwrap()
            .and_hms_opt(3, 0, 0)
            .unwrap(),
        Duration::parse("30m"),
        ClosedWindow::Both,
        TimeUnit::Milliseconds,
        None,
    )?;
    let df = df!(
        "time" => time,
        "groups" => &["a", "a", "a", "b", "b", "a", "a"],
        "values" => &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    )?;
    println!("{}", &df);
    // --8<-- [end:df]

    // --8<-- [start:upsample]
    let out1 = df
        .clone()
        .upsample::<[String; 0]>([], "time", Duration::parse("15m"))?
        .fill_null(FillNullStrategy::Forward(None))?;
    println!("{}", &out1);
    // --8<-- [end:upsample]

    // --8<-- [start:upsample2]
    let out2 = df
        .clone()
        .upsample::<[String; 0]>([], "time", Duration::parse("15m"))?
        .lazy()
        .with_columns([col("values").interpolate(InterpolationMethod::Linear)])
        .collect()?
        .fill_null(FillNullStrategy::Forward(None))?;
    println!("{}", &out2);
    // --8<-- [end:upsample2]
    Ok(())
}
