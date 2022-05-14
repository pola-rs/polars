use polars::prelude::*;
use serde_json::json;

fn main() -> Result<()> {
    //let s: Series = [
        //&json!(r#"{"a": 1, "b": [{"c": 0}, {"c": 1}]}"#).to_string(),
        //&json!(r#"{"a": 2, "b": [{"c": 2}, {"c": 5}]}"#).to_string(),
    //].iter().collect();
    let s = Series::new(
        "json",
        [
            r#"{"a": 1, "b": [{"c": 0}, {"c": 1}]}"#,
            r#"{"a": 2, "b": [{"c": 2}, {"c": 5}]}"#,
        ]
    );
    let ca = s.utf8()?;

    dbg!(ca);
    dbg!(ca.str_lengths().into_series());
    dbg!(ca.json_path_match("$.a")?);
    dbg!(ca.json_path_extract("$.a")?);
    dbg!(ca.json_path_match("$.b")?);
    dbg!(ca.json_path_extract("$.b")?);
    dbg!(ca.json_path_extract("$.b")?.dtype());
    dbg!(ca.json_path_extract("$.b[:].c")?);
    dbg!(ca.json_path_extract("$.b[:].c")?.dtype());
    Ok(())
}
