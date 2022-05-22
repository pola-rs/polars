use polars::prelude::*;

fn main() -> Result<()> {
    let s = Series::new(
        "json",
        [
            Some(r#"{"a": 1, "b": [{"c": 0}, {"c": 1}]}"#),
            Some(r#"{"a": 2, "b": [{"c": 2}, {"c": 5}]}"#),
            None,
        ]
    );
    let ca = s.utf8()?;

    dbg!(ca);
    dbg!(ca.str_lengths().into_series());
    dbg!(ca.json_path_select("$.a")?);
    dbg!(ca.json_path_extract("$.a", None)?);
    dbg!(ca.json_path_select("$.b")?);
    dbg!(ca.json_path_extract("$.b", None)?);
    dbg!(ca.json_path_extract("$.b", None)?.dtype());
    dbg!(ca.json_path_extract("$.b[:].c", None)?);
    dbg!(ca.json_path_extract("$.b[:].c", None)?.dtype());
    Ok(())
}
