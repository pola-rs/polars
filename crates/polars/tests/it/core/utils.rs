use super::*;

#[test]
fn test_df_macro_trailing_commas() -> PolarsResult<()> {
    let a = df! {
        "a" => &["a one", "a two"],
        "b" => &["b one", "b two"],
        "c" => &[1, 2]
    }?;

    let b = df! {
        "a" => &["a one", "a two"],
        "b" => &["b one", "b two"],
        "c" => &[1, 2],
    }?;

    assert!(a.frame_equal(&b));
    Ok(())
}
