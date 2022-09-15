use polars_core::utils::{accumulate_dataframes_vertical, split_df};

use super::*;

#[test]
fn test_chunked_left_join() -> PolarsResult<()> {
    let mut band_members = df![
        "name" => ["john", "paul", "mick", "bob"],
        "band" => ["beatles", "beatles", "stones", "wailers"],
    ]?;

    let mut band_instruments = df![
        "name" => ["john", "paul", "keith"],
        "plays" => ["guitar", "bass", "guitar"]
    ]?;

    let band_instruments = accumulate_dataframes_vertical(split_df(&mut band_instruments, 2)?)?;
    let band_members = accumulate_dataframes_vertical(split_df(&mut band_members, 2)?)?;
    assert_eq!(band_instruments.n_chunks()?, 2);
    assert_eq!(band_members.n_chunks()?, 2);

    let out = band_instruments.join(&band_members, ["name"], ["name"], JoinType::Left, None)?;
    let expected = df![
        "name" => ["john", "paul", "keith"],
        "plays" => ["guitar", "bass", "guitar"],
        "band" => [Some("beatles"), Some("beatles"), None],
    ]?;
    assert!(out.frame_equal_missing(&expected));

    Ok(())
}
