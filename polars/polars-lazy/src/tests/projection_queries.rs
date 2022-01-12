use super::*;

#[test]
fn test_join_suffix_and_drop() -> Result<()> {
    let weight = df![
        "id" => [1, 2, 3, 4, 5, 0],
        "wgt" => [4.32, 5.23, 2.33, 23.399, 392.2, 0.0]
    ]?
    .lazy();

    let ped = df![
        "id"=> [1, 2, 3, 4, 5],
        "sireid"=> [0, 0, 1, 3, 3]
    ]?
    .lazy();

    let sumry = weight
        .clone()
        .filter(col("id").eq(lit(2i32)))
        .inner_join(ped, "id", "id");

    let out = sumry
        .join_builder()
        .with(weight)
        .left_on([col("sireid")])
        .right_on([col("id")])
        .suffix("_sire")
        .finish()
        .drop_columns(["sireid"])
        .collect()?;

    assert_eq!(out.shape(), (1, 3));

    Ok(())
}

#[test]
fn test_union_and_agg_projections() -> Result<()> {
    init_files();
    // a union vstacks columns and aggscan optimization determines columns to aggregate in a
    // hashmap, if that doesn't set them sorted the vstack will panic.
    let glob = "../../examples/aggregate_multiple_files_in_chunks/datasets/*.parquet";
    let lf1 = LazyFrame::scan_parquet(glob.into(), Default::default())?;
    let glob = "../../examples/aggregate_multiple_files_in_chunks/datasets/*.ipc";
    let lf2 = LazyFrame::scan_ipc(glob.into(), Default::default())?;
    let glob = "../../examples/aggregate_multiple_files_in_chunks/datasets/*.csv";
    let lf3 = LazyCsvReader::new(glob.into()).finish()?;

    for lf in [lf1, lf2, lf3] {
        let lf = lf.filter(col("category").eq(lit("vegetables"))).select([
            col("fats_g").sum().alias("sum"),
            col("fats_g").cast(DataType::Float64).mean().alias("mean"),
            col("fats_g").min().alias("min"),
        ]);

        let out = lf.collect()?;
        assert_eq!(out.shape(), (1, 3));
    }

    Ok(())
}
