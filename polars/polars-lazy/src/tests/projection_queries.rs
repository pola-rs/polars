use super::*;

#[test]
fn test_join_suffix_and_drop() -> PolarsResult<()> {
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
#[cfg(feature = "cross_join")]
fn test_cross_join_pd() -> PolarsResult<()> {
    let food = df![
        "name"=> ["Omelette", "Fried Egg"],
        "price" => [8, 5]
    ]?;

    let drink = df![
        "name" => ["Orange Juice", "Tea"],
        "price" => [5, 4]
    ]?;

    let q = food.lazy().cross_join(drink.lazy()).select([
        col("name").alias("food"),
        col("name_right").alias("beverage"),
        (col("price") + col("price_right")).alias("total"),
    ]);

    let out = q.collect()?;
    let expected = df![
        "food" => ["Omelette", "Omelette", "Fried Egg", "Fried Egg"],
        "beverage" => ["Orange Juice", "Tea", "Orange Juice", "Tea"],
        "total" => [13, 12, 10, 9]
    ]?;

    assert!(out.frame_equal(&expected));
    Ok(())
}

#[test]
fn test_row_count_pd() -> PolarsResult<()> {
    let df = df![
        "x" => [1, 2, 3],
        "y" => [3, 2, 1],
    ]?;

    let df = df
        .lazy()
        .with_row_count("row_count", None)
        .select([col("row_count"), col("x") * lit(3i32)])
        .collect()?;

    let expected = df![
        "row_count" => [0 as IdxSize, 1, 2],
        "x" => [3i32, 6, 9]
    ]?;

    assert!(df.frame_equal(&expected));

    Ok(())
}

#[test]
#[cfg(feature = "cse")]
fn scan_join_same_file() -> PolarsResult<()> {
    let lf = LazyCsvReader::new(FOODS_CSV.to_string()).finish()?;

    for cse in [true, false] {
        let partial = lf.clone().select([col("category")]).limit(5);
        let q = lf
            .clone()
            .join(
                partial,
                [col("category")],
                [col("category")],
                JoinType::Inner,
            )
            .with_common_subplan_elimination(cse);
        let out = q.collect()?;
        assert_eq!(
            out.get_column_names(),
            &["category", "calories", "fats_g", "sugars_g"]
        );
    }
    Ok(())
}

#[test]
#[cfg(all(feature = "regex", feature = "concat_str"))]
fn concat_str_regex_expansion() -> PolarsResult<()> {
    let df = df![
        "a"=> [1, 1, 1],
        "b_a_1"=> ["a--", "", ""],
        "b_a_2"=> ["", "b--", ""],
        "b_a_3"=> ["", "", "c--"]
    ]?
    .lazy();
    let out = df
        .select([concat_str([col(r"^b_a_\d$")], ";").alias("concatenated")])
        .collect()?;
    let s = out.column("concatenated")?;
    assert_eq!(s, &Series::new("concatenated", ["a--;;", ";b--;", ";;c--"]));

    Ok(())
}
