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
fn test_cross_join_pd() -> Result<()> {
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
