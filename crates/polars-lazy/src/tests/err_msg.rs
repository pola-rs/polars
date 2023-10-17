use polars_core::error::ErrString;

use super::*;

#[test]
fn error_messages() {
    fn test_it(df: &LazyFrame, n: usize, base_err_msg: &str) {
        let plan_err_str;
        let collect_err;

        if n == 0 {
            plan_err_str = format!(
                "ErrorState(NotYetEncountered { \
                 err: ColumnNotFound(ErrString(\"{base_err_msg}\")) })"
            );
            collect_err = PolarsError::ColumnNotFound(ErrString::from(&base_err_msg));
        } else {
            plan_err_str = format!(
                "ErrorState(AlreadyEncountered { \
                 n_times: {n}, prev_err_msg: \"not found: {base_err_msg}\" })"
            );
            collect_err = PolarsError::ComputeError(ErrString::from(format!(
                "LogicalPlan already failed (depth: {n}) \
                 with error: 'not found: {base_err_msg}"
            )))
        };

        assert_eq!(df.describe_plan(), plan_err_str);
        assert_eq!(df.collect().unwrap_err(), collect_err);
    }

    let initial_projection: &str = r#"DF ["c1"]; PROJECT */1 COLUMNS; SELECTION: "None""#;
    let base_err_msg =
        format!("xyz\n\nError originated just after this operation:\n{INITIAL_PROJECTION}");

    let err_msg =
        format!("xyz\n\nError originated just after this operation:\n{INITIAL_PROJECTION}");

    let df = df! [
        "c1" => [0, 1],
    ]
    .unwrap()
    .lazy();

    assert_eq!(df.describe_plan(), initial_projection);

    let df0 = df.clone().select([col("xyz")]);
    test_it(&df0, 0, &base_err_msg);

    let df1 = df.clone().select([col("xyz")]).select([col("c1")]);
    test_it(&df1, 1, &base_err_msg);

    let df2 = df
        .clone()
        .select([col("xyz")])
        .select([col("c1")])
        .select([col("c2")]);
    test_it(&df2, 2, &base_err_msg);

    let df3 = df
        .clone()
        .select([col("xyz")])
        .select([col("c1")])
        .select([col("c2")])
        .select([col("c3")]);
    test_it(&df3, 3, &base_err_msg);
}
