use polars_core::error::ErrString;

use super::*;

#[test]
fn error_messages() {
    fn assert_errors_eq(e1: &PolarsError, e2: &PolarsError) {
        use PolarsError::*;
        match (e1, e2) {
            (ColumnNotFound(s1), ColumnNotFound(s2)) => {
                assert_eq!(s1.as_ref(), s2.as_ref());
            },
            (ComputeError(s1), ComputeError(s2)) => {
                assert_eq!(s1.as_ref(), s2.as_ref());
            },
            _ => panic!("{e1:?} != {e2:?}"),
        }
    }

    fn test_it(df: LazyFrame, n: usize) {
        let base_err_msg = format!(
            "xyz\n\nError originated just after this \
             operation:\n{INITIAL_PROJECTION_STR}"
        );
        let col_not_found_err_msg = format!("not found: {base_err_msg}");

        let plan_err_str;
        let collect_err;

        if n == 0 {
            plan_err_str = format!(
                "ErrorState(NotYetEncountered {{ \
                 err: ColumnNotFound(ErrString({base_err_msg:?})) }})"
            );
            collect_err = PolarsError::ColumnNotFound(ErrString::from(base_err_msg.to_owned()));
        } else {
            plan_err_str = format!(
                "ErrorState(AlreadyEncountered {{ n_times: {n}, \
                 prev_err_msg: {col_not_found_err_msg:?} }})",
            );
            collect_err = PolarsError::ComputeError(ErrString::from(format!(
                "LogicalPlan already failed (depth: {n}) \
                 with error: '{col_not_found_err_msg}'"
            )))
        };

        assert_eq!(df.describe_plan(), plan_err_str);
        assert_errors_eq(&df.collect().unwrap_err(), &collect_err);
    }

    const INITIAL_PROJECTION_STR: &str = r#"DF ["c1"]; PROJECT */1 COLUMNS; SELECTION: "None""#;

    let df = df! [
        "c1" => [0, 1],
    ]
    .unwrap()
    .lazy();

    assert_eq!(df.describe_plan(), INITIAL_PROJECTION_STR);

    let df0 = df.clone().select([col("xyz")]);
    test_it(df0, 0);

    let df1 = df.clone().select([col("xyz")]).select([col("c1")]);
    test_it(df1, 1);

    let df2 = df
        .clone()
        .select([col("xyz")])
        .select([col("c1")])
        .select([col("c2")]);
    test_it(df2, 2);

    let df3 = df
        .clone()
        .select([col("xyz")])
        .select([col("c1")])
        .select([col("c2")])
        .select([col("c3")]);
    test_it(df3, 3);
}
