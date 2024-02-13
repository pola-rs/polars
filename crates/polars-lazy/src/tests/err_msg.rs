use polars_core::error::ErrString;

use super::*;

const INITIAL_PROJECTION_STR: &str = r#"DF ["c1"]; PROJECT */1 COLUMNS; SELECTION: "None""#;

fn make_df() -> LazyFrame {
    df! [ "c1" => [0, 1] ].unwrap().lazy()
}

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

#[test]
fn col_not_found_error_messages() {
    fn get_err_msg(err_msg: &str, n: usize) -> String {
        let plural_s;
        let was_were;

        if n == 1 {
            plural_s = "";
            was_were = "was"
        } else {
            plural_s = "s";
            was_were = "were";
        };
        format!(
            "{err_msg}\n\nLogicalPlan had already failed with the above error; \
             after failure, {n} additional operation{plural_s} \
             {was_were} attempted on the LazyFrame"
        )
    }
    fn test_col_not_found(df: LazyFrame, n: usize) {
        let err_msg = format!(
            "xyz\n\nError originated just after this \
             operation:\n{INITIAL_PROJECTION_STR}"
        );

        let plan_err_str =
            format!("ErrorState {{ n_times: {n}, err: ColumnNotFound(ErrString({err_msg:?})) }}");

        let collect_err = if n == 0 {
            PolarsError::ColumnNotFound(ErrString::from(err_msg.to_owned()))
        } else {
            PolarsError::ColumnNotFound(ErrString::from(get_err_msg(&err_msg, n)))
        };

        assert_eq!(df.describe_plan(), plan_err_str);
        assert_errors_eq(&df.collect().unwrap_err(), &collect_err);
    }

    let df = make_df();

    assert_eq!(df.describe_plan(), INITIAL_PROJECTION_STR);

    test_col_not_found(df.clone().select([col("xyz")]), 0);
    test_col_not_found(df.clone().select([col("xyz")]).select([col("c1")]), 1);
    test_col_not_found(
        df.clone()
            .select([col("xyz")])
            .select([col("c1")])
            .select([col("c2")]),
        2,
    );
    test_col_not_found(
        df.clone()
            .select([col("xyz")])
            .select([col("c1")])
            .select([col("c2")])
            .select([col("c3")]),
        3,
    );
}
