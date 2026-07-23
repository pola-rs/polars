use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_sql::*;

fn eval(needle_sql: &str, set_values: Option<Vec<Option<i32>>>) -> PolarsResult<Option<bool>> {
    let mut ctx = SQLContext::new();
    let set = match set_values {
        Some(values) => df! { "v" => values }?,
        None => DataFrame::new(0, vec![Column::new("v".into(), Vec::<i32>::new())])?,
    };
    ctx.register("set_tbl", set.lazy());
    let out = ctx
        .execute(&format!(
            "SELECT {needle_sql} IN (SELECT v FROM set_tbl) AS r"
        ))?
        .collect()?;
    Ok(out.column("r")?.bool()?.get(0))
}

fn eval_not(needle_sql: &str, set_values: Option<Vec<Option<i32>>>) -> PolarsResult<Option<bool>> {
    let mut ctx = SQLContext::new();
    let set = match set_values {
        Some(values) => df! { "v" => values }?,
        None => DataFrame::new(0, vec![Column::new("v".into(), Vec::<i32>::new())])?,
    };
    ctx.register("set_tbl", set.lazy());
    let out = ctx
        .execute(&format!(
            "SELECT {needle_sql} NOT IN (SELECT v FROM set_tbl) AS r"
        ))?
        .collect()?;
    Ok(out.column("r")?.bool()?.get(0))
}

#[test]
fn test_in_subquery_empty_set_non_null_needle() -> PolarsResult<()> {
    // `x IN (empty set)` is FALSE, even though nothing matches.
    assert_eq!(eval("1", None)?, Some(false));
    assert_eq!(eval_not("1", None)?, Some(true));
    Ok(())
}

#[test]
fn test_in_subquery_empty_set_null_needle() -> PolarsResult<()> {
    // An empty right-hand set beats a NULL left-hand operand: still FALSE / TRUE,
    // not unknown.
    assert_eq!(eval("null", None)?, Some(false));
    assert_eq!(eval_not("null", None)?, Some(true));
    Ok(())
}

#[test]
fn test_in_subquery_null_needle_non_empty_set() -> PolarsResult<()> {
    // NULL compared against a non-empty set (no NULLs in it) is unknown.
    assert_eq!(eval("null", Some(vec![Some(2), Some(3), Some(4)]))?, None);
    assert_eq!(
        eval_not("null", Some(vec![Some(2), Some(3), Some(4)]))?,
        None
    );
    Ok(())
}

#[test]
fn test_in_subquery_value_absent_no_nulls_in_set() -> PolarsResult<()> {
    // Value not present, set has no NULLs -> definitively FALSE / TRUE.
    assert_eq!(
        eval("1", Some(vec![Some(2), Some(3), Some(4)]))?,
        Some(false)
    );
    assert_eq!(
        eval_not("1", Some(vec![Some(2), Some(3), Some(4)]))?,
        Some(true)
    );
    Ok(())
}

#[test]
fn test_in_subquery_value_absent_set_has_null() -> PolarsResult<()> {
    // Value not present, but the set contains a NULL -> unknown, since that NULL
    // might have matched.
    assert_eq!(eval("1", Some(vec![Some(2), Some(3), None]))?, None);
    assert_eq!(eval_not("1", Some(vec![Some(2), Some(3), None]))?, None);
    Ok(())
}

#[test]
fn test_in_subquery_value_present_despite_set_null() -> PolarsResult<()> {
    // An actual match wins over a NULL elsewhere in the set.
    assert_eq!(eval("2", Some(vec![Some(2), Some(3), None]))?, Some(true));
    assert_eq!(
        eval_not("2", Some(vec![Some(2), Some(3), None]))?,
        Some(false)
    );
    Ok(())
}
