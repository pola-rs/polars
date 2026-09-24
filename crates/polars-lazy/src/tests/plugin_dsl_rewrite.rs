use super::*;

/// Rewrites cannot be lowered to IR if no functionality to resolve them is given.
#[test]
fn test_rewrite_input_outside_rewrite() {
    assert!(
        load_df()
            .lazy()
            .select([Expr::RewriteInput(0) + col("a")])
            .collect_schema()
            .is_err()
    );
}
