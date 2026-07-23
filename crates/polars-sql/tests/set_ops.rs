use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_sql::*;

#[test]
#[cfg(feature = "semi_anti_join")]
fn except_distinct_basic() {
    let df1 = df!["x" => [1, 1, 2, 3]].unwrap();
    let df2 = df!["x" => [2]].unwrap();
    let mut ctx = SQLContext::new();
    ctx.register("df1", df1.lazy());
    ctx.register("df2", df2.lazy());

    let actual = ctx
        .execute("SELECT x FROM df1 EXCEPT SELECT x FROM df2 ORDER BY x")
        .unwrap()
        .collect()
        .unwrap();
    let expected = df!["x" => [1, 3]].unwrap();
    assert!(actual.equals(&expected));
}

#[test]
#[cfg(feature = "semi_anti_join")]
fn intersect_distinct_basic() {
    let df1 = df!["x" => [1, 1, 2, 3]].unwrap();
    let df2 = df!["x" => [1, 2, 2]].unwrap();
    let mut ctx = SQLContext::new();
    ctx.register("df1", df1.lazy());
    ctx.register("df2", df2.lazy());

    let actual = ctx
        .execute("SELECT x FROM df1 INTERSECT SELECT x FROM df2 ORDER BY x")
        .unwrap()
        .collect()
        .unwrap();
    let expected = df!["x" => [1, 2]].unwrap();
    assert!(actual.equals(&expected));
}

#[test]
#[cfg(feature = "semi_anti_join")]
fn except_all_bag_semantics() {
    // x=1: 3 copies in df1, 2 in df2 -> 1 remains
    // x=2: 3 copies in df1, 2 in df2 -> 1 remains
    // x=3: 1 copy in df1, 0 in df2 -> 1 remains
    let df1 = df!["x" => [1, 1, 1, 2, 2, 2, 3]].unwrap();
    let df2 = df!["x" => [1, 1, 2, 2]].unwrap();
    let mut ctx = SQLContext::new();
    ctx.register("df1", df1.lazy());
    ctx.register("df2", df2.lazy());

    let actual = ctx
        .execute("SELECT x FROM df1 EXCEPT ALL SELECT x FROM df2 ORDER BY x")
        .unwrap()
        .collect()
        .unwrap();
    let expected = df!["x" => [1, 2, 3]].unwrap();
    assert!(actual.equals(&expected));
}

#[test]
#[cfg(feature = "semi_anti_join")]
fn intersect_all_bag_semantics() {
    // x=1: min(3, 2) = 2
    // x=2: min(3, 2) = 2
    // x=3: min(1, 0) = 0
    let df1 = df!["x" => [1, 1, 1, 2, 2, 2, 3]].unwrap();
    let df2 = df!["x" => [1, 1, 2, 2]].unwrap();
    let mut ctx = SQLContext::new();
    ctx.register("df1", df1.lazy());
    ctx.register("df2", df2.lazy());

    let actual = ctx
        .execute("SELECT x FROM df1 INTERSECT ALL SELECT x FROM df2 ORDER BY x")
        .unwrap()
        .collect()
        .unwrap();
    let expected = df!["x" => [1, 1, 2, 2]].unwrap();
    assert!(actual.equals(&expected));
}

#[test]
#[cfg(feature = "semi_anti_join")]
fn except_distinct_nulls_compare_equal() {
    // NULLs must compare equal for set-op purposes: the NULL row in df1
    // is matched away by the NULL row in df2.
    let df1 = df!["x" => [Some(1), Some(2), None]].unwrap();
    let df2 = df!["x" => [Some(2), None]].unwrap();
    let mut ctx = SQLContext::new();
    ctx.register("df1", df1.lazy());
    ctx.register("df2", df2.lazy());

    let actual = ctx
        .execute("SELECT x FROM df1 EXCEPT SELECT x FROM df2")
        .unwrap()
        .collect()
        .unwrap();
    // NULL in df2 matches (removes) the NULL row in df1; only the non-matching
    // value 1 remains
    let expected = df!["x" => [1]].unwrap();
    assert!(actual.equals(&expected));
}

#[test]
#[cfg(feature = "semi_anti_join")]
fn except_all_nulls_bag_counts() {
    // bag semantics: df1 has two NULLs, df2 has one -> one NULL remains
    let df1 = df!["x" => [Some(1), None, None]].unwrap();
    let df2 = df!["x" => [Some(1), None]].unwrap();
    let mut ctx = SQLContext::new();
    ctx.register("df1", df1.lazy());
    ctx.register("df2", df2.lazy());

    let actual = ctx
        .execute("SELECT x FROM df1 EXCEPT ALL SELECT x FROM df2")
        .unwrap()
        .collect()
        .unwrap();
    let expected = df!["x" => [None::<i32>]].unwrap();
    assert!(actual.equals_missing(&expected));
}

#[test]
#[cfg(feature = "semi_anti_join")]
fn mixed_chain_three_terms() {
    // (a UNION b) EXCEPT c, tree per sqlparser's left-to-right equal-precedence
    // parsing of UNION/EXCEPT (INTERSECT would bind tighter, not relevant here)
    let a = df!["v" => [1, 2]].unwrap();
    let b = df!["v" => [2, 3]].unwrap();
    let c = df!["v" => [2]].unwrap();
    let mut ctx = SQLContext::new();
    ctx.register("a", a.lazy());
    ctx.register("b", b.lazy());
    ctx.register("c", c.lazy());

    let actual = ctx
        .execute("SELECT v FROM a UNION SELECT v FROM b EXCEPT SELECT v FROM c ORDER BY v")
        .unwrap()
        .collect()
        .unwrap();
    let expected = df!["v" => [1, 3]].unwrap();
    assert!(actual.equals(&expected));
}

#[test]
#[cfg(feature = "semi_anti_join")]
fn mixed_chain_with_intersect_precedence() {
    // INTERSECT binds tighter than UNION/EXCEPT: `a UNION b INTERSECT c` parses
    // as `a UNION (b INTERSECT c)`.
    let a = df!["v" => [1, 2]].unwrap();
    let b = df!["v" => [2, 3]].unwrap();
    let c = df!["v" => [1, 3]].unwrap();
    let mut ctx = SQLContext::new();
    ctx.register("a", a.lazy());
    ctx.register("b", b.lazy());
    ctx.register("c", c.lazy());

    let actual = ctx
        .execute("SELECT v FROM a UNION SELECT v FROM b INTERSECT SELECT v FROM c ORDER BY v")
        .unwrap()
        .collect()
        .unwrap();
    // b INTERSECT c = {3}; a UNION {3} = {1, 2, 3}
    let expected = df!["v" => [1, 2, 3]].unwrap();
    assert!(actual.equals(&expected));
}

#[test]
#[cfg(feature = "semi_anti_join")]
fn four_term_mixed_chain() {
    let a = df!["v" => [1, 2]].unwrap();
    let b = df!["v" => [2, 3]].unwrap();
    let c = df!["v" => [1, 3]].unwrap();
    let d = df!["v" => [1, 2, 3, 4]].unwrap();
    let mut ctx = SQLContext::new();
    ctx.register("a", a.lazy());
    ctx.register("b", b.lazy());
    ctx.register("c", c.lazy());
    ctx.register("d", d.lazy());

    // (a UNION ALL b) EXCEPT (c INTERSECT d): c INTERSECT d = {1, 3};
    // a UNION ALL b = [1, 2, 2, 3]; removing {1, 3} then DISTINCT -> {2}
    let actual = ctx
        .execute(
            "SELECT v FROM a UNION ALL SELECT v FROM b \
             EXCEPT SELECT v FROM c INTERSECT SELECT v FROM d ORDER BY v",
        )
        .unwrap()
        .collect()
        .unwrap();
    let expected = df!["v" => [2]].unwrap();
    assert!(actual.equals(&expected));
}

#[test]
#[cfg(feature = "semi_anti_join")]
fn except_positional_column_matching_differing_names() {
    // set-op columns are matched positionally, using the first operand's
    // names for the output -- names need not match between operands
    let df1 = df!["a" => [1, 2, 3], "b" => [10, 20, 30]].unwrap();
    let df2 = df!["x" => [2, 3, 4], "y" => [20, 30, 40]].unwrap();
    let mut ctx = SQLContext::new();
    ctx.register("df1", df1.lazy());
    ctx.register("df2", df2.lazy());

    let actual = ctx
        .execute("SELECT a, b FROM df1 EXCEPT SELECT x, y FROM df2 ORDER BY a")
        .unwrap()
        .collect()
        .unwrap();
    let expected = df!["a" => [1], "b" => [10]].unwrap();
    assert!(actual.equals(&expected));
    assert_eq!(actual.get_column_names(), vec!["a", "b"]);
}

#[test]
#[cfg(feature = "semi_anti_join")]
fn union_chain_order_by_and_limit_on_whole_chain() {
    let a = df!["v" => [3, 1]].unwrap();
    let b = df!["v" => [5, 2]].unwrap();
    let c = df!["v" => [4, 6]].unwrap();
    let mut ctx = SQLContext::new();
    ctx.register("a", a.lazy());
    ctx.register("b", b.lazy());
    ctx.register("c", c.lazy());

    let actual = ctx
        .execute(
            "SELECT v FROM a UNION SELECT v FROM b UNION SELECT v FROM c \
             ORDER BY v LIMIT 3",
        )
        .unwrap()
        .collect()
        .unwrap();
    let expected = df!["v" => [1, 2, 3]].unwrap();
    assert!(actual.equals(&expected));
}

#[test]
#[cfg(feature = "semi_anti_join")]
fn except_intersect_all_unsupported_by_name() {
    let df1 = df!["n" => [1, 1, 1, 2, 2, 2, 3]].unwrap();
    let df2 = df!["n" => [1, 1, 2, 2]].unwrap();
    let mut ctx = SQLContext::new();
    ctx.register("df1", df1.lazy());
    ctx.register("df2", df2.lazy());

    let res = ctx.execute("SELECT * FROM df1 EXCEPT ALL BY NAME SELECT * FROM df2");
    let err = match res {
        Err(e) => e,
        Ok(_) => panic!("expected an error"),
    };
    assert!(err.to_string().contains("EXCEPT ALL BY NAME"));
}
