//! The tpch files only got ten rows, so after all the joins filters there is not data
//! Still we can use this to test the schema, operation correctness on empty data, and optimizations
//! taken.
use super::*;

const fn base_path() -> &'static str {
    "../../examples/datasets/tpc_heads"
}

fn region() -> LazyFrame {
    let base_path = base_path();
    LazyFrame::scan_ipc(
        format!("{base_path}/region.feather"),
        ScanArgsIpc::default(),
    )
    .unwrap()
}
fn nation() -> LazyFrame {
    let base_path = base_path();
    LazyFrame::scan_ipc(
        format!("{base_path}/nation.feather"),
        ScanArgsIpc::default(),
    )
    .unwrap()
}

fn supplier() -> LazyFrame {
    let base_path = base_path();
    LazyFrame::scan_ipc(
        format!("{base_path}/supplier.feather"),
        ScanArgsIpc::default(),
    )
    .unwrap()
}

fn part() -> LazyFrame {
    let base_path = base_path();
    LazyFrame::scan_ipc(format!("{base_path}/part.feather"), ScanArgsIpc::default()).unwrap()
}

fn partsupp() -> LazyFrame {
    let base_path = base_path();
    LazyFrame::scan_ipc(
        format!("{base_path}/partsupp.feather"),
        ScanArgsIpc::default(),
    )
    .unwrap()
}

#[test]
fn test_q2() -> PolarsResult<()> {
    let q1 = part()
        .inner_join(partsupp(), "p_partkey", "ps_partkey")
        .inner_join(supplier(), "ps_suppkey", "s_suppkey")
        .inner_join(nation(), "s_nationkey", "n_nationkey")
        .inner_join(region(), "n_regionkey", "r_regionkey")
        .filter(col("p_size").eq(15))
        .filter(col("p_type").str().ends_with(lit("BRASS".to_string())));
    let q = q1
        .clone()
        .group_by([col("p_partkey")])
        .agg([col("ps_supplycost").min()])
        .join(
            q1,
            [col("p_partkey"), col("ps_supplycost")],
            [col("p_partkey"), col("ps_supplycost")],
            JoinType::Inner.into(),
        )
        .select([cols([
            "s_acctbal",
            "s_name",
            "n_name",
            "p_partkey",
            "p_mfgr",
            "s_address",
            "s_phone",
            "s_comment",
        ])])
        .sort_by_exprs(
            [cols(["s_acctbal", "n_name", "s_name", "p_partkey"])],
            SortMultipleOptions::default()
                .with_order_descending_multi([true, false, false, false])
                .with_maintain_order(true),
        )
        .limit(100)
        .with_comm_subplan_elim(true);

    let IRPlan {
        lp_top, lp_arena, ..
    } = q.clone().to_alp_optimized().unwrap();
    assert_eq!(
        (&lp_arena)
            .iter(lp_top)
            .filter(|(_, alp)| matches!(alp, IR::Cache { .. }))
            .count(),
        2
    );

    let out = q.collect()?;
    let schema = Schema::from_iter([
        Field::new("s_acctbal", DataType::Float64),
        Field::new("s_name", DataType::String),
        Field::new("n_name", DataType::String),
        Field::new("p_partkey", DataType::Int64),
        Field::new("p_mfgr", DataType::String),
        Field::new("s_address", DataType::String),
        Field::new("s_phone", DataType::String),
        Field::new("s_comment", DataType::String),
    ]);
    assert_eq!(&out.schema(), &schema);

    Ok(())
}
