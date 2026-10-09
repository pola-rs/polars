use polars_core::series::Series;
use polars_defs::join::JoinValidation;
use polars_error::{PolarsResult, polars_ensure};

pub(crate) fn validate_probe(
    validation: JoinValidation,
    s_left: &Series,
    s_right: &Series,
    build_shortest_table: bool,
    nulls_equal: bool,
) -> PolarsResult<()> {
    // In default, probe is the left series.
    //
    // In inner join and outer join, the shortest relation will be used to create a hash table.
    // In left join, always use the right side to create.
    //
    // If `build_shortest_table` and left is shorter, swap. Then rhs will be the probe.
    // If left == right, swap too. (apply the same logic as `det_hash_prone_order`)
    let should_swap = build_shortest_table && s_left.len() <= s_right.len();
    let probe = if should_swap { s_right } else { s_left };

    use JoinValidation::*;
    let valid = match validation.swap(should_swap) {
        // Only check the `build` side.
        // The other side use `validate_build` to check
        ManyToMany | ManyToOne => true,
        OneToMany | OneToOne => {
            if !nulls_equal && probe.null_count() > 0 {
                probe.n_unique()? - 1 == probe.len() - probe.null_count()
            } else {
                probe.n_unique()? == probe.len()
            }
        },
    };
    polars_ensure!(valid, ComputeError: "join keys did not fulfill {} validation", validation);
    Ok(())
}

pub(crate) fn validate_build(
    validation: JoinValidation,
    build_size: usize,
    expected_size: usize,
    swapped: bool,
) -> PolarsResult<()> {
    use JoinValidation::*;

    // In default, build is in rhs.
    let valid = match validation.swap(swapped) {
        // Only check the `build` side.
        // The other side use `validate_prone` to check
        ManyToMany | OneToMany => true,
        ManyToOne | OneToOne => build_size == expected_size,
    };
    polars_ensure!(valid, ComputeError: "join keys did not fulfill {} validation", validation);
    Ok(())
}
