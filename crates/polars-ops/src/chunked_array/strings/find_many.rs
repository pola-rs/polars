use aho_corasick::{AhoCorasick, AhoCorasickBuilder};
use polars_core::prelude::*;

fn build_ac(patterns: &Utf8Chunked, ascii_case_insensitive: bool) -> PolarsResult<AhoCorasick> {
    AhoCorasickBuilder::new()
        .ascii_case_insensitive(ascii_case_insensitive)
        .build(patterns.downcast_iter().flatten().flatten())
        .map_err(|e| polars_err!(ComputeError: "could not build aho corasick automaton {}", e))
}

pub fn contains_any(
    ca: &Utf8Chunked,
    patterns: &Utf8Chunked,
    ascii_case_insensitive: bool,
) -> PolarsResult<BooleanChunked> {
    let ac = build_ac(patterns, ascii_case_insensitive)?;

    Ok(ca.apply_generic(|opt_val| opt_val.map(|val| ac.find(val).is_some())))
}

pub fn replace_all(
    ca: &Utf8Chunked,
    patterns: &Utf8Chunked,
    replace_with: &Utf8Chunked,
    ascii_case_insensitive: bool,
) -> PolarsResult<Utf8Chunked> {
    let replace_with = if replace_with.len() == 1 && patterns.len() > 1 {
        replace_with.new_from_index(0, patterns.len())
    } else {
        replace_with.clone()
    };

    polars_ensure!(patterns.len() == replace_with.len(), InvalidOperation: "expected the same amount of patterns as replacement strings");
    polars_ensure!(patterns.null_count() == 0 && replace_with.null_count() == 0, InvalidOperation: "'patterns'/'replace_with' should not have nulls");
    let replace_with = replace_with
        .downcast_iter()
        .flatten()
        .flatten()
        .collect::<Vec<_>>();

    let ac = build_ac(patterns, ascii_case_insensitive)?;

    Ok(ca.apply_generic(|opt_val| opt_val.map(|val| ac.replace_all(val, replace_with.as_slice()))))
}
