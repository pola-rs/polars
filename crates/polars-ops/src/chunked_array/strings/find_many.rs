use std::ops::Deref;

use aho_corasick::{AhoCorasick, AhoCorasickBuilder};
use arrow::array::{Utf8ViewArray, View};
use arrow::buffer::Buffer;
use polars_core::prelude::*;
use polars_core::utils::align_chunks_binary;

use super::utils::subview;

fn build_ac(patterns: &StringChunked, ascii_case_insensitive: bool) -> PolarsResult<AhoCorasick> {
    AhoCorasickBuilder::new()
        .ascii_case_insensitive(ascii_case_insensitive)
        .build(patterns.downcast_iter().flatten().flatten())
        .map_err(|e| polars_err!(ComputeError: "could not build aho corasick automaton {}", e))
}

fn build_ac_arr(
    patterns: &Utf8ViewArray,
    ascii_case_insensitive: bool,
) -> PolarsResult<AhoCorasick> {
    AhoCorasickBuilder::new()
        .ascii_case_insensitive(ascii_case_insensitive)
        .build(patterns.into_iter().flatten())
        .map_err(|e| polars_err!(ComputeError: "could not build aho corasick automaton {}", e))
}

pub fn contains_any(
    ca: &StringChunked,
    patterns: &StringChunked,
    ascii_case_insensitive: bool,
) -> PolarsResult<BooleanChunked> {
    let ac = build_ac(patterns, ascii_case_insensitive)?;

    Ok(ca.apply_generic(|opt_val| opt_val.map(|val| ac.find(val).is_some())))
}

pub fn replace_all(
    ca: &StringChunked,
    patterns: &StringChunked,
    replace_with: &StringChunked,
    ascii_case_insensitive: bool,
) -> PolarsResult<StringChunked> {
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

fn push(
    val: &str,
    builder: &mut ListStringChunkedBuilder,
    ac: &AhoCorasick,
    overlapping: bool,
    view: View,
    buffers: &[Buffer<u8>],
) {
    if overlapping {
        let iter = ac.find_overlapping_iter(val);
        let iter = iter.map(|m| subview(val, view, m.start(), m.end()));
        builder.append_views_iter(iter, buffers);
    } else {
        let iter = ac.find_iter(val);
        let iter = iter.map(|m| subview(val, view, m.start(), m.end()));
        builder.append_views_iter(iter, buffers);
    }
}

pub fn extract_many(
    ca: &StringChunked,
    patterns: &Series,
    ascii_case_insensitive: bool,
    overlapping: bool,
) -> PolarsResult<ListChunked> {
    match patterns.dtype() {
        DataType::List(inner) if inner.is_string() => {
            let mut builder = ListStringChunkedBuilder::new(ca.name(), ca.len(), ca.len() * 2);
            let patterns = patterns.list().unwrap();
            let (ca, patterns) = align_chunks_binary(ca, patterns);

            for (arr, pat_arr) in ca.downcast_iter().zip(patterns.downcast_iter()) {
                let buffers = arr.data_buffers().deref();
                for z in arr.iter().zip(arr.views().iter()).zip(pat_arr) {
                    match z {
                        ((Some(val), &view), Some(pat)) => {
                            let pat = pat.as_any().downcast_ref::<Utf8ViewArray>().unwrap();
                            let ac = build_ac_arr(pat, ascii_case_insensitive)?;
                            push(val, &mut builder, &ac, overlapping, view, buffers);
                        },
                        _ => builder.append_null(),
                    }
                }
            }
            Ok(builder.finish())
        },
        DataType::String => {
            let patterns = patterns.str().unwrap();
            let ac = build_ac(patterns, ascii_case_insensitive)?;
            let mut builder = ListStringChunkedBuilder::new(ca.name(), ca.len(), ca.len() * 2);

            for arr in ca.downcast_iter() {
                let buffers = arr.data_buffers().deref();
                for (opt_val, &view) in arr.iter().zip(arr.views().iter()) {
                    if let Some(val) = opt_val {
                        push(val, &mut builder, &ac, overlapping, view, buffers);
                    } else {
                        builder.append_null();
                    }
                }
            }
            Ok(builder.finish())
        },
        _ => {
            polars_bail!(InvalidOperation: "expected 'String/List<String>' datatype for 'patterns' argument")
        },
    }
}
