use polars_io::local_path;
use polars_utils::aliases::{PlIndexMap, PlIndexSet};

use super::*;

/// Raise an error if a sink is used as the input of another operation, or if a file sink at
/// `root` (a sink or multiple sinks) writes to a local file that the query also scans, as the
/// sink would overwrite the file while it is being read.
pub(super) fn validate_sinks(root: Node, lp_arena: &Arena<IR>) -> PolarsResult<()> {
    let sink_nodes = match lp_arena.get(root) {
        IR::SinkMultiple { inputs } => inputs.as_slice(),
        _ => std::slice::from_ref(&root),
    };
    // Sinks keyed by file name, so scanned paths are only canonicalized on a name match.
    let mut sinks: PlIndexMap<_, Vec<_>> = PlIndexMap::default();
    for &node in sink_nodes {
        let IR::Sink {
            payload:
                SinkTypeIR::File(FileSinkOptions {
                    target: SinkTarget::Path(sink_path),
                    ..
                }),
            ..
        } = lp_arena.get(node)
        else {
            continue;
        };
        let Some(local) = local_path(sink_path) else {
            continue;
        };
        let Some(file_name) = local.file_name() else {
            continue;
        };
        let Ok(canonical) = std::fs::canonicalize(&local) else {
            continue;
        };
        sinks
            .entry(file_name.to_owned())
            .or_default()
            .push((canonical, sink_path));
    }

    let mut stack = vec![root];
    let mut visited = PlIndexSet::default();
    while let Some(node) = stack.pop() {
        if !visited.insert(node) {
            continue;
        }
        let ir = lp_arena.get(node);
        let is_sink_multiple = matches!(ir, IR::SinkMultiple { .. });
        for input in ir.inputs() {
            let is_valid_input = match lp_arena.get(input) {
                IR::Sink { .. } => is_sink_multiple,
                IR::SinkMultiple { .. } => false,
                _ => true,
            };
            polars_ensure!(
                is_valid_input,
                InvalidOperation: "a sink cannot be used as the input of another operation"
            );
            stack.push(input);
        }

        if sinks.is_empty() {
            continue;
        }
        let IR::Scan {
            sources: ScanSources::Paths(paths),
            ..
        } = ir
        else {
            continue;
        };
        for path in paths.iter() {
            let Some(local) = local_path(path) else {
                continue;
            };
            let Some(matching_sinks) = local.file_name().and_then(|name| sinks.get(name)) else {
                continue;
            };
            let Ok(canonical) = std::fs::canonicalize(&local) else {
                continue;
            };
            if let Some((_, sink_path)) = matching_sinks
                .iter()
                .find(|(sink_canonical, _)| *sink_canonical == canonical)
            {
                polars_bail!(
                    InvalidOperation: "cannot sink to '{}' because the query also reads from this file; \
                    collect the query first or write to a different path",
                    sink_path
                );
            }
        }
    }
    Ok(())
}
