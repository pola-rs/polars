use std::hash::{BuildHasher, Hash, Hasher};

use IR::*;
use polars_core::error::PolarsResult;
use polars_utils::aliases::{PlFixedStateQuality, PlIndexMap};
use polars_utils::arena::{Arena, Node};
use polars_utils::idx_vec::UnitVec;
use polars_utils::unitvec;

use super::OptimizationRule;
use crate::dsl::{FileScanIR, PredicateFileSkip, ScanSources, UnifiedScanArgs};
use crate::plans::{
    AExpr, ArenaExprIter, ExprIR, ExpressionComparator, FileInfo, IRFunctionExpr, ScanStats,
    is_elementwise_rec, is_scalar_ae,
};
use crate::prelude::IR;

/// Flattens nested unions, and merges union inputs that are the same row-wise nodes over scans
/// of the same kind into one scan over all their sources.
pub struct FlattenUnionRule {}

impl OptimizationRule for FlattenUnionRule {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> PolarsResult<Option<IR>> {
        let Union { inputs, options } = lp_arena.get(node) else {
            return Ok(None);
        };
        let mut options = *options;

        let flatten = inputs.iter().any(|node| match lp_arena.get(*node) {
            Union { options, .. } => !options.flattened_by_opt && options.slice.is_none(),
            _ => false,
        });

        let flattened = flatten.then(|| {
            let mut new_inputs = Vec::with_capacity(inputs.len() * 2);
            for node in inputs {
                match lp_arena.get(*node) {
                    IR::Union {
                        inputs, options, ..
                    } if options.slice.is_none() => new_inputs.extend_from_slice(inputs),
                    _ => new_inputs.push(*node),
                }
            }
            new_inputs
        });

        let groups = ScanGroups::find(
            flattened.as_deref().unwrap_or(inputs),
            options.maintain_order,
            lp_arena,
            expr_arena,
        );
        if flattened.is_none() && groups.is_none() {
            return Ok(None);
        }
        options.flattened_by_opt |= flatten;

        let inputs = flattened.unwrap_or_else(|| inputs.clone());
        let inputs = match groups {
            Some(groups) => {
                groups.merge(&inputs, options.maintain_order, options.rechunk, lp_arena)
            },
            None => inputs,
        };

        if let [input] = inputs.as_slice()
            && options.slice.is_none()
        {
            return Ok(Some(lp_arena.get(*input).clone()));
        }

        Ok(Some(Union { inputs, options }))
    }
}

/// A scan with the row-wise nodes on top of it, top first.
struct ScanChain {
    row_wise: UnitVec<Node>,
    scan: Node,
    /// Equal for chains that can be merged, so only chains with the same key are compared.
    key: u64,
}

/// Whether `node` gives the same rows when it runs once over the merged scan as when it runs on
/// each scan. Only elementwise expressions do. A sorted flag only holds for each scan, and a
/// literal with more than one value (e.g. the sets of `is_in`) lines up with each scan's rows.
fn is_row_wise(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    is_elementwise_rec(node, expr_arena)
        && !expr_arena.iter(node).any(|(_, ae)| match ae {
            AExpr::Function {
                function: IRFunctionExpr::SetSortedFlag(_),
                ..
            } => true,
            AExpr::Literal(lv) => !lv.is_scalar(),
            _ => false,
        })
}

/// Compares expressions by value.
struct ExprEq<'a>(&'a Arena<AExpr>);

impl ExpressionComparator for ExprEq<'_> {
    fn equals(&self, lhs: &ExprIR, rhs: &ExprIR) -> bool {
        lhs.output_name_inner() == rhs.output_name_inner()
            && self
                .0
                .get(lhs.node())
                .is_expr_equal_to(self.0.get(rhs.node()), self.0)
    }
}

fn hash_exprs(exprs: &[ExprIR], expr_arena: &Arena<AExpr>, state: &mut impl Hasher) {
    for e in exprs {
        e.output_name_inner().hash(state);
        for (_, ae) in expr_arena.iter(e.node()) {
            ae.hash(state);
        }
    }
}

impl ScanChain {
    fn new(mut node: Node, lp_arena: &Arena<IR>, expr_arena: &Arena<AExpr>) -> Option<Self> {
        let mut row_wise = UnitVec::new();
        let mut state = PlFixedStateQuality::default().build_hasher();
        loop {
            let ir = lp_arena.get(node);
            std::mem::discriminant(ir).hash(&mut state);
            let input = match ir {
                Scan {
                    scan_type,
                    predicate,
                    hive_parts,
                    unified_scan_args,
                    ..
                } => {
                    if hive_parts.is_some() || unified_scan_args.has_per_scan_state() {
                        return None;
                    }
                    scan_type.hash_concat_key(&mut state);
                    unified_scan_args.hash_concat_key(&mut state);
                    hash_exprs(predicate.as_slice(), expr_arena, &mut state);
                    return Some(Self {
                        row_wise,
                        scan: node,
                        key: state.finish(),
                    });
                },
                SimpleProjection { input, columns } => {
                    columns.len().hash(&mut state);
                    *input
                },
                Filter { input, predicate } => {
                    if !is_row_wise(predicate.node(), expr_arena) {
                        return None;
                    }
                    hash_exprs(std::slice::from_ref(predicate), expr_arena, &mut state);
                    *input
                },
                // A select of only scalars outputs one row, not one row per input row.
                Select { input, expr, .. } => {
                    if !expr.iter().all(|e| is_row_wise(e.node(), expr_arena))
                        || expr.iter().all(|e| is_scalar_ae(e.node(), expr_arena))
                    {
                        return None;
                    }
                    hash_exprs(expr, expr_arena, &mut state);
                    *input
                },
                HStack { input, exprs, .. } => {
                    if !exprs.iter().all(|e| is_row_wise(e.node(), expr_arena)) {
                        return None;
                    }
                    hash_exprs(exprs, expr_arena, &mut state);
                    *input
                },
                _ => return None,
            };
            row_wise.push(node);
            node = input;
        }
    }

    fn can_concat(&self, other: &Self, lp_arena: &Arena<IR>, expr_arena: &Arena<AExpr>) -> bool {
        self.key == other.key
            && self.row_wise.len() == other.row_wise.len()
            && self
                .row_wise
                .iter()
                .zip(other.row_wise.iter())
                .all(|(l, r)| {
                    let (l, r) = (lp_arena.get(*l), lp_arena.get(*r));
                    same_schema(l, r) && l.is_ir_equal_shallow(r, &ExprEq(expr_arena))
                })
            && can_concat_scans(
                lp_arena.get(self.scan),
                lp_arena.get(other.scan),
                expr_arena,
            )
    }
}

/// Union inputs grouped by whether they can be merged. Each group holds indices into the inputs.
struct ScanGroups {
    chains: Vec<Option<ScanChain>>,
    groups: Vec<UnitVec<usize>>,
}

impl ScanGroups {
    /// Groups inputs that are the same row-wise nodes over scans that can be read as one scan.
    /// An ordered union only groups neighbouring inputs. Returns `None` if no group has more than
    /// one input.
    fn find(
        inputs: &[Node],
        maintain_order: bool,
        lp_arena: &Arena<IR>,
        expr_arena: &Arena<AExpr>,
    ) -> Option<Self> {
        let chains: Vec<_> = inputs
            .iter()
            .map(|node| ScanChain::new(*node, lp_arena, expr_arena))
            .collect();
        if chains.iter().filter(|c| c.is_some()).count() < 2 {
            return None;
        }

        // An unordered union looks up groups by chain key.
        let mut groups: Vec<UnitVec<usize>> = Vec::with_capacity(inputs.len());
        let mut groups_by_key: PlIndexMap<u64, UnitVec<usize>> = PlIndexMap::default();
        for (i, chain) in chains.iter().enumerate() {
            let Some(chain) = chain else {
                groups.push(unitvec![i]);
                continue;
            };
            let fits = |g: usize| {
                chains[groups[g][0]]
                    .as_ref()
                    .is_some_and(|l| l.can_concat(chain, lp_arena, expr_arena))
            };
            let group = if maintain_order {
                groups.len().checked_sub(1).filter(|g| fits(*g))
            } else {
                groups_by_key
                    .get(&chain.key)
                    .and_then(|gs| gs.iter().copied().find(|g| fits(*g)))
            };
            match group {
                Some(g) => groups[g].push(i),
                None => {
                    if !maintain_order {
                        groups_by_key
                            .entry(chain.key)
                            .or_default()
                            .push(groups.len());
                    }
                    groups.push(unitvec![i]);
                },
            }
        }

        (groups.len() < inputs.len()).then_some(Self { chains, groups })
    }

    /// Replaces each group of `inputs` with one merged scan and the row-wise nodes of the group's
    /// first input on top of it.
    fn merge(
        self,
        inputs: &[Node],
        maintain_order: bool,
        rechunk: bool,
        lp_arena: &mut Arena<IR>,
    ) -> Vec<Node> {
        let Self { chains, groups } = self;
        groups
            .into_iter()
            .map(|group| {
                if let [i] = group.as_slice() {
                    return inputs[*i];
                }
                let scans: UnitVec<_> = group
                    .iter()
                    .map(|i| chains[*i].as_ref().unwrap().scan)
                    .collect();
                let scan = concat_scans(&scans, maintain_order, rechunk, lp_arena);
                let mut node = lp_arena.add(scan);
                for row_wise in chains[group[0]].as_ref().unwrap().row_wise.iter().rev() {
                    let ir = lp_arena.get(*row_wise).clone().with_inputs([node]);
                    node = lp_arena.add(ir);
                }
                node
            })
            .collect()
    }
}

fn same_schema(l: &IR, r: &IR) -> bool {
    match (l, r) {
        (Select { schema: l, .. }, Select { schema: r, .. })
        | (HStack { schema: l, .. }, HStack { schema: r, .. }) => l == r,
        _ => true,
    }
}

fn can_concat_scans(l: &IR, r: &IR, expr_arena: &Arena<AExpr>) -> bool {
    let (
        Scan {
            sources: l_sources,
            file_info: l_file_info,
            hive_parts: l_hive_parts,
            predicate: l_predicate,
            predicate_file_skip_applied: l_file_skip,
            output_schema: l_output_schema,
            scan_type: l_scan_type,
            unified_scan_args: l_args,
            maintain_order: _,
        },
        Scan {
            sources: r_sources,
            file_info: r_file_info,
            hive_parts: r_hive_parts,
            predicate: r_predicate,
            predicate_file_skip_applied: r_file_skip,
            output_schema: r_output_schema,
            scan_type: r_scan_type,
            unified_scan_args: r_args,
            maintain_order: _,
        },
    ) = (l, r)
    else {
        return false;
    };

    let same_source_kind = matches!(
        (l_sources, r_sources),
        (ScanSources::Paths(_), ScanSources::Paths(_))
            | (ScanSources::Buffers(_), ScanSources::Buffers(_))
    );
    let same_predicate = match (l_predicate, r_predicate) {
        (None, None) => true,
        (Some(l), Some(r)) => expr_arena
            .get(l.node())
            .is_expr_equal_to(expr_arena.get(r.node()), expr_arena),
        _ => false,
    };

    // Hive columns come from each scan's own paths.
    same_source_kind
        && l_hive_parts.is_none()
        && r_hive_parts.is_none()
        && l_scan_type.can_concat_sources(r_scan_type)
        && l_args.can_concat_sources(r_args)
        && l_file_info.schema == r_file_info.schema
        && l_file_info.reader_schema == r_file_info.reader_schema
        && l_output_schema == r_output_schema
        && l_file_skip.map(|s| s.no_residual_predicate)
            == r_file_skip.map(|s| s.no_residual_predicate)
        && same_predicate
}

/// One scan over the sources of `group`. All scans in `group` must pass `can_concat_scans`.
fn concat_scans(group: &[Node], maintain_order: bool, rechunk: bool, lp_arena: &Arena<IR>) -> IR {
    let mut sources = Vec::with_capacity(group.len());
    let mut stats = Vec::with_capacity(group.len());
    let mut scan_types = Vec::with_capacity(group.len());
    let mut args = Vec::with_capacity(group.len());
    let mut files_before_skip = 0;
    let mut all_maintain_order = maintain_order;
    for node in group {
        let Scan {
            sources: s,
            file_info,
            predicate_file_skip_applied,
            scan_type,
            unified_scan_args,
            maintain_order,
            ..
        } = lp_arena.get(*node)
        else {
            unreachable!()
        };
        sources.push(s);
        stats.push(&file_info.stats);
        scan_types.push((scan_type.as_ref(), s.len()));
        args.push(unified_scan_args.as_ref());
        files_before_skip += predicate_file_skip_applied.map_or(0, |s| s.original_len);
        all_maintain_order &= *maintain_order;
    }

    let Scan {
        file_info,
        predicate,
        predicate_file_skip_applied,
        output_schema,
        ..
    } = lp_arena.get(group[0])
    else {
        unreachable!()
    };

    let mut unified_scan_args = UnifiedScanArgs::concat_sources(&args);
    unified_scan_args.rechunk = rechunk;

    Scan {
        sources: ScanSources::concat(&sources).unwrap(),
        file_info: FileInfo::new(
            file_info.schema.clone(),
            file_info.reader_schema.clone(),
            ScanStats::concat(&stats),
        ),
        hive_parts: None,
        predicate: predicate.clone(),
        predicate_file_skip_applied: predicate_file_skip_applied.map(|s| PredicateFileSkip {
            no_residual_predicate: s.no_residual_predicate,
            original_len: files_before_skip,
        }),
        output_schema: output_schema.clone(),
        scan_type: Box::new(FileScanIR::concat_sources(&scan_types)),
        unified_scan_args: Box::new(unified_scan_args),
        maintain_order: all_maintain_order,
    }
}
