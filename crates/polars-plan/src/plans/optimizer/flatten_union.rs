use std::cell::RefCell;
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

        let flattened = flatten.then(|| flatten_inputs(inputs, lp_arena));

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
        let merged = groups.is_some();
        let inputs = match groups {
            Some(groups) => {
                groups.merge(&inputs, options.maintain_order, options.rechunk, lp_arena)
            },
            None => inputs,
        };

        if let [input] = inputs.as_slice()
            && options.slice.is_none()
        {
            // A single input left after a merge is the new merged scan.
            return Ok(Some(if merged {
                lp_arena.take(*input)
            } else {
                lp_arena.get(*input).clone()
            }));
        }

        Ok(Some(Union { inputs, options }))
    }
}

/// The inputs of `inputs` with all nested unions without a slice replaced by their inputs.
fn flatten_inputs(inputs: &[Node], lp_arena: &Arena<IR>) -> Vec<Node> {
    let mut new_inputs = Vec::with_capacity(inputs.len() * 2);
    let mut stack = vec![inputs.iter()];
    while let Some(iter) = stack.last_mut() {
        let Some(node) = iter.next() else {
            stack.pop();
            continue;
        };
        match lp_arena.get(*node) {
            Union { inputs, options } if options.slice.is_none() => stack.push(inputs.iter()),
            _ => new_inputs.push(*node),
        }
    }
    new_inputs
}

/// The scan under the row-wise nodes on top of `node`, if that scan can be merged with others.
/// Doesn't look at the expressions of the nodes on top.
fn mergeable_scan(mut node: Node, lp_arena: &Arena<IR>) -> Option<Node> {
    loop {
        match lp_arena.get(node) {
            // Hive columns come from each scan's own paths.
            Scan {
                sources: ScanSources::Paths(_) | ScanSources::Buffers(_),
                scan_type,
                hive_parts: None,
                unified_scan_args,
                ..
            } => {
                return (scan_type.supports_concat_sources()
                    && !unified_scan_args.has_per_scan_state())
                .then_some(node);
            },
            SimpleProjection { input, .. }
            | Filter { input, .. }
            | Select { input, .. }
            | HStack { input, .. } => node = *input,
            _ => return None,
        }
    }
}

/// A scan with the row-wise nodes on top of it, top first.
struct ScanChain {
    row_wise: UnitVec<Node>,
    scan: Node,
    /// Equal for chains that can be merged, so only chains with the same key are compared.
    key: u64,
}

/// Hashes `exprs`, and returns whether they give the same rows when they run once over the
/// merged scan as when they run on each scan. Only elementwise expressions do. A sorted flag
/// only holds for each scan, and a literal with more than one value (e.g. the sets of `is_in`)
/// lines up with each scan's rows.
fn hash_row_wise_exprs(
    exprs: &[ExprIR],
    expr_arena: &Arena<AExpr>,
    state: &mut impl Hasher,
) -> bool {
    exprs.iter().all(|e| {
        e.output_name_inner().hash(state);
        is_elementwise_rec(e.node(), expr_arena)
            && expr_arena.iter(e.node()).all(|(_, ae)| {
                ae.hash(state);
                !matches!(
                    ae,
                    AExpr::Function {
                        function: IRFunctionExpr::SetSortedFlag(_),
                        ..
                    }
                ) && !matches!(ae, AExpr::Literal(lv) if !lv.is_scalar())
            })
    })
}

fn hash_exprs(exprs: &[ExprIR], expr_arena: &Arena<AExpr>, state: &mut impl Hasher) {
    for e in exprs {
        e.output_name_inner().hash(state);
        for (_, ae) in expr_arena.iter(e.node()) {
            ae.hash(state);
        }
    }
}

/// Compares expressions by value.
struct ExprEq<'a> {
    expr_arena: &'a Arena<AExpr>,
    stacks: RefCell<(Vec<Node>, Vec<Node>)>,
}

impl<'a> ExprEq<'a> {
    fn new(expr_arena: &'a Arena<AExpr>) -> Self {
        Self {
            expr_arena,
            stacks: RefCell::default(),
        }
    }

    fn nodes_equal(&self, l: Node, r: Node) -> bool {
        let (l_stack, r_stack) = &mut *self.stacks.borrow_mut();
        self.expr_arena.get(l).is_expr_equal_to_amortized(
            self.expr_arena.get(r),
            self.expr_arena,
            l_stack,
            r_stack,
        )
    }
}

impl ExpressionComparator for ExprEq<'_> {
    fn equals(&self, lhs: &ExprIR, rhs: &ExprIR) -> bool {
        lhs.output_name_inner() == rhs.output_name_inner()
            && self.nodes_equal(lhs.node(), rhs.node())
    }
}

impl ScanChain {
    fn new(mut node: Node, lp_arena: &Arena<IR>, expr_arena: &Arena<AExpr>) -> Option<Self> {
        let scan = mergeable_scan(node, lp_arena)?;
        let mut row_wise = UnitVec::new();
        let mut state = PlFixedStateQuality::default().build_hasher();
        while node != scan {
            let ir = lp_arena.get(node);
            std::mem::discriminant(ir).hash(&mut state);
            let input = match ir {
                SimpleProjection { input, columns } => {
                    columns.len().hash(&mut state);
                    *input
                },
                Filter { input, predicate } => {
                    if !hash_row_wise_exprs(std::slice::from_ref(predicate), expr_arena, &mut state)
                    {
                        return None;
                    }
                    *input
                },
                // A select of only scalars outputs one row, not one row per input row.
                Select { input, expr, .. } => {
                    if !hash_row_wise_exprs(expr, expr_arena, &mut state)
                        || expr.iter().all(|e| is_scalar_ae(e.node(), expr_arena))
                    {
                        return None;
                    }
                    *input
                },
                HStack { input, exprs, .. } => {
                    if !hash_row_wise_exprs(exprs, expr_arena, &mut state) {
                        return None;
                    }
                    *input
                },
                _ => unreachable!(),
            };
            row_wise.push(node);
            node = input;
        }

        let Scan {
            scan_type,
            predicate,
            unified_scan_args,
            ..
        } = lp_arena.get(scan)
        else {
            unreachable!()
        };
        scan_type.hash_concat_key(&mut state);
        unified_scan_args.hash_concat_key(&mut state);
        hash_exprs(predicate.as_slice(), expr_arena, &mut state);
        Some(Self {
            row_wise,
            scan,
            key: state.finish(),
        })
    }

    fn can_concat(&self, other: &Self, lp_arena: &Arena<IR>, expr_eq: &ExprEq) -> bool {
        self.key == other.key
            && self.row_wise.len() == other.row_wise.len()
            && self
                .row_wise
                .iter()
                .zip(other.row_wise.iter())
                .all(|(l, r)| {
                    let (l, r) = (lp_arena.get(*l), lp_arena.get(*r));
                    same_schema(l, r) && l.is_ir_equal_shallow(r, expr_eq)
                })
            && can_concat_scans(lp_arena.get(self.scan), lp_arena.get(other.scan), expr_eq)
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
        let n_mergeable = inputs
            .iter()
            .filter(|node| mergeable_scan(**node, lp_arena).is_some())
            .take(2)
            .count();
        if n_mergeable < 2 {
            return None;
        }

        let chains: Vec<_> = inputs
            .iter()
            .map(|node| ScanChain::new(*node, lp_arena, expr_arena))
            .collect();
        let expr_eq = ExprEq::new(expr_arena);

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
                    .is_some_and(|l| l.can_concat(chain, lp_arena, &expr_eq))
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

/// Whether two scans from `mergeable_scan` can be read as one scan.
fn can_concat_scans(l: &IR, r: &IR, expr_eq: &ExprEq) -> bool {
    let (
        Scan {
            sources: l_sources,
            file_info: l_file_info,
            hive_parts: _,
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
            hive_parts: _,
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

    std::mem::discriminant(l_sources) == std::mem::discriminant(r_sources)
        && l_scan_type.can_concat_sources(r_scan_type)
        && l_args.can_concat_sources(r_args)
        && l_file_info.schema == r_file_info.schema
        && l_file_info.reader_schema == r_file_info.reader_schema
        && l_output_schema == r_output_schema
        && l_file_skip.map(|s| s.no_residual_predicate)
            == r_file_skip.map(|s| s.no_residual_predicate)
        && match (l_predicate, r_predicate) {
            (None, None) => true,
            (Some(l), Some(r)) => expr_eq.nodes_equal(l.node(), r.node()),
            _ => false,
        }
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
