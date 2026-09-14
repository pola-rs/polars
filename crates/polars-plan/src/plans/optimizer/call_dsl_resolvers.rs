#[cfg(feature = "python")]
use std::cell::LazyCell;
use std::ops::ControlFlow;
use std::pin::Pin;
use std::sync::Arc;

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use polars_core::config::verbose;
use polars_core::error::PolarsResult;
use polars_core::runtime::ASYNC;
use polars_error::{polars_bail, polars_ensure};
use polars_utils::IdxSize;
use polars_utils::aliases::PlIndexSet;
use polars_utils::arena::{Arena, Node};
use polars_utils::async_utils::tokio_handle_ext::AbortOnDropHandle;
use polars_utils::index::idxsize_try_from;
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "python")]
use polars_utils::python_thread_pool::PyThreadPool;
use polars_utils::scratch_vec::ScratchVec;

use crate::dsl::dsl_resolver::{DslResolverTrait as _, ResolveDslArgs, ResolvedDsl};
use crate::plans::optimizer::ir_traversal::ir_graph_traversal;
use crate::plans::optimizer::predicate_pushdown::combine_predicates;
use crate::plans::{AExpr, IR, OptFlags, node_to_expr, optimize, to_alp};
use crate::traversal::visitor::{FnVisitors, SubtreeVisit};
use crate::utils::aexpr_to_leaf_names_iter;

pub(super) fn call_dsl_resolvers(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    opt_flags: OptFlags,
    apply_scan_predicate_to_scan_ir: fn(
        Node,
        &mut Arena<IR>,
        &mut Arena<AExpr>,
    ) -> PolarsResult<()>,
) -> PolarsResult<()> {
    let mut resolve_tasks: FuturesUnordered<Pin<Box<dyn Future<Output = _>>>> =
        FuturesUnordered::new();
    // We can have same args pointing to same cache. This means the same resolver was referenced
    // multiple times in the plan, but were not CSE'd, so we must split the caches.
    #[expect(clippy::disallowed_types)]
    let mut deshare_caches = polars_utils::aliases::PlHashSet::default();
    let mut deshared_cache_count: usize = 0;

    #[cfg(feature = "python")]
    let py_lazyframe_resolve_threadpool: LazyCell<Arc<PyThreadPool>> =
        LazyCell::new(|| Arc::new(PyThreadPool::new()));

    let expr_arena_ref = &*expr_arena;

    match ir_graph_traversal(
        root,
        &mut FnVisitors::new(
            || (),
            |key, storage: &mut Arena<IR>, _| {
                let expr_arena = expr_arena_ref;

                match (|| {
                    let IR::Resolver {
                        resolver,
                        resolver_schema: _,
                        projection,
                        slice,
                        filters,
                        filter_drop_columns_idx,
                        resolved_dsl,
                        resolved_ir,
                    } = storage.get_mut(key)
                    else {
                        return Ok(());
                    };

                    let filter_drop_columns_idx = *filter_drop_columns_idx;

                    let args = ResolveDslArgs {
                        projection: projection.clone(),
                        slice: *slice,
                        filters: filters
                            .iter()
                            .map(|eir| node_to_expr(eir.node(), expr_arena))
                            .collect(),
                        filter_columns: filters
                            .iter()
                            .flat_map(|eir| aexpr_to_leaf_names_iter(eir.node(), expr_arena))
                            .collect::<PlIndexSet<&PlSmallStr>>()
                            .into_iter()
                            .cloned()
                            .collect(),
                        filter_drop_columns_idx,
                    };

                    let resolved_dsl_guard = resolved_dsl.lock().unwrap();
                    let prev_resolved_dsl = resolved_dsl_guard.get(&args);

                    if let Some(
                        prev_resolved_dsl @ ResolvedDsl {
                            version_key: None, ..
                        },
                    ) = prev_resolved_dsl
                    {
                        if resolved_ir.is_none() {
                            resolve_tasks.push(Box::pin(std::future::ready((
                                key,
                                args,
                                Ok(prev_resolved_dsl.clone()),
                            ))))
                        }

                        return Ok(());
                    }

                    let mut existing_resolved_version_key: Option<PlSmallStr> =
                        prev_resolved_dsl.and_then(|x| x.version_key.clone());

                    drop(resolved_dsl_guard);

                    if !deshare_caches
                        .insert((Arc::as_ptr(resolved_dsl) as *const _ as usize, args.clone()))
                    {
                        deshared_cache_count += 1;
                        existing_resolved_version_key = None;
                        *resolved_dsl = Default::default();

                        deshare_caches
                            .insert((Arc::as_ptr(resolved_dsl) as *const _ as usize, args.clone()));
                    }

                    let resolver = Arc::clone(resolver);

                    let resolve_fut = resolver.resolve_dsl(
                        args.clone(),
                        filters.clone(),
                        existing_resolved_version_key,
                        expr_arena,
                        #[cfg(feature = "python")]
                        Arc::clone(&py_lazyframe_resolve_threadpool),
                    )?;

                    let resolve_fut = AbortOnDropHandle(ASYNC.spawn(resolve_fut));
                    let resolve_fut =
                        Box::pin(async move { (key, args, resolve_fut.await.unwrap()) });

                    resolve_tasks.push(resolve_fut);

                    PolarsResult::Ok(())
                })() {
                    Ok(()) => ControlFlow::Continue(SubtreeVisit::Visit),
                    Err(err) => ControlFlow::Break(err),
                }
            },
            |_, _, _| ControlFlow::Continue(()),
        ),
        &mut vec![],
        &mut vec![],
        ir_arena,
    ) {
        ControlFlow::Continue(()) => {},
        ControlFlow::Break(err) => return Err(err),
    }

    let verbose = verbose();

    if verbose && deshared_cache_count > 0 {
        eprintln!("call_dsl_resolvers: split shared memory caches (n = {deshared_cache_count})")
    }

    if !resolve_tasks.is_empty() {
        if verbose {
            let n = resolve_tasks.len();
            eprintln!("call_dsl_resolvers: n_resolvers: {n}")
        }

        ASYNC.block_in_place_on(async {
            let mut optimize_scratch = ScratchVec::default();

            while let Some((node, args, resolved)) = resolve_tasks.next().await {
                let resolved = resolved?;

                let IR::Resolver {
                    resolver: _,
                    resolver_schema,
                    projection,
                    slice,
                    filters,
                    filter_drop_columns_idx,
                    resolved_dsl,
                    resolved_ir,
                } = ir_arena.get(node)
                else {
                    unreachable!()
                };
                let resolver_schema = Arc::clone(resolver_schema);
                let projection = projection.clone();
                let slice = *slice;
                let filters = filters.clone();
                let filter_drop_columns_idx = *filter_drop_columns_idx;

                let ResolvedDsl {
                    dsl: Some(dsl),
                    version_key: _,
                    applied_filters,
                    slice_offset_applied,
                } = (if resolved.dsl.is_some() {
                    resolved_dsl
                        .lock()
                        .unwrap()
                        .insert(args.clone(), resolved.clone());
                    resolved
                } else {
                    let version_key = resolved.version_key.as_ref();
                    let guard = resolved_dsl.lock().unwrap();
                    let existing_resolved_dsl = guard.get(&args);
                    let existing_resolved_version_keys: PlIndexSet<PlSmallStr> = {
                        guard
                            .values()
                            .filter_map(|x| x.version_key.as_ref())
                            .chain(existing_resolved_dsl.and_then(|x| x.version_key.as_ref()))
                            .cloned()
                            .collect()
                    };

                    polars_ensure!(
                        existing_resolved_dsl
                        .as_ref()
                        .zip(version_key)
                        .is_some_and(|(l, r)| l.version_key.as_ref()  == Some(r)),
                        ComputeError:
                        "LazyFrame resolver returned None for the LazyFrame, \
                        and version key does not match or was not found: \
                        version_key: {version_key:?}, \
                        existing_resolved_version_keys: \
                        {existing_resolved_version_keys:?}
                        "
                    );

                    if resolved_ir.is_some() {
                        if verbose {
                            eprintln!("call_dsl_resolvers: use existing version: {version_key:?}");
                        }
                        continue;
                    };

                    existing_resolved_dsl.unwrap().clone()
                })
                else {
                    unreachable!()
                };

                if let Some(&i) = applied_filters.iter().find(|&&i| i >= filters.len()) {
                    let n_filters = filters.len();
                    polars_bail!(
                        ShapeMismatch:
                        "index (i = {i}) contained in `applied_filters` out of bounds \
                        for n_filters = {n_filters}"
                    )
                }

                let mut ir_node = {
                    let mut opt_flags = opt_flags;
                    to_alp(dsl, expr_arena, ir_arena, &mut opt_flags)?
                };

                let ir_node_schema = ir_arena.get(ir_node).schema(ir_arena).into_owned();

                // The resolver is only handed a `limit` (= offset + len), never the
                // offset, so it can never apply a non-zero offset itself. Predicate
                // pushdown therefore refuses to place filters on a resolver that already
                // carries a slice, which means we can always apply the slice here.
                debug_assert!(slice.is_none() || filters.is_empty());

                if let Some((mut offset, len)) = slice {
                    if slice_offset_applied {
                        offset = 0
                    }

                    ir_node = ir_arena.add(IR::Slice {
                        input: ir_node,
                        offset,
                        len: idxsize_try_from(len).unwrap_or(IdxSize::MAX),
                    });
                }

                if let Some(eir) = combine_predicates(
                    filters
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| !applied_filters.contains(i))
                        .map(|(_, x)| x)
                        .cloned(),
                    expr_arena,
                ) {
                    ir_node = ir_arena.add(IR::Filter {
                        input: ir_node,
                        predicate: eir,
                    });
                }

                if let Some(mut projection) = projection.as_deref() {
                    if let Some(filter_drop_columns_idx) = filter_drop_columns_idx {
                        projection = &projection[..filter_drop_columns_idx];
                    }

                    let schema = ir_arena
                        .get(ir_node)
                        .schema(ir_arena)
                        .try_project(projection)?;

                    ir_node = ir_arena.add(IR::SimpleProjection {
                        input: ir_node,
                        columns: Arc::new(schema),
                    });
                } else {
                    if let Some(name) = resolver_schema
                        .iter_names()
                        .find(|name| !ir_node_schema.contains(name))
                    {
                        polars_bail!(
                            SchemaMismatch:
                            "column name '{name}' declared in the schema of the resolver \
                            is missing from the resolved plan \
                            (resolver_schema: {resolver_schema:?})"
                        )
                    }

                    if let Some(name) = ir_node_schema
                        .iter_names()
                        .find(|name| !resolver_schema.contains(name))
                    {
                        polars_bail!(
                            SchemaMismatch:
                            "encountered column name '{name}' in resolved plan that \
                            was not declared in the schema of the resolver \
                            (resolver_schema: {resolver_schema:?})"
                        )
                    }
                }

                ir_node = optimize(
                    ir_node,
                    opt_flags,
                    ir_arena,
                    expr_arena,
                    optimize_scratch.get(),
                    apply_scan_predicate_to_scan_ir,
                )?;

                let IR::Resolver { resolved_ir, .. } = ir_arena.get_mut(node) else {
                    unreachable!()
                };

                *resolved_ir = Some(ir_node);
            }

            PolarsResult::Ok(())
        })?;
    }

    Ok(())
}
