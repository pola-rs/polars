use std::fmt::Write;
use std::sync::Arc;

use polars_arrow::bit_util::unset_bit_raw;
use polars_arrow::export::arrow::array::PrimitiveArray;
use polars_core::frame::groupby::{GroupBy, GroupsProxy};
use polars_core::frame::hash_join::{
    default_join_ids, private_left_join_multiple_keys, JoinOptIds,
};
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::utils::_split_offsets;
use polars_core::utils::arrow::bitmap::MutableBitmap;
use polars_core::{downcast_as_macro_arg_physical, POOL};
use polars_utils::sort::perfect_sort;
use polars_utils::sync::SyncPtr;
use rayon::prelude::*;

use super::*;
use crate::physical_plan::expression_err;
use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;

pub struct WindowExpr {
    /// the root column that the Function will be applied on.
    /// This will be used to create a smaller DataFrame to prevent taking unneeded columns by index
    pub(crate) group_by: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) apply_columns: Vec<Arc<str>>,
    pub(crate) out_name: Option<Arc<str>>,
    /// A function Expr. i.e. Mean, Median, Max, etc.
    pub(crate) function: Expr,
    pub(crate) phys_function: Arc<dyn PhysicalExpr>,
    pub(crate) options: WindowOptions,
    pub(crate) expr: Expr,
}

#[cfg_attr(debug_assertions, derive(Debug))]
enum MapStrategy {
    // Join by key, this the most expensive
    // for reduced aggregations
    Join,
    // explode now
    Explode,
    // will be exploded by subsequent `.flatten()` call
    ExplodeLater,
    // Use an arg_sort to map the values back
    Map,
    Nothing,
}

impl WindowExpr {
    fn map_list_agg_by_arg_sort(
        &self,
        out_column: Series,
        flattened: Series,
        mut ac: AggregationContext,
        gb: GroupBy,
        state: &ExecutionState,
        cache_key: &str,
    ) -> PolarsResult<Series> {
        // idx (new-idx, original-idx)
        let mut idx_mapping = Vec::with_capacity(out_column.len());

        // we already set this buffer so we can reuse the `original_idx` buffer
        // that saves an allocation
        let mut take_idx = vec![];

        // groups are not changed, we can map by doing a standard arg_sort.
        if std::ptr::eq(ac.groups().as_ref(), gb.get_groups()) {
            let mut iter = 0..flattened.len() as IdxSize;
            match ac.groups().as_ref() {
                GroupsProxy::Idx(groups) => {
                    for g in groups.all() {
                        idx_mapping.extend(g.iter().copied().zip(&mut iter));
                    }
                }
                GroupsProxy::Slice { groups, .. } => {
                    for &[first, len] in groups {
                        idx_mapping.extend((first..first + len).zip(&mut iter));
                    }
                }
            }
        }
        // groups are changed, we use the new group indexes as arguments of the arg_sort
        // and sort by the old indexes
        else {
            let mut original_idx = Vec::with_capacity(out_column.len());
            match gb.get_groups() {
                GroupsProxy::Idx(groups) => {
                    for g in groups.all() {
                        original_idx.extend_from_slice(g)
                    }
                }
                GroupsProxy::Slice { groups, .. } => {
                    for &[first, len] in groups {
                        original_idx.extend(first..first + len)
                    }
                }
            };

            let mut original_idx_iter = original_idx.iter().copied();

            match ac.groups().as_ref() {
                GroupsProxy::Idx(groups) => {
                    for g in groups.all() {
                        idx_mapping.extend(g.iter().copied().zip(&mut original_idx_iter));
                    }
                }
                GroupsProxy::Slice { groups, .. } => {
                    for &[first, len] in groups {
                        idx_mapping.extend((first..first + len).zip(&mut original_idx_iter));
                    }
                }
            }
            original_idx.clear();
            take_idx = original_idx;
        }
        cache_gb(gb, state, cache_key);
        // Safety:
        // we only have unique indices ranging from 0..len
        unsafe { perfect_sort(&POOL, &idx_mapping, &mut take_idx) };
        let idx = IdxCa::from_vec("", take_idx);

        // Safety:
        // groups should always be in bounds.
        unsafe { flattened.take_unchecked(&idx) }
    }

    #[allow(clippy::too_many_arguments)]
    fn map_by_arg_sort(
        &self,
        df: &DataFrame,
        out_column: Series,
        flattened: Series,
        mut ac: AggregationContext,
        groupby_columns: &[Series],
        gb: GroupBy,
        state: &ExecutionState,
        cache_key: &str,
    ) -> PolarsResult<Series> {
        // we use an arg_sort to map the values back

        // This is a bit more complicated because the final group tuples may differ from the original
        // so we use the original indices as idx values to arg_sort the original column
        //
        // The example below shows the naive version without group tuple mapping

        // columns
        // a b a a
        //
        // agg list
        // [0, 2, 3]
        // [1]
        //
        // flatten
        //
        // [0, 2, 3, 1]
        //
        // arg_sort
        //
        // [0, 3, 1, 2]
        //
        // take by arg_sorted indexes and voila groups mapped
        // [0, 1, 2, 3]

        if flattened.len() != df.height() {
            let ca = out_column.list().unwrap();
            let non_matching_group =
                ca.into_iter()
                    .zip(ac.groups().iter())
                    .find(|(output, group)| {
                        if let Some(output) = output {
                            output.as_ref().len() != group.len()
                        } else {
                            false
                        }
                    });

            return if let Some((output, group)) = non_matching_group {
                let first = group.first();
                let group = groupby_columns
                    .iter()
                    .map(|s| format!("{}", s.get(first as usize).unwrap()))
                    .collect::<Vec<_>>();
                let err_msg = format!(
                    "{}\n> Group: ",
                    "The length of the window expression did not match that of the group."
                );
                let err_msg = column_delimited(err_msg, &group);
                let err_msg = format!(
                    "{}\n> Group length: {}\n> Output: '{:?}'",
                    err_msg,
                    group.len(),
                    output.unwrap()
                );
                Err(expression_err!(err_msg, self.expr, ComputeError))
            } else {
                let msg = "The length of the window expression did not match that of the group.";
                Err(expression_err!(msg, self.expr, ComputeError))
            };
        }
        self.map_list_agg_by_arg_sort(out_column, flattened, ac, gb, state, cache_key)
    }

    fn run_aggregation<'a>(
        &self,
        df: &DataFrame,
        state: &ExecutionState,
        gb: &'a GroupBy,
    ) -> PolarsResult<AggregationContext<'a>> {
        let ac = self
            .phys_function
            .evaluate_on_groups(df, gb.get_groups(), state)?;
        Ok(ac)
    }

    fn is_explicit_list_agg(&self) -> bool {
        // col("foo").list()
        // col("foo").list().alias()
        // ..
        // col("foo").list().alias().alias()
        //
        // but not:
        // col("foo").list().sum().alias()
        // ..
        // col("foo").min()
        let mut explicit_list = false;
        for e in &self.expr {
            if let Expr::Window { function, .. } = e {
                // or list().alias
                let mut finishes_list = false;
                for e in &**function {
                    match e {
                        Expr::Agg(AggExpr::List(_)) => {
                            finishes_list = true;
                        }
                        Expr::Alias(_, _) => {}
                        _ => break,
                    }
                }
                explicit_list = finishes_list;
            }
        }

        explicit_list
    }

    fn is_simple_column_expr(&self) -> bool {
        // col()
        // or col().alias()
        let mut simple_col = false;
        for e in &self.expr {
            if let Expr::Window { function, .. } = e {
                // or list().alias
                for e in &**function {
                    match e {
                        Expr::Column(_) => {
                            simple_col = true;
                        }
                        Expr::Alias(_, _) => {}
                        _ => break,
                    }
                }
            }
        }
        simple_col
    }

    fn is_aggregation(&self) -> bool {
        // col()
        // or col().agg()
        let mut agg_col = false;
        for e in &self.expr {
            if let Expr::Window { function, .. } = e {
                // or list().alias
                for e in &**function {
                    match e {
                        Expr::Agg(_) => {
                            agg_col = true;
                        }
                        Expr::Alias(_, _) => {}
                        _ => break,
                    }
                }
            }
        }
        agg_col
    }

    /// check if the the branches have an aggregation
    /// when(a > sum)
    /// then (foo)
    /// otherwise(bar - sum)
    fn has_different_group_sources(&self) -> bool {
        let mut has_arity = false;
        let mut agg_col = false;
        for e in &self.expr {
            if let Expr::Window { function, .. } = e {
                // or list().alias
                for e in &**function {
                    match e {
                        Expr::Ternary { .. } | Expr::BinaryExpr { .. } => {
                            has_arity = true;
                        }
                        Expr::Alias(_, _) => {}
                        Expr::Agg(_) => {
                            agg_col = true;
                        }
                        Expr::Function { options, .. }
                        | Expr::AnonymousFunction { options, .. } => {
                            if options.auto_explode
                                && matches!(options.collect_groups, ApplyOptions::ApplyGroups)
                            {
                                agg_col = true;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        has_arity && agg_col
    }

    fn determine_map_strategy(
        &self,
        agg_state: &AggState,
        sorted_keys: bool,
        explicit_list: bool,
        gb: &GroupBy,
    ) -> PolarsResult<MapStrategy> {
        match (self.options.explode, explicit_list, agg_state) {
            // Explode
            // `(col("x").sum() * col("y")).list().over("groups").flatten()`
            (true, true, _) => Ok(MapStrategy::ExplodeLater),
            // Explode all the aggregated lists. Maybe add later?
            (true, false, _) => {
                let msg = "This operation is likely not what you want (you may need '.list()'). Please open an issue if you really want to do this";
                Err(expression_err!(msg, self.expr, ComputeError))
            }
            // explicit list
            // `(col("x").sum() * col("y")).list().over("groups")`
            (false, true, _) => Ok(MapStrategy::Join),
            // aggregations
            //`sum("foo").over("groups")`
            (false, false, AggState::AggregatedFlat(_)) => Ok(MapStrategy::Join),
            // no explicit aggregations, map over the groups
            //`(col("x").sum() * col("y")).over("groups")`
            (false, false, AggState::AggregatedList(_)) => {
                if sorted_keys {
                    if let GroupsProxy::Idx(g) = gb.get_groups() {
                        debug_assert!(g.is_sorted_flag())
                    }
                    // GroupsProxy::Slice is always sorted

                    // Note that group columns must be sorted for this to make sense!!!
                    Ok(MapStrategy::Explode)
                } else {
                    Ok(MapStrategy::Map)
                }
            }
            // no aggregations, just return column
            // or an aggregation that has been flattened
            // we have to check which one
            //`col("foo").over("groups")`
            (false, false, AggState::NotAggregated(_)) => {
                // col()
                // or col().alias()
                if self.is_simple_column_expr() {
                    Ok(MapStrategy::Nothing)
                } else {
                    Ok(MapStrategy::Map)
                }
            }
            // literals, do nothing and let broadcast
            (false, false, AggState::Literal(_)) => Ok(MapStrategy::Nothing),
        }
    }
}

impl PhysicalExpr for WindowExpr {
    // Note: this was first implemented with expression evaluation but this performed really bad.
    // Therefore we choose the groupby -> apply -> self join approach

    // This first cached the groupby and the join tuples, but rayon under a mutex leads to deadlocks:
    // https://github.com/rayon-rs/rayon/issues/592
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        // This method does the following:
        // 1. determine groupby tuples based on the group_column
        // 2. apply an aggregation function
        // 3. join the results back to the original dataframe
        //    this stores all group values on the original df size
        //
        //      we have several strategies for this
        //      - 3.1 JOIN
        //          Use a join for aggregations like
        //              `sum("foo").over("groups")`
        //          and explicit `list` aggregations
        //              `(col("x").sum() * col("y")).list().over("groups")`
        //
        //      - 3.2 EXPLODE
        //          Explicit list aggregations that are followed by `over().flatten()`
        //          # the fastest method to do things over groups when the groups are sorted.
        //          # note that it will require an explicit `list()` call from now on.
        //              `(col("x").sum() * col("y")).list().over("groups").flatten()`
        //
        //      - 3.3. MAP to original locations
        //          This will be done for list aggregations that are not explicitly aggregated as list
        //              `(col("x").sum() * col("y")).over("groups")
        //          This can be used to reverse, sort, shuffle etc. the values in a group

        // 4. select the final column and return
        let groupby_columns = self
            .group_by
            .iter()
            .map(|e| e.evaluate(df, state))
            .collect::<PolarsResult<Vec<_>>>()?;

        // if the keys are sorted
        let sorted_keys = groupby_columns.iter().all(|s| {
            matches!(
                s.is_sorted_flag(),
                IsSorted::Ascending | IsSorted::Descending
            )
        });
        let explicit_list_agg = self.is_explicit_list_agg();

        // A `sort()` in a window function is one level flatter
        // Assume we have column a : i32
        // than a sort in a groupby. A groupby sorts the groups and returns array: list[i32]
        // whereas a window function returns array: i32
        // So a `sort().list()` in a groupby returns: list[list[i32]]
        // whereas in a window function would return: list[i32]
        if explicit_list_agg {
            state.set_finalize_window_as_list();
        }

        // if we flatten this column we need to make sure the groups are sorted.
        let mut sort_groups = self.options.explode ||
            // if not
            //      `col().over()`
            // and not
            //      `col().list().over`
            // and not
            //      `col().sum()`
            // and keys are sorted
            //  we may optimize with explode call
            (!self.is_simple_column_expr() && !explicit_list_agg && sorted_keys && !self.is_aggregation());

        // overwrite sort_groups for some expressions
        // TODO: fully understand the rationale is here.
        if self.has_different_group_sources() {
            sort_groups = true
        }

        let create_groups = || {
            let gb = df.groupby_with_series(groupby_columns.clone(), true, sort_groups)?;
            let out: PolarsResult<GroupsProxy> = Ok(gb.take_groups());
            out
        };

        // Try to get cached grouptuples
        let (groups, _, cache_key) = if state.cache_window() {
            let mut cache_key = String::with_capacity(32 * groupby_columns.len());
            write!(&mut cache_key, "{}", state.branch_idx).unwrap();
            for s in &groupby_columns {
                cache_key.push_str(s.name());
            }

            let mut gt_map = state.group_tuples.lock().unwrap();
            // we run sequential and partitioned
            // and every partition run the cache should be empty so we expect a max of 1.
            debug_assert!(gt_map.len() <= 1);
            if let Some(gt) = gt_map.get_mut(&cache_key) {
                if df.height() > 0 {
                    assert!(!gt.is_empty());
                };
                if sort_groups {
                    gt.sort()
                }

                // We take now, but it is important that we set this before we return!
                // a next windows function may get this cached key and get an empty if this
                // does not happen
                (std::mem::take(gt), true, cache_key)
            } else {
                (create_groups()?, false, cache_key)
            }
        } else {
            (create_groups()?, false, "".to_string())
        };

        // 2. create GroupBy object and apply aggregation
        let apply_columns = self
            .apply_columns
            .iter()
            .map(|s| s.as_ref().to_string())
            .collect();
        let gb = GroupBy::new(df, groupby_columns.clone(), groups, Some(apply_columns));

        let mut ac = self.run_aggregation(df, state, &gb)?;

        use MapStrategy::*;
        match self.determine_map_strategy(ac.agg_state(), sorted_keys, explicit_list_agg, &gb)? {
            Nothing => {
                let mut out = ac.flat_naive().into_owned();
                cache_gb(gb, state, &cache_key);
                if let Some(name) = &self.out_name {
                    out.rename(name.as_ref());
                }
                Ok(out)
            }
            Explode => {
                let mut out = ac.aggregated().explode()?;
                cache_gb(gb, state, &cache_key);
                if let Some(name) = &self.out_name {
                    out.rename(name.as_ref());
                }
                Ok(out)
            }
            ExplodeLater => {
                let mut out = ac.aggregated();
                cache_gb(gb, state, &cache_key);
                if let Some(name) = &self.out_name {
                    out.rename(name.as_ref());
                }
                Ok(out)
            }
            Map => {
                // TODO!
                // investigate if sorted arrays can be return directly
                let out_column = ac.aggregated();
                let flattened = out_column.explode()?;
                // we extend the lifetime as we must convince the compiler that ac lives
                // long enough. We drop `GrouBy` when we are done with `ac`.
                let ac = unsafe {
                    std::mem::transmute::<AggregationContext<'_>, AggregationContext<'static>>(ac)
                };
                self.map_by_arg_sort(
                    df,
                    out_column,
                    flattened,
                    ac,
                    &groupby_columns,
                    gb,
                    state,
                    &cache_key,
                )
            }
            Join => {
                let out_column = ac.aggregated();
                // we try to flatten/extend the array by repeating the aggregated value n times
                // where n is the number of members in that group. That way we can try to reuse
                // the same map by arg_sort logic as done for listed aggregations
                match (
                    &ac.update_groups,
                    set_by_groups(&out_column, &ac.groups, df.height()),
                ) {
                    // for aggregations that reduce like sum, mean, first and are numeric
                    // we take the group locations to directly map them to the right place
                    (UpdateGroups::No, Some(out)) => {
                        cache_gb(gb, state, &cache_key);
                        Ok(out)
                    }
                    (_, _) => {
                        let keys = gb.keys();
                        cache_gb(gb, state, &cache_key);

                        let get_join_tuples = || {
                            if groupby_columns.len() == 1 {
                                // group key from right column
                                let right = &keys[0];
                                groupby_columns[0].hash_join_left(right).1
                            } else {
                                let df_right = DataFrame::new_no_checks(keys);
                                let df_left = DataFrame::new_no_checks(groupby_columns);
                                private_left_join_multiple_keys(&df_left, &df_right, None, None).1
                            }
                        };

                        // try to get cached join_tuples
                        let join_opt_ids = if state.cache_window() {
                            let mut jt_map = state.join_tuples.lock().unwrap();
                            // we run sequential and partitioned
                            // and every partition run the cache should be empty so we expect a max of 1.
                            debug_assert!(jt_map.len() <= 1);
                            if let Some(opt_join_tuples) = jt_map.get_mut(&cache_key) {
                                std::mem::replace(opt_join_tuples, default_join_ids())
                            } else {
                                get_join_tuples()
                            }
                        } else {
                            get_join_tuples()
                        };

                        let mut out = materialize_column(&join_opt_ids, &out_column);

                        if let Some(name) = &self.out_name {
                            out.rename(name.as_ref());
                        }

                        if state.cache_window() {
                            let mut jt_map = state.join_tuples.lock().unwrap();
                            jt_map.insert(cache_key, join_opt_ids);
                        }

                        Ok(out)
                    }
                }
            }
        }
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.function.to_field(input_schema, Context::Default)
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        _df: &DataFrame,
        _groups: &'a GroupsProxy,
        _state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        Err(PolarsError::InvalidOperation(
            "window expression not allowed in aggregation".into(),
        ))
    }

    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn is_valid_aggregation(&self) -> bool {
        false
    }
}

fn materialize_column(join_opt_ids: &JoinOptIds, out_column: &Series) -> Series {
    #[cfg(feature = "chunked_ids")]
    {
        use polars_arrow::export::arrow::Either;

        match join_opt_ids {
            Either::Left(ids) => unsafe {
                out_column.take_opt_iter_unchecked(
                    &mut ids.iter().map(|&opt_i| opt_i.map(|i| i as usize)),
                )
            },
            Either::Right(ids) => unsafe { out_column._take_opt_chunked_unchecked(ids) },
        }
    }

    #[cfg(not(feature = "chunked_ids"))]
    unsafe {
        out_column.take_opt_iter_unchecked(
            &mut join_opt_ids.iter().map(|&opt_i| opt_i.map(|i| i as usize)),
        )
    }
}

fn cache_gb(gb: GroupBy, state: &ExecutionState, cache_key: &str) {
    if state.cache_window() {
        let groups = gb.take_groups();
        let mut gt_map = state.group_tuples.lock().unwrap();
        gt_map.insert(cache_key.to_string(), groups);
    }
}

/// Simple reducing aggregation can be set by the groups
fn set_by_groups(s: &Series, groups: &GroupsProxy, len: usize) -> Option<Series> {
    if s.dtype().to_physical().is_numeric() {
        let dtype = s.dtype();
        let s = s.to_physical_repr();

        macro_rules! dispatch {
            ($ca:expr) => {{
                set_numeric($ca, groups, len)
            }};
        }
        downcast_as_macro_arg_physical!(&s, dispatch).map(|s| s.cast(dtype).unwrap())
    } else {
        None
    }
}

fn set_numeric<T>(ca: &ChunkedArray<T>, groups: &GroupsProxy, len: usize) -> Option<Series>
where
    T: PolarsNumericType,
    ChunkedArray<T>: IntoSeries,
{
    let mut idx_mapping = Vec::with_capacity(len);
    let mut iter = 0..len as IdxSize;
    match groups {
        GroupsProxy::Idx(groups) => {
            for g in groups.all() {
                idx_mapping.extend((&mut iter).take(g.len()).zip(g.iter().copied()));
            }
        }
        GroupsProxy::Slice { groups, .. } => {
            for &[first, len] in groups {
                idx_mapping.extend((&mut iter).take(len as usize).zip(first..first + len));
            }
        }
    }
    let mut values = Vec::with_capacity(len);
    let ptr = values.as_mut_ptr() as *mut T::Native;
    // safety:
    // we will write from different threads but we will never alias.
    let sync_ptr_values = unsafe { SyncPtr::new(ptr) };

    if ca.null_count() == 0 {
        match groups {
            GroupsProxy::Idx(groups) => {
                // this should always succeed as we don't expect any chunks after an aggregation
                let agg_vals = ca.cont_slice().ok()?;
                POOL.install(|| {
                    agg_vals
                        .par_iter()
                        .zip(groups.all().par_iter())
                        .for_each(|(v, g)| {
                            let ptr = sync_ptr_values.get();
                            for idx in g {
                                debug_assert!((*idx as usize) < len);
                                unsafe { *ptr.add(*idx as usize) = *v }
                            }
                        })
                })
            }
            GroupsProxy::Slice { groups, .. } => {
                // this should always succeed as we don't expect any chunks after an aggregation
                let agg_vals = ca.cont_slice().ok()?;
                POOL.install(|| {
                    agg_vals
                        .par_iter()
                        .zip(groups.par_iter())
                        .for_each(|(v, [start, g_len])| {
                            let ptr = sync_ptr_values.get();
                            let start = *start as usize;
                            let end = start + *g_len as usize;
                            for idx in start..end {
                                debug_assert!(idx < len);
                                unsafe { *ptr.add(idx) = *v }
                            }
                        })
                });
            }
        }

        // safety: we have written all slots
        unsafe { values.set_len(len) }
        Some(ChunkedArray::new_vec(ca.name(), values).into_series())
    } else {
        let mut validity = MutableBitmap::with_capacity(len);
        validity.extend_constant(len, true);
        let validity_ptr = validity.as_slice_mut().as_mut_ptr();
        let sync_ptr_validity = unsafe { SyncPtr::new(validity_ptr) };

        let n_threads = POOL.current_num_threads();
        let offsets = _split_offsets(ca.len(), n_threads);
        match groups {
            GroupsProxy::Idx(groups) => offsets.par_iter().for_each(|(offset, offset_len)| {
                let offset = *offset;
                let offset_len = *offset_len;
                let ca = ca.slice(offset as i64, offset_len);
                let groups = &groups.all()[offset..offset + offset_len];
                let values_ptr = sync_ptr_values.get();

                ca.into_iter().zip(groups.iter()).for_each(|(opt_v, g)| {
                    for idx in g {
                        let idx = *idx as usize;
                        debug_assert!(idx < len);
                        unsafe {
                            match opt_v {
                                Some(v) => {
                                    *values_ptr.add(idx) = v;
                                }
                                None => {
                                    *values_ptr.add(idx) = T::Native::default();
                                    unset_bit_raw(sync_ptr_validity.get(), idx)
                                }
                            };
                        }
                    }
                })
            }),
            GroupsProxy::Slice { groups, .. } => {
                offsets.par_iter().for_each(|(offset, offset_len)| {
                    let offset = *offset;
                    let offset_len = *offset_len;
                    let ca = ca.slice(offset as i64, offset_len);
                    let groups = &groups[offset..offset + offset_len];
                    let values_ptr = sync_ptr_values.get();

                    for (opt_v, [start, g_len]) in ca.into_iter().zip(groups.iter()) {
                        let start = *start as usize;
                        let end = start + *g_len as usize;
                        for idx in start..end {
                            debug_assert!(idx < len);
                            unsafe {
                                match opt_v {
                                    Some(v) => {
                                        *values_ptr.add(idx) = v;
                                    }
                                    None => {
                                        *values_ptr.add(idx) = T::Native::default();
                                        unset_bit_raw(sync_ptr_validity.get(), idx)
                                    }
                                };
                            }
                        }
                    }
                })
            }
        }
        // safety: we have written all slots
        unsafe { values.set_len(len) }
        let arr = PrimitiveArray::new(
            T::get_dtype().to_physical().to_arrow(),
            values.into(),
            Some(validity.into()),
        );
        Some(Series::try_from((ca.name(), arr.boxed())).unwrap())
    }
}
