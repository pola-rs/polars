//! Moves a subquery that reads a relation of a query further out than the query
//! directly around it into the FROM of that query, which then reads the outer
//! relation itself. The decorrelation in `subquery.rs` handles that query. A shape
//! that isn't moved keeps its deeper references.
//!
//! The query `q` below and its subquery `s` read the outer relation `o`:
//!
//! ```sql
//! SELECT .. FROM f WHERE f.x IN (SELECT g.y FROM g WHERE g.c = o.k AND <rest>)
//! ```
//!
//! For a fixed row of `o`, `q` keeps the rows of `f` that have a match in `s`.
//! Those are the rows of `f` joined with the distinct `(g.y, g.c)` pairs of
//! `s` without the outer predicate, where the pair also matches `o.k`:
//!
//! ```sql
//! SELECT .. FROM f, (SELECT DISTINCT g.y AS v, g.c AS k0 FROM g WHERE <rest>) AS n
//! WHERE f.x = n.v AND n.k0 = o.k
//! ```
//!
//! Each row of `f` matches at most one distinct pair, so `q` returns the same rows.
//! The derived table is planned with `strict_relations`, so that it can only read
//! the relations it declares itself.
use std::ops::ControlFlow;

use polars_core::prelude::*;
use polars_core::utils::get_supertype;
use polars_lazy::prelude::*;
use sqlparser::ast::{
    BinaryOperator as SQLBinaryOperator, Distinct, ExcludeSelectItem, Expr as SQLExpr, Function,
    FunctionArgumentList, FunctionArguments, GroupByExpr, Ident, JoinConstraint, JoinOperator,
    ObjectName, OrderByExpr, OrderByKind, Query, Select, SelectItem, SetExpr, TableAlias,
    TableFactor, TableWithJoins, UnaryOperator as SQLUnaryOperator, Value as SQLValue, Visit,
    Visitor, VisitorMut, WindowSpec, WindowType,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::context::{TableInfo, combine_conditions, flatten_and_conditions};
use crate::dyn_visit::{visit_expressions, visit_expressions_mut};
use crate::functions::PolarsSQLFunctions;
use crate::sql_expr::{parse_sql_expr, to_sql_interface_err};
use crate::sql_visitors::{
    expr_contains_scalar_subquery, expr_contains_subquery, expr_has_window_functions,
    expr_references_any_column, is_subquery_expr,
};
use crate::subquery::{
    eligible_query_block, eligible_subquery_body, exists_aggregates, first_row_limit, strip_nested,
};
use crate::{SQLContext, dyn_visit, unique_column_name};

/// The alias prefix of the derived tables that hold a moved subquery.
pub(crate) const NESTED_PREFIX: &str = "__POLARS_NESTED_";

/// How the query around a subquery reads the rows of the subquery.
#[derive(Clone, Copy)]
pub(crate) enum ReadAs {
    /// Only which rows exist counts, as for EXISTS and IN.
    Set,
    /// The value of its row, as for a scalar subquery.
    Value,
    /// The rows are grouped afterwards.
    Groups,
}

/// The columns of the frame a subquery is decorrelated against.
#[derive(Clone, Copy)]
pub(crate) enum ScopeColumns<'a> {
    /// A SELECT without FROM, which declares nothing.
    None,
    Frame(&'a Schema),
    /// The two inputs of a join, for its ON condition.
    Join(&'a TableInfo, &'a TableInfo),
}

/// A query that a column of a subquery may read: its context resolves the qualifiers.
#[derive(Clone, Copy)]
struct Scope<'a> {
    ctx: &'a SQLContext,
    columns: ScopeColumns<'a>,
}

/// What a qualified column names in a scope.
enum Found {
    Column(DataType),
    /// A field of a struct column.
    Field,
    /// A column the named relation doesn't have.
    Missing,
}

/// The scope a column binds to.
enum Binding {
    /// The subquery itself.
    Local,
    /// A query around the subquery, with the dtype of a plain column.
    Foreign {
        depth: usize,
        dtype: Option<DataType>,
    },
}

/// A conjunct `inner = foreign` of the subquery: `inner` reads the subquery alone and
/// `foreign` is a column of a query around it.
struct CorrelationKey {
    inner: SQLExpr,
    foreign: SQLExpr,
    dtype: DataType,
}

/// A subquery moved into the FROM of the query around it.
struct Moved {
    table: TableWithJoins,
    columns: Vec<String>,
    replaced: Vec<SQLExpr>,
}

/// The operand compared with the value of a subquery, how, and whether the subquery is on
/// the left.
type Compare<'a> = (&'a SQLExpr, SQLBinaryOperator, bool);

impl SQLContext {
    /// `subquery`, about to be decorrelated against `columns` of this context, with the
    /// subqueries of its WHERE that read this context or further out moved into its FROM.
    /// `None` if nothing moved.
    pub(crate) fn pull_up_nested(
        &self,
        columns: ScopeColumns<'_>,
        subquery: &Query,
        read_as: ReadAs,
    ) -> Option<Query> {
        // An error while preparing leaves the subquery unchanged.
        prepare(&[Scope { ctx: self, columns }], subquery, read_as).unwrap_or(None)
    }

    /// `expr` with `pull_up_nested` applied to each of its subqueries, or `None` if nothing
    /// moved.
    pub(crate) fn pull_up_nested_in(
        &self,
        columns: ScopeColumns<'_>,
        expr: &SQLExpr,
    ) -> Option<SQLExpr> {
        struct Prepare<'a, 'b> {
            ctx: &'a SQLContext,
            columns: ScopeColumns<'b>,
            query_depth: usize,
            prepared: Vec<Option<Query>>,
        }

        impl Visitor for Prepare<'_, '_> {
            type Break = ();

            fn pre_visit_query(&mut self, _query: &Query) -> ControlFlow<()> {
                self.query_depth += 1;
                ControlFlow::Continue(())
            }

            fn post_visit_query(&mut self, _query: &Query) -> ControlFlow<()> {
                self.query_depth -= 1;
                ControlFlow::Continue(())
            }

            fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<()> {
                if self.query_depth == 0
                    && let Some(subquery) = expr_subquery(expr)
                {
                    let read_as = match expr {
                        SQLExpr::Subquery(_) => ReadAs::Value,
                        _ => ReadAs::Set,
                    };
                    let prepared = self.ctx.pull_up_nested(self.columns, subquery, read_as);
                    self.prepared.push(prepared);
                }
                ControlFlow::Continue(())
            }
        }

        struct Replace {
            query_depth: usize,
            prepared: std::vec::IntoIter<Option<Query>>,
        }

        impl VisitorMut for Replace {
            type Break = ();

            fn pre_visit_query(&mut self, _query: &mut Query) -> ControlFlow<()> {
                self.query_depth += 1;
                ControlFlow::Continue(())
            }

            fn post_visit_query(&mut self, _query: &mut Query) -> ControlFlow<()> {
                self.query_depth -= 1;
                ControlFlow::Continue(())
            }

            fn pre_visit_expr(&mut self, expr: &mut SQLExpr) -> ControlFlow<()> {
                if self.query_depth == 0 && is_subquery_expr(expr) {
                    let prepared = self.prepared.next().flatten();
                    if let (Some(prepared), Some(subquery)) = (prepared, expr_subquery_mut(expr)) {
                        *subquery = prepared;
                    }
                }
                ControlFlow::Continue(())
            }
        }

        if !expr_contains_subquery(expr) {
            return None;
        }
        let mut prepare = Prepare {
            ctx: self,
            columns,
            query_depth: 0,
            prepared: Vec::new(),
        };
        let _ = dyn_visit::visit(expr, &mut prepare);
        if prepare.prepared.iter().all(Option::is_none) {
            return None;
        }
        let mut expr = expr.clone();
        let _ = dyn_visit::visit_mut(
            &mut expr,
            &mut Replace {
                query_depth: 0,
                prepared: prepare.prepared.into_iter(),
            },
        );
        Some(expr)
    }
}

fn expr_subquery(expr: &SQLExpr) -> Option<&Query> {
    match expr {
        SQLExpr::Subquery(subquery)
        | SQLExpr::Exists { subquery, .. }
        | SQLExpr::InSubquery { subquery, .. } => Some(subquery),
        _ => None,
    }
}

fn expr_subquery_mut(expr: &mut SQLExpr) -> Option<&mut Query> {
    match expr {
        SQLExpr::Subquery(subquery)
        | SQLExpr::Exists { subquery, .. }
        | SQLExpr::InSubquery { subquery, .. } => Some(subquery),
        _ => None,
    }
}

/// `x` with the subqueries of its WHERE that read one of `scopes` (the queries around
/// `x`, outermost first) moved into its FROM, or `None` if nothing moved. Inner queries
/// are handled first. The conjuncts of an inner join condition filter the joined rows as
/// those of the WHERE do, and their subqueries move as well.
fn prepare(scopes: &[Scope<'_>], x: &Query, read_as: ReadAs) -> PolarsResult<Option<Query>> {
    let Some(select) = eligible_query_block(x) else {
        return Ok(None);
    };
    let (mut split_from, on_conjuncts) = match split_join_candidates(&select.from) {
        Some((from, conjuncts)) => (Some(from), conjuncts),
        None => (None, Vec::new()),
    };
    let inlined = select.selection.as_ref().and_then(inline_from_less);
    let unchanged = || {
        inlined
            .clone()
            .map(|selection| with_selection(x, select, Some(selection)))
    };
    let conjuncts = inlined
        .as_ref()
        .or(select.selection.as_ref())
        .map(flatten_and_conditions)
        .unwrap_or_default();
    if on_conjuncts.is_empty()
        && !conjuncts
            .iter()
            .any(|conj| candidate(conj).is_some() || expr_contains_scalar_subquery(conj))
    {
        return Ok(unchanged());
    }
    // The added columns are excluded from `SELECT *`, which can't be combined with
    // these.
    if select.projection.iter().any(|item| {
        matches!(item, SelectItem::Wildcard(options) if options.opt_except.is_some()
            || options.opt_ilike.is_some()
            || options.opt_replace.is_some()
            || options.opt_rename.is_some())
    }) {
        return Ok(unchanged());
    }
    // The FROM of `x` is resolved as it is when `x` is planned. Only what moves into the
    // derived tables is planned strictly, and so is everything that reads it here. A FROM
    // without named relations lets a qualifier name any table outside strict planning.
    let mut x_ctx = scopes.last().unwrap().ctx.isolated();
    let x_frame = match select.from.is_empty() {
        true => None,
        false => {
            let resolved = match &split_from {
                None => x_ctx.resolve_subquery_from(select)?,
                Some(from) => {
                    let mut stripped = select.clone();
                    stripped.from = from.clone();
                    x_ctx.resolve_subquery_from(&stripped)?
                },
            };
            let Some((names, lf, schema)) = resolved else {
                return Ok(unchanged());
            };
            if names.is_empty() {
                return Ok(unchanged());
            }
            Some((lf, schema))
        },
    };
    x_ctx.strict_relations = true;

    let repeats_harmless = repeats_harmless(x, select, read_as);
    let mut kept = Vec::new();
    let mut moved = Vec::new();
    let conjuncts = conjuncts
        .into_iter()
        .map(|conj| (conj, false))
        .chain(on_conjuncts.iter().map(|conj| (conj, true)));
    for (conj, in_join) in conjuncts {
        let Some((compare, y)) = candidate(conj) else {
            kept.push(conj.clone());
            continue;
        };
        let x_frame = x_frame.as_ref().map(|(lf, schema)| (lf, schema.as_ref()));
        let y_read_as = match strip_nested(conj) {
            SQLExpr::Exists { .. } | SQLExpr::InSubquery { .. } => ReadAs::Set,
            _ => ReadAs::Value,
        };
        let prepared_y = {
            let mut inner = scopes.to_vec();
            inner.push(Scope {
                ctx: &x_ctx,
                columns: frame_columns(x_frame),
            });
            prepare(&inner, y, y_read_as).unwrap_or(None)
        };
        let y = prepared_y.as_ref().unwrap_or(y);
        // A filter can match a row of `x` with several values of a scalar `y`, which must
        // instead raise.
        let filters_allowed = repeats_harmless && matches!(y_read_as, ReadAs::Set);
        match try_move(scopes, &mut x_ctx, x_frame, compare, y, filters_allowed).unwrap_or(None) {
            Some(m) => {
                kept.extend(m.replaced);
                moved.push((m.table, m.columns));
            },
            None => match (in_join, &mut split_from) {
                (true, Some(from)) => restore_join_conjunct(from, conj.clone()),
                _ => kept.push(conj.clone()),
            },
        }
    }
    // A conjunct that reads both `x` and a query around it is decorrelated as a whole,
    // which a subquery in it prevents. An uncorrelated scalar subquery in it is read from
    // a derived table of one row instead.
    let x_frame = x_frame.as_ref().map(|(lf, schema)| (lf, schema.as_ref()));
    let x_chain: Vec<Scope> = std::iter::once(Scope {
        ctx: &x_ctx,
        columns: frame_columns(x_frame),
    })
    .chain(scopes.iter().rev().copied())
    .collect();
    for conj in &mut kept {
        if !expr_contains_scalar_subquery(conj) || !reads_local_and_foreign(&x_chain, conj)? {
            continue;
        }
        if let Some((hoisted, tables)) = hoist_uncorrelated(&x_ctx, conj) {
            *conj = hoisted;
            moved.extend(tables);
        }
    }
    if moved.is_empty() {
        return Ok(unchanged());
    }

    let mut select = select.clone();
    // The added columns must not show up in `SELECT *`.
    let added: Vec<ObjectName> = moved
        .iter()
        .flat_map(|(_, columns)| columns)
        .map(|name| ObjectName::from(vec![Ident::new(name)]))
        .collect();
    for item in &mut select.projection {
        if let SelectItem::Wildcard(options) = item {
            let mut excluded = match options.opt_exclude.take() {
                None => Vec::new(),
                Some(ExcludeSelectItem::Single(name)) => vec![name],
                Some(ExcludeSelectItem::Multiple(names)) => names,
            };
            excluded.extend(added.iter().cloned());
            options.opt_exclude = Some(ExcludeSelectItem::Multiple(excluded));
        }
    }
    if let Some(from) = split_from {
        select.from = from;
    }
    select
        .from
        .extend(moved.into_iter().map(|(table, _)| table));
    Ok(Some(with_selection(
        x,
        &select,
        combine_conditions(kept, SQLBinaryOperator::And),
    )))
}

/// `conj` with each of its scalar subqueries read from a derived table of one row that
/// holds its value, and those tables with their column. `None` if a subquery reads a
/// relation it doesn't declare, which `x_ctx` (strict) rejects.
#[allow(clippy::type_complexity)]
fn hoist_uncorrelated(
    x_ctx: &SQLContext,
    conj: &SQLExpr,
) -> Option<(SQLExpr, Vec<(TableWithJoins, Vec<String>)>)> {
    struct Hoist<'a> {
        x_ctx: &'a SQLContext,
        query_depth: usize,
        tables: Vec<(TableWithJoins, Vec<String>)>,
    }

    impl VisitorMut for Hoist<'_> {
        type Break = ();

        fn pre_visit_query(&mut self, _query: &mut Query) -> ControlFlow<()> {
            self.query_depth += 1;
            ControlFlow::Continue(())
        }

        fn post_visit_query(&mut self, _query: &mut Query) -> ControlFlow<()> {
            self.query_depth -= 1;
            ControlFlow::Continue(())
        }

        fn pre_visit_expr(&mut self, expr: &mut SQLExpr) -> ControlFlow<()> {
            if self.query_depth > 0 {
                return ControlFlow::Continue(());
            }
            let SQLExpr::Subquery(subquery) = expr else {
                return ControlFlow::Continue(());
            };
            let name = format!("{NESTED_PREFIX}{}", unique_column_name());
            let value_name = format!("{name}_v");
            let Ok(query) = select_value(SQLExpr::Subquery(subquery.clone()), &value_name) else {
                return ControlFlow::Break(());
            };
            let mut ctx = self.x_ctx.isolated();
            let planned = ctx
                .execute_query(&query)
                .and_then(|mut lf| ctx.get_frame_schema(&mut lf));
            if planned.is_err() {
                return ControlFlow::Break(());
            }
            *expr = SQLExpr::CompoundIdentifier(vec![Ident::new(&name), Ident::new(&value_name)]);
            self.tables
                .push((derived_table(query, &name), vec![value_name]));
            ControlFlow::Continue(())
        }
    }

    let mut conj = conj.clone();
    let mut hoist = Hoist {
        x_ctx,
        query_depth: 0,
        tables: Vec::new(),
    };
    if dyn_visit::visit_mut(&mut conj, &mut hoist).is_break() {
        return None;
    }
    Some((conj, hoist.tables))
}

/// `SELECT <value> AS <alias>`.
fn select_value(value: SQLExpr, alias: &str) -> PolarsResult<Query> {
    let mut query = Parser::new(&GenericDialect)
        .try_with_sql("SELECT NULL")
        .and_then(|mut parser| parser.parse_query())
        .map_err(to_sql_interface_err)?;
    let SetExpr::Select(select) = query.body.as_mut() else {
        unreachable!()
    };
    select.projection = vec![SelectItem::ExprWithAlias {
        expr: value,
        alias: Ident::new(alias),
    }];
    Ok(*query)
}

/// `query` as a derived table named `name`.
fn derived_table(query: Query, name: &str) -> TableWithJoins {
    TableWithJoins {
        relation: TableFactor::Derived {
            lateral: false,
            subquery: Box::new(query),
            alias: Some(TableAlias {
                explicit: true,
                name: Ident::new(name),
                columns: vec![],
                at: None,
            }),
            sample: None,
        },
        joins: vec![],
    }
}

/// `from` without the conjuncts of the condition of its last join that hold a subquery,
/// and those conjuncts, or `None` if there are none. Only a single FROM item whose joins
/// are all inner gives them up: they then filter the rows as WHERE does, and read the
/// same relations.
#[allow(clippy::type_complexity)]
fn split_join_candidates(from: &[TableWithJoins]) -> Option<(Vec<TableWithJoins>, Vec<SQLExpr>)> {
    let [item] = from else {
        return None;
    };
    let all_inner = item.joins.iter().all(|join| {
        matches!(
            join.join_operator,
            JoinOperator::Join(_) | JoinOperator::Inner(_) | JoinOperator::CrossJoin(_)
        )
    });
    let (JoinOperator::Join(JoinConstraint::On(on)) | JoinOperator::Inner(JoinConstraint::On(on))) =
        &item.joins.last()?.join_operator
    else {
        return None;
    };
    if !all_inner {
        return None;
    }
    let (moving, rest): (Vec<&SQLExpr>, Vec<&SQLExpr>) = flatten_and_conditions(on)
        .into_iter()
        .partition(|conj| candidate(conj).is_some());
    if moving.is_empty() {
        return None;
    }
    let moving = moving.into_iter().cloned().collect();
    let rest = combine_conditions(rest.into_iter().cloned().collect(), SQLBinaryOperator::And);
    let mut from = from.to_vec();
    from[0].joins.last_mut().unwrap().join_operator = match rest {
        Some(on) => JoinOperator::Inner(JoinConstraint::On(on)),
        None => JoinOperator::CrossJoin(JoinConstraint::None),
    };
    Some((from, moving))
}

/// Add `conj` back to the condition of the last join of `from`.
fn restore_join_conjunct(from: &mut [TableWithJoins], conj: SQLExpr) {
    let join = from[0].joins.last_mut().unwrap();
    let on = match std::mem::replace(
        &mut join.join_operator,
        JoinOperator::CrossJoin(JoinConstraint::None),
    ) {
        JoinOperator::Inner(JoinConstraint::On(on)) => SQLExpr::BinaryOp {
            left: Box::new(on),
            op: SQLBinaryOperator::And,
            right: Box::new(conj),
        },
        _ => conj,
    };
    join.join_operator = JoinOperator::Inner(JoinConstraint::On(on));
}

/// `x`, whose body is `select`, with `selection` as the WHERE of `select`.
fn with_selection(x: &Query, select: &Select, selection: Option<SQLExpr>) -> Query {
    let mut select = select.clone();
    select.selection = selection;
    let mut x = x.clone();
    x.body = Box::new(SetExpr::Select(Box::new(select)));
    x
}

/// `selection` with each scalar subquery that selects one expression without FROM
/// replaced by that expression, which it is on one row. `None` if there is none. A
/// function may be an aggregate, which reads the rows of the query around it.
fn inline_from_less(selection: &SQLExpr) -> Option<SQLExpr> {
    fn value(subquery: &Query) -> Option<&SQLExpr> {
        let select = eligible_query_block(subquery)?;
        first_row_limit(subquery)?;
        // A clause that is dropped must not be left unchecked.
        if !select.from.is_empty()
            || select.selection.is_some()
            || subquery.order_by.is_some()
            || matches!(select.distinct, Some(Distinct::On(_)))
            || select.exclude.is_some()
            || !select.named_window.is_empty()
            || !select.cluster_by.is_empty()
            || !select.distribute_by.is_empty()
            || !select.sort_by.is_empty()
        {
            return None;
        }
        let [SelectItem::UnnamedExpr(value) | SelectItem::ExprWithAlias { expr: value, .. }] =
            select.projection.as_slice()
        else {
            return None;
        };
        let plain = visit_expressions(value, |e| match e {
            SQLExpr::Function(_) => ControlFlow::Break(()),
            e if is_subquery_expr(e) => ControlFlow::Break(()),
            _ => ControlFlow::Continue(()),
        })
        .is_continue();
        plain.then_some(value)
    }

    struct Inline {
        query_depth: usize,
        inlined: bool,
    }

    impl VisitorMut for Inline {
        type Break = ();

        fn pre_visit_query(&mut self, _query: &mut Query) -> ControlFlow<()> {
            self.query_depth += 1;
            ControlFlow::Continue(())
        }

        fn post_visit_query(&mut self, _query: &mut Query) -> ControlFlow<()> {
            self.query_depth -= 1;
            ControlFlow::Continue(())
        }

        fn pre_visit_expr(&mut self, expr: &mut SQLExpr) -> ControlFlow<()> {
            if self.query_depth == 0
                && let SQLExpr::Subquery(subquery) = &*expr
                && let Some(value) = value(subquery)
            {
                *expr = SQLExpr::Nested(Box::new(value.clone()));
                self.inlined = true;
            }
            ControlFlow::Continue(())
        }
    }

    let mut selection = selection.clone();
    let mut inline = Inline {
        query_depth: 0,
        inlined: false,
    };
    let _ = dyn_visit::visit_mut(&mut selection, &mut inline);
    inline.inlined.then_some(selection)
}

fn frame_columns<'a>(frame: Option<(&LazyFrame, &'a Schema)>) -> ScopeColumns<'a> {
    match frame {
        Some((_, schema)) => ScopeColumns::Frame(schema),
        None => ScopeColumns::None,
    }
}

/// The comparison and subquery of a conjunct `lhs IN (y)`, `lhs <op> (y)` or `EXISTS (y)`.
fn candidate(conj: &SQLExpr) -> Option<(Option<Compare<'_>>, &Query)> {
    use SQLBinaryOperator::*;
    let (compare, subquery) = match strip_nested(conj) {
        SQLExpr::InSubquery {
            expr,
            subquery,
            negated: false,
        } => (Some((expr.as_ref(), Eq, false)), subquery.as_ref()),
        SQLExpr::Exists {
            subquery,
            negated: false,
        } => (None, subquery.as_ref()),
        SQLExpr::BinaryOp {
            left,
            op: op @ (Eq | NotEq | Lt | LtEq | Gt | GtEq),
            right,
        } => match (strip_nested(left), strip_nested(right)) {
            (SQLExpr::Subquery(subquery), lhs) => {
                (Some((lhs, op.clone(), true)), subquery.as_ref())
            },
            (lhs, SQLExpr::Subquery(subquery)) => {
                (Some((lhs, op.clone(), false)), subquery.as_ref())
            },
            _ => return None,
        },
        _ => return None,
    };
    let lhs = compare.as_ref().map(|(lhs, ..)| *lhs);
    if lhs.is_some_and(|lhs| expr_contains_subquery(lhs) || matches!(lhs, SQLExpr::Tuple(_))) {
        return None;
    }
    Some((compare, subquery))
}

/// Move `y`, compared by `compare` (or under EXISTS), into the FROM of `x`, whose context
/// is `x_ctx` and whose frame is `x_frame` (`None` without FROM). `scopes` are the queries
/// around `x`. `filters_allowed` is whether a row of `x` may match several rows of `y`.
/// `None` if `y` reads none of `scopes`, or isn't a shape that moves.
fn try_move(
    scopes: &[Scope<'_>],
    x_ctx: &mut SQLContext,
    x_frame: Option<(&LazyFrame, &Schema)>,
    compare: Option<Compare<'_>>,
    y: &Query,
    filters_allowed: bool,
) -> PolarsResult<Option<Moved>> {
    use SQLBinaryOperator::*;
    let Some(first_row) = first_row_limit(y) else {
        return Ok(None);
    };
    let Some(select) = eligible_subquery_body(y) else {
        return Ok(None);
    };
    if matches!(select.distinct, Some(Distinct::On(_))) || !select.named_window.is_empty() {
        return Ok(None);
    }
    let order_by_exprs: &[OrderByExpr] = match y.order_by.as_ref().map(|o| &o.kind) {
        None => &[],
        Some(OrderByKind::Expressions(exprs)) => exprs,
        Some(_) => return Ok(None),
    };
    let order_aggregates = order_by_exprs
        .iter()
        .any(|ob| has_aggregate(x_ctx, &ob.expr));
    let value = match compare {
        Some(_) => {
            let [SelectItem::UnnamedExpr(value) | SelectItem::ExprWithAlias { expr: value, .. }] =
                select.projection.as_slice()
            else {
                return Ok(None);
            };
            if expr_has_window_functions(value) {
                return Ok(None);
            }
            Some(value)
        },
        // An aggregate makes EXISTS return one row whatever the rows are. The projection
        // and the order are otherwise ignored.
        None if order_aggregates
            || select
                .projection
                .iter()
                .any(|item| has_aggregate(x_ctx, item)) =>
        {
            return Ok(None);
        },
        None => None,
    };
    // An aggregate is taken per value of the correlation keys. A key with no rows then
    // has no group, which is only the same as a NULL value.
    let aggregates = value.is_some_and(|value| has_aggregate(x_ctx, value));
    if aggregates && !null_over_no_rows(x_ctx, value.unwrap()) {
        return Ok(None);
    }
    // An aggregate in the order alone also makes the subquery return one row.
    if value.is_some() && !aggregates && order_aggregates {
        return Ok(None);
    }
    // Without a limit, the distinct values of a key can be several. Only an equality
    // matches each row of the query around it with at most one of them. A limit doesn't
    // change whether EXISTS finds a row, nor how many rows an aggregate returns.
    let first_row = first_row && value.is_some() && !aggregates;
    if compare.as_ref().is_some_and(|(_, op, _)| *op != Eq) && !aggregates && !first_row {
        return Ok(None);
    }
    let order_by = match first_row {
        true => match first_row_order(y, select, order_by_exprs) {
            Some(order_by) => order_by,
            None => return Ok(None),
        },
        false => Vec::new(),
    };
    // The moved table reads the rows of every key at once.
    if value
        .into_iter()
        .chain(&select.selection)
        .chain(order_by.iter().map(|ob| &ob.expr))
        .any(|e| numbers_rows(x_ctx, e))
    {
        return Ok(None);
    }

    let mut y_ctx = x_ctx.isolated();
    let Some((y_names, _, y_schema)) = y_ctx.resolve_subquery_from(select)? else {
        return Ok(None);
    };
    // Dropping the projection of EXISTS must not change its rows, as `UNNEST` does.
    if compare.is_none() && exists_aggregates(&y_ctx, y, select, &y_names, &y_schema) != Some(false)
    {
        return Ok(None);
    }
    // The compared operand may be a column of a query around `x`, which `x` then reads
    // next to the moved subquery. That only helps if `y` reads `x`: otherwise the
    // comparison already reads no column of `x`.
    let foreign_lhs = match compare {
        Some((lhs, ..)) => {
            let x_chain: Vec<Scope> = std::iter::once(Scope {
                ctx: &*x_ctx,
                columns: frame_columns(x_frame),
            })
            .chain(scopes.iter().rev().copied())
            .collect();
            match bind(&x_chain, lhs)? {
                Some(Binding::Foreign {
                    dtype: Some(dtype), ..
                }) => Some(dtype),
                _ => None,
            }
        },
        None => None,
    };
    let mut kept = Vec::new();
    let mut keys = Vec::new();
    // Other conjuncts that read a query around `y`. They move into `x`, and read the
    // columns of `y` that they need from the moved table. A row of `x` can then match
    // several of its rows.
    let mut filters = Vec::new();
    let mut locals = Vec::new();
    let mut reads_outer = false;
    if let Some(selection) = &select.selection {
        // Innermost first: `y`, `x`, then the queries around `x`.
        let chain: Vec<Scope> = [
            Scope {
                ctx: &y_ctx,
                columns: ScopeColumns::Frame(&y_schema),
            },
            Scope {
                ctx: &*x_ctx,
                columns: frame_columns(x_frame),
            },
        ]
        .into_iter()
        .chain(scopes.iter().rev().copied())
        .collect();
        // A conjunct with a subquery stays: planning the derived table rejects it if it
        // reads anything around `y`.
        for c in flatten_and_conditions(selection) {
            if expr_contains_subquery(c) || reads_local_only(&chain, c)? == Some(true) {
                kept.push(c.clone());
                continue;
            }
            if let Some((key, depth)) = correlation_key(&chain, c)? {
                reads_outer |= depth >= 2;
                keys.push(key);
                continue;
            }
            let Some((columns, depth)) = foreign_filter(&chain, c)? else {
                return Ok(None);
            };
            reads_outer |= depth >= 2;
            filters.push(c);
            for column in columns {
                if !locals.contains(&column) {
                    locals.push(column);
                }
            }
        }
    }
    reads_outer |= foreign_lhs.is_some() && !(keys.is_empty() && filters.is_empty());
    if !reads_outer {
        return Ok(None);
    }
    // A limit or an aggregate per key must see the rows that the filters drop.
    let distinct = !aggregates && !first_row;
    if !filters.is_empty() && (!filters_allowed || !distinct) {
        return Ok(None);
    }
    // The moved table computes the value before the filters drop rows, so the value must
    // not be able to fail on a row that a filter drops: it is a column or a literal.
    let plain = |value: &SQLExpr| {
        matches!(
            strip_nested(value),
            SQLExpr::Identifier(_) | SQLExpr::CompoundIdentifier(_) | SQLExpr::Value(_)
        )
    };
    if !filters.is_empty() && value.is_some_and(|value| !plain(value)) {
        return Ok(None);
    }
    // A moved table without columns would have no rows.
    if value.is_none() && keys.is_empty() && locals.is_empty() {
        return Ok(None);
    }

    let name = format!("{NESTED_PREFIX}{}", unique_column_name());
    let value_name = format!("{name}_v");
    let key_names: Vec<String> = (0..keys.len()).map(|i| format!("{name}_k{i}")).collect();
    let local_names: Vec<String> = (0..locals.len()).map(|i| format!("{name}_c{i}")).collect();
    let mut pulled = select.clone();
    pulled.projection = value
        .map(|value| (value.clone(), value_name.clone()))
        .into_iter()
        .chain(
            keys.iter()
                .map(|k| k.inner.clone())
                .zip(key_names.iter().cloned()),
        )
        .chain(locals.iter().cloned().zip(local_names.iter().cloned()))
        .map(|(expr, alias)| SelectItem::ExprWithAlias {
            expr,
            alias: Ident::new(alias),
        })
        .collect();
    pulled.selection = combine_conditions(kept, And);
    let partition_by: Vec<SQLExpr> = keys.iter().map(|k| k.inner.clone()).collect();
    if aggregates {
        pulled.group_by = GroupByExpr::Expressions(partition_by, vec![]);
    } else if first_row {
        pulled.distinct = None;
        pulled.qualify = Some(SQLExpr::BinaryOp {
            left: Box::new(row_number(partition_by, order_by)),
            op: Eq,
            right: Box::new(SQLExpr::value(SQLValue::Number("1".to_string(), false))),
        });
    } else {
        pulled.distinct = Some(Distinct::Distinct);
    }
    let mut pulled_query = y.clone();
    pulled_query.body = Box::new(SetExpr::Select(Box::new(pulled)));
    pulled_query.order_by = None;
    pulled_query.limit_clause = None;
    pulled_query.fetch = None;

    // Planning the derived table as the FROM will (strict, as `x_ctx` is) shows that it
    // reads only what it declares, and gives the dtypes of its columns.
    let mut pulled_ctx = x_ctx.isolated();
    let mut pulled_lf = pulled_ctx.execute_query(&pulled_query)?;
    let pulled_schema = pulled_ctx.get_frame_schema(&mut pulled_lf)?;
    // Deduplicating or grouping keeps one row per value that the comparison tells apart
    // only when the comparison doesn't merge values of the derived table.
    for (key, key_name) in keys.iter().zip(&key_names) {
        let Some(dtype) = pulled_schema.get(key_name.as_str()) else {
            return Ok(None);
        };
        if !compares_exactly(dtype, &key.dtype) {
            return Ok(None);
        }
    }
    if distinct && let Some((lhs, ..)) = compare {
        let Some(value_dtype) = pulled_schema.get(value_name.as_str()) else {
            return Ok(None);
        };
        let lhs_dtype = match foreign_lhs {
            Some(dtype) => dtype,
            None => match local_dtype(scopes, x_ctx, x_frame, lhs)? {
                Some(dtype) => dtype,
                None => return Ok(None),
            },
        };
        if !compares_exactly(value_dtype, &lhs_dtype) {
            return Ok(None);
        }
    }

    let table = derived_table(pulled_query, &name);
    let column =
        |column: &str| SQLExpr::CompoundIdentifier(vec![Ident::new(&name), Ident::new(column)]);
    let binary = |left: SQLExpr, op: SQLBinaryOperator, right: SQLExpr| SQLExpr::BinaryOp {
        left: Box::new(left),
        op,
        right: Box::new(right),
    };
    let filters = filters.into_iter().map(|filter| {
        let mut filter = filter.clone();
        let _ = visit_expressions_mut(&mut filter, |e| {
            if let Some(i) = locals.iter().position(|local| local == e) {
                *e = column(&local_names[i]);
            }
            ControlFlow::<()>::Continue(())
        });
        filter
    });
    let replaced = compare
        .map(|(lhs, op, subquery_left)| match subquery_left {
            true => binary(column(&value_name), op, lhs.clone()),
            false => binary(lhs.clone(), op, column(&value_name)),
        })
        .into_iter()
        .chain(
            keys.into_iter()
                .zip(&key_names)
                .map(|(key, key_name)| binary(column(key_name), Eq, key.foreign)),
        )
        .chain(filters)
        .collect();
    let columns = value
        .map(|_| value_name)
        .into_iter()
        .chain(key_names)
        .chain(local_names)
        .collect();
    Ok(Some(Moved {
        table,
        columns,
        replaced,
    }))
}

/// The `ORDER BY` of a first-row subquery, to order the rows of each key by. `None` for an
/// order that isn't read from the row alone.
fn first_row_order(y: &Query, select: &Select, exprs: &[OrderByExpr]) -> Option<Vec<OrderByExpr>> {
    if y.order_by.as_ref().is_some_and(|o| o.interpolate.is_some()) {
        return None;
    }
    let alias = match &select.projection[0] {
        SelectItem::ExprWithAlias { alias, .. } => Some(alias),
        _ => None,
    };
    // An order by the output column, by name or by position, isn't kept. A window
    // function would see the rows of the other keys.
    let unsupported = |e: &SQLExpr| match strip_nested(e) {
        SQLExpr::Identifier(ident) => alias.is_some_and(|a| a.value == ident.value),
        SQLExpr::Value(_) => true,
        e => expr_has_window_functions(e),
    };
    if exprs
        .iter()
        .any(|ob| ob.with_fill.is_some() || unsupported(&ob.expr))
    {
        return None;
    }
    Some(exprs.to_vec())
}

/// Call `f` on each column that `expr` reads outside of its subqueries.
fn visit_columns<E>(expr: &SQLExpr, f: impl FnMut(&SQLExpr) -> ControlFlow<E>) -> ControlFlow<E> {
    struct Columns<F> {
        query_depth: usize,
        f: F,
    }

    impl<E, F: FnMut(&SQLExpr) -> ControlFlow<E>> Visitor for Columns<F> {
        type Break = E;

        fn pre_visit_query(&mut self, _query: &Query) -> ControlFlow<E> {
            self.query_depth += 1;
            ControlFlow::Continue(())
        }

        fn post_visit_query(&mut self, _query: &Query) -> ControlFlow<E> {
            self.query_depth -= 1;
            ControlFlow::Continue(())
        }

        fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<E> {
            match expr {
                SQLExpr::Identifier(_) | SQLExpr::CompoundIdentifier(_)
                    if self.query_depth == 0 =>
                {
                    (self.f)(expr)
                },
                _ => ControlFlow::Continue(()),
            }
        }
    }

    dyn_visit::visit(expr, &mut Columns { query_depth: 0, f })
}

/// The scope in `chain` (innermost first) that a column binds to: the first one with a
/// relation the qualifier names, or with a struct column of the qualifier's name. An
/// unqualified column, and one no scope binds, is left to the innermost scope. `None` for
/// a column that its relation doesn't have.
fn bind(chain: &[Scope<'_>], column: &SQLExpr) -> PolarsResult<Option<Binding>> {
    let SQLExpr::CompoundIdentifier(parts) = column else {
        return Ok(Some(Binding::Local));
    };
    for (depth, scope) in chain.iter().enumerate() {
        let Some(found) = scope.find(parts)? else {
            continue;
        };
        return Ok(match (found, depth) {
            (Found::Missing, _) => None,
            (_, 0) => Some(Binding::Local),
            (Found::Column(dtype), depth) => Some(Binding::Foreign {
                depth,
                dtype: Some(dtype),
            }),
            (Found::Field, depth) => Some(Binding::Foreign { depth, dtype: None }),
        });
    }
    Ok(Some(Binding::Local))
}

impl Scope<'_> {
    fn find(&self, parts: &[Ident]) -> PolarsResult<Option<Found>> {
        if matches!(self.columns, ScopeColumns::None) {
            return Ok(None);
        }
        if let Some((relation, len)) = self.ctx.resolve_qualifier(parts)? {
            if len + 1 != parts.len() {
                return Ok(Some(Found::Field));
            }
            return Ok(Some(
                match self.column_dtype(&relation, &parts[len].value) {
                    Some(dtype) => Found::Column(dtype),
                    None => Found::Missing,
                },
            ));
        }
        let is_struct =
            |schema: &Schema| matches!(schema.get(&parts[0].value), Some(DataType::Struct(_)));
        let has_struct = match self.columns {
            ScopeColumns::None => false,
            ScopeColumns::Frame(schema) => is_struct(schema),
            ScopeColumns::Join(left, right) => is_struct(&left.schema) || is_struct(&right.schema),
        };
        Ok(has_struct.then_some(Found::Field))
    }

    fn column_dtype(&self, relation: &str, column: &str) -> Option<DataType> {
        let schema = match self.columns {
            ScopeColumns::None => return None,
            ScopeColumns::Frame(schema) => schema,
            ScopeColumns::Join(left, right) => {
                if self.ctx.is_right_relation(relation, right) {
                    return right.schema.get(column).cloned();
                }
                if !self.ctx.is_joined_relation(relation, left) {
                    return None;
                }
                &left.schema
            },
        };
        schema
            .get(&self.ctx.resolve_name(relation, column))
            .cloned()
    }
}

/// Whether every column `expr` reads binds to the innermost scope of `chain`, or `None`
/// for a column that its relation doesn't have.
fn reads_local_only(chain: &[Scope<'_>], expr: &SQLExpr) -> PolarsResult<Option<bool>> {
    let mut local = true;
    let flow = visit_columns(expr, |column| match bind(chain, column) {
        Ok(Some(Binding::Local)) => ControlFlow::Continue(()),
        Ok(Some(Binding::Foreign { .. })) => {
            local = false;
            ControlFlow::Continue(())
        },
        Ok(None) => ControlFlow::Break(None),
        Err(e) => ControlFlow::Break(Some(e)),
    });
    match flow {
        ControlFlow::Break(Some(e)) => Err(e),
        ControlFlow::Break(None) => Ok(None),
        ControlFlow::Continue(()) => Ok(Some(local)),
    }
}

/// Whether `expr` reads a column of the innermost scope of `chain` and one of a scope
/// around it.
fn reads_local_and_foreign(chain: &[Scope<'_>], expr: &SQLExpr) -> PolarsResult<bool> {
    let (mut local, mut foreign) = (false, false);
    let flow = visit_columns(expr, |column| match bind(chain, column) {
        Ok(Some(Binding::Local)) => {
            local = true;
            ControlFlow::Continue(())
        },
        Ok(Some(Binding::Foreign { .. })) => {
            foreign = true;
            ControlFlow::Continue(())
        },
        Ok(None) => ControlFlow::Break(None),
        Err(e) => ControlFlow::Break(Some(e)),
    });
    match flow {
        ControlFlow::Break(Some(e)) => Err(e),
        ControlFlow::Break(None) => Ok(false),
        ControlFlow::Continue(()) => Ok(local && foreign),
    }
}

/// `conj` as an equality between an expression over the subquery alone and a column of
/// a query around it, with the depth of that query in `chain`.
fn correlation_key(
    chain: &[Scope<'_>],
    conj: &SQLExpr,
) -> PolarsResult<Option<(CorrelationKey, usize)>> {
    let SQLExpr::BinaryOp {
        left,
        op: SQLBinaryOperator::Eq,
        right,
    } = strip_nested(conj)
    else {
        return Ok(None);
    };
    for (inner, foreign) in [(left, right), (right, left)] {
        let foreign = strip_nested(foreign);
        if expr_has_window_functions(inner)
            || has_aggregate(chain[0].ctx, inner.as_ref())
            || !expr_references_any_column(inner)
            || reads_local_only(chain, inner)? != Some(true)
        {
            continue;
        }
        if let Some(Binding::Foreign {
            depth,
            dtype: Some(dtype),
        }) = bind(chain, foreign)?
        {
            let key = CorrelationKey {
                inner: inner.as_ref().clone(),
                foreign: foreign.clone(),
                dtype,
            };
            return Ok(Some((key, depth)));
        }
    }
    Ok(None)
}

/// The columns of the subquery that `conj` reads, and the depth in `chain` of the
/// furthest query around it that it reads. `None` if it calls a function, or reads a
/// column that its relation doesn't have.
fn foreign_filter(
    chain: &[Scope<'_>],
    conj: &SQLExpr,
) -> PolarsResult<Option<(Vec<SQLExpr>, usize)>> {
    // A function may read other rows or draw a random value, which differs once the
    // filter runs per pair of rows.
    if calls_function(conj) {
        return Ok(None);
    }
    let mut locals = Vec::new();
    let mut max_depth = 0;
    let flow = visit_columns(conj, |column| match bind(chain, column) {
        Ok(Some(Binding::Local)) => {
            locals.push(column.clone());
            ControlFlow::Continue(())
        },
        Ok(Some(Binding::Foreign { depth, .. })) => {
            max_depth = max_depth.max(depth);
            ControlFlow::Continue(())
        },
        Ok(None) => ControlFlow::Break(None),
        Err(e) => ControlFlow::Break(Some(e)),
    });
    match flow {
        ControlFlow::Break(Some(e)) => Err(e),
        ControlFlow::Break(None) => Ok(None),
        ControlFlow::Continue(()) => Ok((max_depth > 0).then_some((locals, max_depth))),
    }
}

/// Whether repeating a row of `x` (whose body is `select`) keeps what the query around it
/// reads: `x` is read as a set or for its first row, and calls no function outside of its
/// subqueries. A function may aggregate, number the rows or draw a random value.
fn repeats_harmless(x: &Query, select: &Select, read_as: ReadAs) -> bool {
    let read_once = match (read_as, first_row_limit(x)) {
        (ReadAs::Groups, _) => false,
        (_, Some(true)) => true,
        (ReadAs::Set, Some(false)) => true,
        _ => false,
    };
    let order_by: &[OrderByExpr] = match x.order_by.as_ref().map(|o| &o.kind) {
        None => &[],
        Some(OrderByKind::Expressions(exprs)) => exprs,
        Some(_) => return false,
    };
    let projection = select.projection.iter().filter_map(|item| match item {
        SelectItem::UnnamedExpr(e) | SelectItem::ExprWithAlias { expr: e, .. } => Some(e),
        _ => None,
    });
    let join_conditions = select
        .from
        .iter()
        .flat_map(|item| &item.joins)
        .filter_map(|join| match &join.join_operator {
            JoinOperator::Join(JoinConstraint::On(on))
            | JoinOperator::Inner(JoinConstraint::On(on)) => Some(on),
            _ => None,
        });
    read_once
        && !matches!(select.distinct, Some(Distinct::On(_)))
        && !projection
            .chain(order_by.iter().map(|ob| &ob.expr))
            .chain(&select.selection)
            .chain(join_conditions)
            .any(calls_function)
}

/// Whether `expr` calls a function without OVER that reads the other rows of its frame,
/// such as `ROW_NUMBER()`, outside of its subqueries. A function that isn't known counts
/// as one.
fn numbers_rows(ctx: &SQLContext, expr: &SQLExpr) -> bool {
    use PolarsSQLFunctions::*;
    struct Functions<'a> {
        ctx: &'a SQLContext,
        query_depth: usize,
    }

    impl Visitor for Functions<'_> {
        type Break = ();

        fn pre_visit_query(&mut self, _query: &Query) -> ControlFlow<()> {
            self.query_depth += 1;
            ControlFlow::Continue(())
        }

        fn post_visit_query(&mut self, _query: &Query) -> ControlFlow<()> {
            self.query_depth -= 1;
            ControlFlow::Continue(())
        }

        fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<()> {
            match expr {
                SQLExpr::Function(function) if self.query_depth == 0 && function.over.is_none() => {
                    match PolarsSQLFunctions::try_from_sql(function, self.ctx) {
                        Ok(
                            CumeDist | DenseRank | Lag | Lead | NthValue | Ntile | PercentRank
                            | Rank | RowNumber | Udf(_),
                        )
                        | Err(_) => ControlFlow::Break(()),
                        Ok(_) => ControlFlow::Continue(()),
                    }
                },
                _ => ControlFlow::Continue(()),
            }
        }
    }

    dyn_visit::visit(
        expr,
        &mut Functions {
            ctx,
            query_depth: 0,
        },
    )
    .is_break()
}

/// Whether `expr` calls a function outside of its subqueries.
fn calls_function(expr: &SQLExpr) -> bool {
    struct Functions {
        query_depth: usize,
    }

    impl Visitor for Functions {
        type Break = ();

        fn pre_visit_query(&mut self, _query: &Query) -> ControlFlow<()> {
            self.query_depth += 1;
            ControlFlow::Continue(())
        }

        fn post_visit_query(&mut self, _query: &Query) -> ControlFlow<()> {
            self.query_depth -= 1;
            ControlFlow::Continue(())
        }

        fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<()> {
            match expr {
                SQLExpr::Function(_) if self.query_depth == 0 => ControlFlow::Break(()),
                _ => ControlFlow::Continue(()),
            }
        }
    }

    dyn_visit::visit(expr, &mut Functions { query_depth: 0 }).is_break()
}

/// The dtype of `expr` over the frame of `x`, or `None` if it reads anything else.
fn local_dtype(
    scopes: &[Scope<'_>],
    x_ctx: &mut SQLContext,
    x_frame: Option<(&LazyFrame, &Schema)>,
    expr: &SQLExpr,
) -> PolarsResult<Option<DataType>> {
    let Some((lf, schema)) = x_frame else {
        return Ok(None);
    };
    let local = {
        let chain: Vec<Scope> = std::iter::once(Scope {
            ctx: &*x_ctx,
            columns: ScopeColumns::Frame(schema),
        })
        .chain(scopes.iter().rev().copied())
        .collect();
        reads_local_only(&chain, expr)?
    };
    if local != Some(true) {
        return Ok(None);
    }
    let parsed = parse_sql_expr(expr, x_ctx, Some(schema))?;
    let mut probe = lf.clone().select([parsed]);
    let probe_schema = x_ctx.get_frame_schema(&mut probe)?;
    Ok(probe_schema.get_at_index(0).map(|(_, dtype)| dtype.clone()))
}

/// Whether comparing a value of `inner` with one of `other` casts it without merging
/// values, so that distinct values of `inner` never equal the same value of `other`.
pub(crate) fn compares_exactly(inner: &DataType, other: &DataType) -> bool {
    if inner == other {
        return true;
    }
    get_supertype(inner, other).is_some_and(|to| holds_values(inner, &to))
}

/// Whether `to` holds every value of the numeric dtype `from`.
fn holds_values(from: &DataType, to: &DataType) -> bool {
    let bits = |dtype: &DataType| match dtype {
        DataType::Int8 | DataType::UInt8 => 8,
        DataType::Int16 | DataType::UInt16 => 16,
        DataType::Int32 | DataType::UInt32 | DataType::Float32 => 32,
        DataType::Int64 | DataType::UInt64 | DataType::Float64 => 64,
        DataType::Int128 | DataType::UInt128 => 128,
        _ => 0,
    };
    match (from, to) {
        _ if from == to => true,
        (DataType::Float32, DataType::Float64) => true,
        // The mantissa holds every integer of up to 24 or 53 bits.
        (from, DataType::Float32) if from.is_integer() => bits(from) <= 16,
        (from, DataType::Float64) if from.is_integer() => bits(from) <= 32,
        (from, to) if from.is_integer() && to.is_integer() => {
            match (from.is_signed_integer(), to.is_signed_integer()) {
                (true, true) | (false, false) => bits(from) <= bits(to),
                (false, true) => bits(from) < bits(to),
                (true, false) => false,
            }
        },
        _ => false,
    }
}

/// Whether `node` calls an aggregate outside of its subqueries. A function that isn't
/// known counts as one.
pub(crate) fn has_aggregate(ctx: &SQLContext, node: &(impl Visit + ?Sized)) -> bool {
    struct Aggregates<'a> {
        ctx: &'a SQLContext,
        query_depth: usize,
    }

    impl Visitor for Aggregates<'_> {
        type Break = ();

        fn pre_visit_query(&mut self, _query: &Query) -> ControlFlow<()> {
            self.query_depth += 1;
            ControlFlow::Continue(())
        }

        fn post_visit_query(&mut self, _query: &Query) -> ControlFlow<()> {
            self.query_depth -= 1;
            ControlFlow::Continue(())
        }

        fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<()> {
            match expr {
                SQLExpr::Function(function) if self.query_depth == 0 && function.over.is_none() => {
                    match PolarsSQLFunctions::try_from_sql(function, self.ctx) {
                        Ok(PolarsSQLFunctions::Udf(_)) | Err(_) => ControlFlow::Break(()),
                        Ok(f) if f.is_builtin_aggregate() => ControlFlow::Break(()),
                        Ok(_) => ControlFlow::Continue(()),
                    }
                },
                _ => ControlFlow::Continue(()),
            }
        }
    }

    dyn_visit::visit(
        node,
        &mut Aggregates {
            ctx,
            query_depth: 0,
        },
    )
    .is_break()
}

/// Whether `expr` is NULL when it aggregates no rows: an aggregate other than a count,
/// possibly cast, negated or combined with a constant by arithmetic.
fn null_over_no_rows(ctx: &SQLContext, expr: &SQLExpr) -> bool {
    use PolarsSQLFunctions::*;
    match strip_nested(expr) {
        SQLExpr::Function(function) if function.over.is_none() => matches!(
            PolarsSQLFunctions::try_from_sql(function, ctx),
            Ok(Avg
                | Corr
                | CovarPop
                | CovarSamp
                | First
                | FirstValue
                | Last
                | Max
                | Median
                | Min
                | QuantileCont
                | QuantileDisc
                | StdDev
                | Sum
                | Variance)
        ),
        SQLExpr::Cast { expr, .. }
        | SQLExpr::UnaryOp {
            op: SQLUnaryOperator::Minus,
            expr,
        } => null_over_no_rows(ctx, expr),
        SQLExpr::BinaryOp {
            left,
            op:
                SQLBinaryOperator::Plus
                | SQLBinaryOperator::Minus
                | SQLBinaryOperator::Multiply
                | SQLBinaryOperator::Divide
                | SQLBinaryOperator::Modulo,
            right,
        } => {
            let constant = |e: &SQLExpr| !expr_references_any_column(e) && !has_aggregate(ctx, e);
            (null_over_no_rows(ctx, left) && constant(right))
                || (constant(left) && null_over_no_rows(ctx, right))
        },
        _ => false,
    }
}

/// `ROW_NUMBER() OVER (PARTITION BY .. ORDER BY ..)`.
fn row_number(partition_by: Vec<SQLExpr>, order_by: Vec<OrderByExpr>) -> SQLExpr {
    SQLExpr::Function(Function {
        name: ObjectName::from(vec![Ident::new("ROW_NUMBER")]),
        uses_odbc_syntax: false,
        parameters: FunctionArguments::None,
        args: FunctionArguments::List(FunctionArgumentList {
            duplicate_treatment: None,
            args: vec![],
            clauses: vec![],
        }),
        filter: None,
        null_treatment: None,
        over: Some(WindowType::WindowSpec(WindowSpec {
            window_name: None,
            partition_by,
            order_by,
            window_frame: None,
        })),
        within_group: vec![],
    })
}
