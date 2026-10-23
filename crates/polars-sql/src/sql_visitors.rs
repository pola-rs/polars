//! SQLVisitor helper implementations for traversing SQL AST expressions.
//!
//! This module provides visitor implementations used throughout the SQL interface
//! to analyze and check SQL expressions for various properties.

use std::ops::ControlFlow;

use polars_core::prelude::*;
use sqlparser::ast::{
    Expr as SQLExpr, Function as SQLFunction, FunctionArg, FunctionArgExpr, FunctionArgumentList,
    FunctionArguments, ObjectName, Query, SetExpr, TableFactor, Visitor as SQLVisitor, VisitorMut,
    With,
};
use sqlparser::keywords::ALL_KEYWORDS;

use crate::context::{RelationName, match_qualifier, object_name_to_string};
use crate::dyn_visit::{self, visit_expressions};

// ---------------------------------------------------------------------------
// FindTableIdentifier
// ---------------------------------------------------------------------------

/// Visitor that checks if an expression tree contains a reference to a specific table.
pub(crate) struct FindTableIdentifier<'a> {
    table_name: &'a str,
    relations: &'a [RelationName],
    found: bool,
}

impl<'a> FindTableIdentifier<'a> {
    fn new(table_name: &'a str, relations: &'a [RelationName]) -> Self {
        Self {
            table_name,
            relations,
            found: false,
        }
    }
}

impl<'a> SQLVisitor for FindTableIdentifier<'a> {
    type Break = ();

    fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<Self::Break> {
        if let SQLExpr::CompoundIdentifier(idents) = expr {
            // the longest leading run naming a relation decides (`a.b.k` names `a.b`, not `a`)
            if match_qualifier(idents, self.relations.iter())
                .is_some_and(|(matches, _)| matches.iter().any(|r| r.key() == self.table_name))
            {
                self.found = true; // return immediately on first match
                return ControlFlow::Break(());
            }
        }
        ControlFlow::Continue(())
    }
}

/// Check if a SQL expression contains a reference to a specific table, among `relations`
/// in scope.
pub(crate) fn expr_refers_to_table(
    expr: &SQLExpr,
    table_name: &str,
    relations: &[RelationName],
) -> bool {
    let mut table_finder = FindTableIdentifier::new(table_name, relations);
    let _ = dyn_visit::visit(expr, &mut table_finder);
    table_finder.found
}

// ---------------------------------------------------------------------------
// UnqualifiedColumnsInSchema
// ---------------------------------------------------------------------------

/// Visitor that collects unqualified column identifiers (`SQLExpr::Identifier`, as opposed
/// to `SQLExpr::CompoundIdentifier`) referenced in an expression tree, checking each one
/// against a given `Schema` as it goes.
struct UnqualifiedColumnsInSchema<'a> {
    schema: &'a Schema,
    found_any: bool,
    all_in_schema: bool,
}

impl<'a> UnqualifiedColumnsInSchema<'a> {
    fn new(schema: &'a Schema) -> Self {
        Self {
            schema,
            found_any: false,
            all_in_schema: true,
        }
    }
}

impl<'a> SQLVisitor for UnqualifiedColumnsInSchema<'a> {
    type Break = ();

    fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<Self::Break> {
        if let SQLExpr::Identifier(ident) = expr {
            self.found_any = true;
            if !self.schema.contains(ident.value.as_str()) {
                self.all_in_schema = false;
                return ControlFlow::Break(()); // short-circuit on first non-match
            }
        }
        ControlFlow::Continue(())
    }
}

/// Check whether a raw SQL expression contains at least one *unqualified* column
/// identifier, and that every unqualified identifier it contains exists in `schema`.
pub(crate) fn sql_expr_cols_all_in_schema(expr: &SQLExpr, schema: &Schema) -> bool {
    let mut visitor = UnqualifiedColumnsInSchema::new(schema);
    let _ = dyn_visit::visit(expr, &mut visitor);
    visitor.found_any && visitor.all_in_schema
}

// ---------------------------------------------------------------------------
// QualifyExpression
// ---------------------------------------------------------------------------

/// Visitor used to check a SQL expression used in a QUALIFY clause.
/// (Confirms window functions are present and collects column refs in one pass).
pub(crate) struct QualifyExpression {
    has_window_functions: bool,
    column_refs: PlHashSet<String>,
}

impl QualifyExpression {
    fn new() -> Self {
        Self {
            has_window_functions: false,
            column_refs: PlHashSet::new(),
        }
    }

    pub(crate) fn analyze(expr: &SQLExpr) -> (bool, PlHashSet<String>) {
        let mut analyzer = Self::new();
        let _ = dyn_visit::visit(expr, &mut analyzer);
        (analyzer.has_window_functions, analyzer.column_refs)
    }
}

impl SQLVisitor for QualifyExpression {
    type Break = ();

    fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<Self::Break> {
        match expr {
            SQLExpr::Function(func) if func.over.is_some() => {
                self.has_window_functions = true;
            },
            SQLExpr::Identifier(ident) => {
                self.column_refs.insert(ident.value.clone());
            },
            SQLExpr::CompoundIdentifier(idents) if !idents.is_empty() => {
                self.column_refs
                    .insert(idents.last().unwrap().value.clone());
            },
            _ => {},
        }
        ControlFlow::Continue(())
    }
}

// ---------------------------------------------------------------------------
// AmbiguousColumnVisitor
// ---------------------------------------------------------------------------

/// Format an identifier, quoting only if necessary (or `force` is true).
fn maybe_quote(s: &str, force: bool) -> String {
    let needs_quoting = force
        || s.is_empty()
        || s.starts_with(|c: char| c.is_ascii_digit())
        || !s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        || ALL_KEYWORDS.contains(&s.to_ascii_uppercase().as_str());
    if needs_quoting {
        format!("\"{s}\"")
    } else {
        s.to_string()
    }
}

/// Visitor that checks for unqualified references to columns that exist in
/// multiple tables (see `check_ambiguous_column`). Subqueries are skipped, as
/// their columns resolve against their own FROM first.
struct AmbiguousColumnVisitor<'a> {
    joined_aliases: &'a PlHashMap<String, PlHashMap<String, String>>,
    base_table_name: &'a str,
    using_cols: &'a [(String, String)],
    relations: &'a [RelationName],
    subquery_depth: usize,
}

impl SQLVisitor for AmbiguousColumnVisitor<'_> {
    type Break = PolarsError;

    fn pre_visit_query(&mut self, _query: &Query) -> ControlFlow<Self::Break> {
        self.subquery_depth += 1;
        ControlFlow::Continue(())
    }

    fn post_visit_query(&mut self, _query: &Query) -> ControlFlow<Self::Break> {
        self.subquery_depth -= 1;
        ControlFlow::Continue(())
    }

    fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<Self::Break> {
        if self.subquery_depth > 0 {
            return ControlFlow::Continue(());
        }
        if let SQLExpr::Identifier(ident) = expr {
            if let Err(err) = check_ambiguous_column(
                &ident.value,
                self.joined_aliases,
                self.base_table_name,
                self.using_cols,
                self.relations,
            ) {
                return ControlFlow::Break(err);
            }
        }
        ControlFlow::Continue(())
    }
}

/// Check that an unqualified column name exists in at most one of the joined
/// tables. `using_cols` holds the (joined table, column) pairs of USING clauses,
/// which are coalesced with the same column on the left of that join.
pub(crate) fn check_ambiguous_column(
    col: &str,
    joined_aliases: &PlHashMap<String, PlHashMap<String, String>>,
    base_table_name: &str,
    using_cols: &[(String, String)],
    relations: &[RelationName],
) -> PolarsResult<()> {
    let mut tables: Vec<_> = joined_aliases
        .iter()
        .filter(|(t, cols)| {
            cols.contains_key(col) && !using_cols.iter().any(|(ut, uc)| ut == *t && uc == col)
        })
        .map(|(t, _)| t.as_str())
        .collect();

    if !tables.is_empty() {
        tables.push(base_table_name);
        tables.sort();
        if tables.windows(2).any(|w| w[0] == w[1]) {
            polars_bail!(
                SQLInterface: "ambiguous reference to column {} (a relation appears more than once in FROM; give each an alias)",
                maybe_quote(col, true)
            );
        }
        let col_hint = maybe_quote(col, false);
        // Quote each part of a relation's name, so `"a.b"` keeps its quotes and `s.t` has none.
        let hints = tables.iter().map(|t| {
            let qualifier = match relations.iter().find(|r| r.key() == *t) {
                Some(relation) => {
                    let parts: Vec<String> = relation
                        .parts()
                        .iter()
                        .map(|p| maybe_quote(p, false))
                        .collect();
                    parts.join(".")
                },
                None => maybe_quote(t, false),
            };
            format!("{qualifier}.{col_hint}")
        });
        polars_bail!(
            SQLInterface: "ambiguous reference to column {} (use one of: {})",
            maybe_quote(col, true), hints.collect::<Vec<_>>().join(", ")
        );
    }
    Ok(())
}

/// Check a SQL expression for unqualified references to columns that
/// exist in multiple tables (see `check_ambiguous_column`).
pub(crate) fn check_for_ambiguous_column_refs(
    expr: &SQLExpr,
    joined_aliases: &PlHashMap<String, PlHashMap<String, String>>,
    base_table_name: &str,
    using_cols: &[(String, String)],
    relations: &[RelationName],
) -> PolarsResult<()> {
    match dyn_visit::visit(
        expr,
        &mut AmbiguousColumnVisitor {
            joined_aliases,
            base_table_name,
            using_cols,
            relations,
            subquery_depth: 0,
        },
    ) {
        ControlFlow::Break(err) => Err(err),
        ControlFlow::Continue(()) => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// TableIdentifierCollector
// ---------------------------------------------------------------------------

/// Visitor that collects all table identifiers referenced in a SQL query.
#[derive(Default)]
pub(crate) struct TableIdentifierCollector {
    pub(crate) tables: Vec<String>,
    pub(crate) include_schema: bool,
}

impl TableIdentifierCollector {
    pub(crate) fn collect_from_set_expr(&mut self, set_expr: &SetExpr) {
        // Recursively collect table identifiers from SetExpr nodes
        match set_expr {
            SetExpr::Table(tbl) => {
                self.tables.extend(if self.include_schema {
                    match (&tbl.schema_name, &tbl.table_name) {
                        (Some(schema), Some(table)) => Some(format!("{schema}.{table}")),
                        (None, Some(table)) => Some(table.clone()),
                        _ => None,
                    }
                } else {
                    tbl.table_name.clone()
                });
            },
            SetExpr::SetOperation { left, right, .. } => {
                self.collect_from_set_expr(left);
                self.collect_from_set_expr(right);
            },
            SetExpr::Query(query) => self.collect_from_set_expr(&query.body),
            _ => {},
        }
    }
}

impl SQLVisitor for TableIdentifierCollector {
    type Break = ();

    fn pre_visit_query(&mut self, query: &Query) -> ControlFlow<Self::Break> {
        // Collect from SetExpr nodes in the query body
        self.collect_from_set_expr(&query.body);
        ControlFlow::Continue(())
    }

    fn pre_visit_relation(&mut self, relation: &ObjectName) -> ControlFlow<Self::Break> {
        // Table relation (eg: appearing in FROM clause)
        self.tables.extend(if self.include_schema {
            let parts: Vec<_> = relation
                .0
                .iter()
                .filter_map(|p| p.as_ident().map(|i| i.value.as_str()))
                .collect();
            (!parts.is_empty()).then(|| parts.join("."))
        } else {
            relation
                .0
                .last()
                .and_then(|p| p.as_ident())
                .map(|i| i.value.clone())
        });
        ControlFlow::Continue(())
    }
}

// ---------------------------------------------------------------------------
// CteReferenceCollector
// ---------------------------------------------------------------------------

/// Visitor that collects which of the CTEs `ctes` a query reads, as `FROM`/`TABLE` resolve
/// names: the whole name, ignoring case, unless a nested `WITH` defines the same name.
struct CteReferenceCollector<'a> {
    ctes: &'a [String],
    // The CTE names that each nested `WITH` being visited makes visible.
    scopes: Vec<Vec<String>>,
    with_stack: Vec<Option<With>>,
    found: PlHashSet<usize>,
}

impl CteReferenceCollector<'_> {
    fn reference(&mut self, name: &str) {
        let shadowed = self
            .scopes
            .iter()
            .flatten()
            .any(|n| n.eq_ignore_ascii_case(name));
        if !shadowed {
            if let Some(idx) = self.ctes.iter().position(|c| c.eq_ignore_ascii_case(name)) {
                self.found.insert(idx);
            }
        }
    }

    fn reference_set_expr_tables(&mut self, body: &SetExpr) {
        match body {
            SetExpr::SetOperation { left, right, .. } => {
                self.reference_set_expr_tables(left);
                self.reference_set_expr_tables(right);
            },
            SetExpr::Table(tbl) => {
                if let Some(table_name) = &tbl.table_name {
                    match &tbl.schema_name {
                        Some(schema_name) => self.reference(&format!("{schema_name}.{table_name}")),
                        None => self.reference(table_name),
                    }
                }
            },
            _ => {},
        }
    }
}

impl VisitorMut for CteReferenceCollector<'_> {
    type Break = ();

    fn pre_visit_query(&mut self, query: &mut Query) -> ControlFlow<()> {
        // Visit the CTE bodies here, each with the names it can see, and hide the `WITH` from
        // the generic traversal until `post_visit_query`.
        let mut with = query.with.take();
        let mut names = Vec::new();
        if let Some(with) = with.as_mut() {
            let all: Vec<String> = with
                .cte_tables
                .iter()
                .map(|cte| cte.alias.name.value.clone())
                .collect();
            for (idx, cte) in with.cte_tables.iter_mut().enumerate() {
                let visible = if with.recursive {
                    all.clone()
                } else {
                    all[..idx].to_vec()
                };
                self.scopes.push(visible);
                let _ = dyn_visit::visit_mut(&mut *cte.query, self);
                self.scopes.pop();
            }
            names = all;
        }
        self.scopes.push(names);
        self.with_stack.push(with);
        self.reference_set_expr_tables(&query.body);

        // The CTEs of a parenthesized query stay visible to the clauses wrapping it.
        let mut inner_names = Vec::new();
        let mut body = &*query.body;
        while let SetExpr::Query(inner) = body {
            if let Some(inner_with) = &inner.with {
                inner_names.extend(
                    inner_with
                        .cte_tables
                        .iter()
                        .map(|cte| cte.alias.name.value.clone()),
                );
            }
            body = &inner.body;
        }
        if !inner_names.is_empty() {
            self.scopes.push(inner_names);
            let _ = dyn_visit::visit_mut(&mut query.order_by, self);
            let _ = dyn_visit::visit_mut(&mut query.limit_clause, self);
            let _ = dyn_visit::visit_mut(&mut query.fetch, self);
            self.scopes.pop();
            // Already visited; the visitor works on a copy.
            query.order_by = None;
            query.limit_clause = None;
            query.fetch = None;
        }
        ControlFlow::Continue(())
    }

    fn post_visit_query(&mut self, query: &mut Query) -> ControlFlow<()> {
        self.scopes.pop();
        query.with = self.with_stack.pop().flatten();
        ControlFlow::Continue(())
    }

    fn pre_visit_table_factor(&mut self, factor: &mut TableFactor) -> ControlFlow<()> {
        // A name with arguments is a table function.
        if let TableFactor::Table {
            name, args: None, ..
        } = factor
        {
            self.reference(&object_name_to_string(name));
        }
        ControlFlow::Continue(())
    }
}

/// The indices of the CTEs in `ctes` that `query` reads.
pub(crate) fn referenced_ctes(query: &Query, ctes: &[String]) -> PlHashSet<usize> {
    let mut collector = CteReferenceCollector {
        ctes,
        scopes: Vec::new(),
        with_stack: Vec::new(),
        found: PlHashSet::new(),
    };
    // The visitor takes each `WITH` out while visiting, so it works on a copy.
    let _ = dyn_visit::visit_mut(&mut query.clone(), &mut collector);
    collector.found
}

// ---------------------------------------------------------------------------
// WindowFunctionFinder
// ---------------------------------------------------------------------------

/// Visitor that checks if a SQL expression contains explicit window functions.
/// Uses early-exit for efficiency when only the boolean result is needed.
struct WindowFunctionFinder;

impl SQLVisitor for WindowFunctionFinder {
    type Break = ();

    fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<()> {
        if matches!(expr, SQLExpr::Function(f) if f.over.is_some()) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

/// Check if a SQL expression contains explicit window functions.
pub(crate) fn expr_has_window_functions(expr: &SQLExpr) -> bool {
    dyn_visit::visit(expr, &mut WindowFunctionFinder).is_break()
}

// ---------------------------------------------------------------------------
// ColumnRefFinder
// ---------------------------------------------------------------------------

/// Visitor that checks if a SQL expression contains any column reference at all.
struct ColumnRefFinder;

impl SQLVisitor for ColumnRefFinder {
    type Break = ();

    fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<()> {
        let is_column_ref = match expr {
            SQLExpr::Identifier(_)
            | SQLExpr::CompoundIdentifier(_)
            | SQLExpr::Wildcard(_)
            | SQLExpr::QualifiedWildcard(..) => true,
            // wildcard arguments, eg: COUNT(*) / COLUMNS(*)
            SQLExpr::Function(func) => match &func.args {
                FunctionArguments::List(args) => args.args.iter().any(|arg| {
                    matches!(
                        arg,
                        FunctionArg::Unnamed(
                            FunctionArgExpr::Wildcard | FunctionArgExpr::QualifiedWildcard(_)
                        )
                    )
                }),
                _ => false,
            },
            _ => false,
        };
        if is_column_ref {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

/// Check if a SQL expression references any column (as opposed to being a
/// constant expression composed only of literals/operators/functions).
pub(crate) fn expr_references_any_column(expr: &SQLExpr) -> bool {
    dyn_visit::visit(expr, &mut ColumnRefFinder).is_break()
}

/// Whether this expression node is itself a subquery, in any of its forms.
pub(crate) fn is_subquery_expr(expr: &SQLExpr) -> bool {
    matches!(
        expr,
        SQLExpr::Subquery(_) | SQLExpr::Exists { .. } | SQLExpr::InSubquery { .. }
    )
}

/// Check if a SQL expression contains a subquery, in any of its forms.
pub(crate) fn expr_contains_subquery(expr: &SQLExpr) -> bool {
    expr_contains(expr, is_subquery_expr)
}

/// Check if a SQL expression contains a scalar subquery (`(SELECT ...)` used as
/// a value), as opposed to a predicate subquery (`EXISTS`, `IN`).
pub(crate) fn expr_contains_scalar_subquery(expr: &SQLExpr) -> bool {
    expr_contains(expr, |e| matches!(e, SQLExpr::Subquery(_)))
}

fn expr_contains(expr: &SQLExpr, is_match: impl Fn(&SQLExpr) -> bool) -> bool {
    visit_expressions(expr, |e| {
        if is_match(e) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })
    .is_break()
}

/// The argument expressions of a `GROUPING()` / `GROUPING_ID()` call; `None` for
/// any other function or an argument list that is not plain expressions.
pub(crate) fn grouping_call_args(func: &SQLFunction) -> Option<Vec<SQLExpr>> {
    let is_grouping = func
        .name
        .0
        .last()
        .and_then(|p| p.as_ident())
        .is_some_and(|i| {
            i.value.eq_ignore_ascii_case("grouping") || i.value.eq_ignore_ascii_case("grouping_id")
        });
    if !is_grouping {
        return None;
    }
    let FunctionArguments::List(FunctionArgumentList { args, .. }) = &func.args else {
        return None;
    };
    args.iter()
        .map(|arg| match arg {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => Some(e.clone()),
            _ => None,
        })
        .collect()
}

/// Collect the argument lists of the `GROUPING()` calls in a SQL expression,
/// leaving out those inside subqueries, which belong to their own query block.
pub(crate) fn collect_grouping_calls(expr: &SQLExpr) -> Vec<Vec<SQLExpr>> {
    struct Collector {
        subquery_depth: usize,
        calls: Vec<Vec<SQLExpr>>,
    }
    impl SQLVisitor for Collector {
        type Break = ();

        fn pre_visit_query(&mut self, _query: &Query) -> ControlFlow<()> {
            self.subquery_depth += 1;
            ControlFlow::Continue(())
        }

        fn post_visit_query(&mut self, _query: &Query) -> ControlFlow<()> {
            self.subquery_depth -= 1;
            ControlFlow::Continue(())
        }

        fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<()> {
            if self.subquery_depth == 0 {
                if let SQLExpr::Function(f) = expr {
                    if let Some(args) = grouping_call_args(f) {
                        self.calls.push(args);
                    }
                }
            }
            ControlFlow::Continue(())
        }
    }
    let mut collector = Collector {
        subquery_depth: 0,
        calls: Vec::new(),
    };
    let _ = dyn_visit::visit(expr, &mut collector);
    collector.calls
}
