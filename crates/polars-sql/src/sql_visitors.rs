//! SQLVisitor helper implementations for traversing SQL AST expressions.
//!
//! This module provides visitor implementations used throughout the SQL interface
//! to analyze and check SQL expressions for various properties.

use std::ops::ControlFlow;

use polars_core::prelude::*;
use sqlparser::ast::{Expr as SQLExpr, ObjectName, Query, SetExpr, Visit, Visitor as SQLVisitor};
use sqlparser::keywords::ALL_KEYWORDS;

// ---------------------------------------------------------------------------
// FindTableIdentifier
// ---------------------------------------------------------------------------

/// Visitor that checks if an expression tree contains a reference to a specific table.
pub(crate) struct FindTableIdentifier<'a> {
    table_name: &'a str,
    found: bool,
}

impl<'a> FindTableIdentifier<'a> {
    fn new(table_name: &'a str) -> Self {
        Self {
            table_name,
            found: false,
        }
    }
}

impl<'a> SQLVisitor for FindTableIdentifier<'a> {
    type Break = ();

    fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<Self::Break> {
        if let SQLExpr::CompoundIdentifier(idents) = expr {
            if idents.len() >= 2 && idents[0].value.as_str() == self.table_name {
                self.found = true; // return immediately on first match
                return ControlFlow::Break(());
            }
        }
        ControlFlow::Continue(())
    }
}

/// Check if a SQL expression contains a reference to a specific table.
pub(crate) fn expr_refers_to_table(expr: &SQLExpr, table_name: &str) -> bool {
    let mut table_finder = FindTableIdentifier::new(table_name);
    let _ = expr.visit(&mut table_finder);
    table_finder.found
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
        let _ = expr.visit(&mut analyzer);
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
/// multiple tables (columns appearing in a USING clause are excluded from
/// the check as they are implicitly coalesced).
struct AmbiguousColumnVisitor<'a> {
    joined_aliases: &'a PlHashMap<String, PlHashMap<String, String>>,
    base_table_name: &'a str,
    using_cols: &'a PlHashSet<String>,
}

impl SQLVisitor for AmbiguousColumnVisitor<'_> {
    type Break = PolarsError;

    fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<Self::Break> {
        if let SQLExpr::Identifier(ident) = expr {
            let col = &ident.value;
            if self.using_cols.contains(col) {
                return ControlFlow::Continue(());
            }
            let mut tables: Vec<_> = self
                .joined_aliases
                .iter()
                .filter_map(|(t, cols)| cols.contains_key(col).then_some(t.as_str()))
                .collect();

            if !tables.is_empty() {
                tables.push(self.base_table_name);
                tables.sort();
                let col_hint = maybe_quote(col, false);
                let hints = tables
                    .iter()
                    .map(|t| format!("{}.{}", maybe_quote(t, false), col_hint));
                return ControlFlow::Break(polars_err!(
                    SQLInterface: "ambiguous reference to column {} (use one of: {})",
                    maybe_quote(col, true), hints.collect::<Vec<_>>().join(", ")
                ));
            }
        }
        ControlFlow::Continue(())
    }
}

/// Check a SQL expression for unqualified references to columns that
/// exist in multiple tables (columns appearing in a USING clause are
/// excluded from the check as they are implicitly coalesced).
pub(crate) fn check_for_ambiguous_column_refs(
    expr: &SQLExpr,
    joined_aliases: &PlHashMap<String, PlHashMap<String, String>>,
    base_table_name: &str,
    using_cols: &PlHashSet<String>,
) -> PolarsResult<()> {
    match expr.visit(&mut AmbiguousColumnVisitor {
        joined_aliases,
        base_table_name,
        using_cols,
    }) {
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
    expr.visit(&mut WindowFunctionFinder).is_break()
}
