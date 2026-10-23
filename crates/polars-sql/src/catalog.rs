//! Catalogs attached to a [`SQLContext`](crate::SQLContext): resolving table names to tables
//! of a catalog.
use std::ops::ControlFlow;

use polars_core::prelude::*;
use polars_utils::aliases::PlHashMap;
use sqlparser::ast::{
    Ident, ObjectName, ObjectNamePart, Query, SetExpr, Statement, TableFactor, Visitor, VisitorMut,
    With,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;
use sqlparser::tokenizer::{Location, Span};

use crate::dyn_visit;

/// A table of an attached catalog that a statement reads.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CatalogTable {
    /// The name the catalog was attached under.
    pub catalog: String,
    /// The table's identifier in the catalog: namespace parts, then the table name.
    pub identifier: Vec<String>,
}

/// A statement changing a catalog table, for the caller to perform (see
/// [`SQLContext::catalog_ddl`](crate::SQLContext::catalog_ddl)).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CatalogDdl {
    /// `DROP TABLE [IF EXISTS] <name> [PURGE]`.
    DropTable {
        /// The table name as the statement wrote it.
        name: String,
        /// The catalog table it resolves to.
        table: CatalogTable,
        /// `IF EXISTS`.
        if_exists: bool,
        /// `PURGE`: also delete the data files.
        purge: bool,
    },
}

impl CatalogTable {
    /// The catalog name followed by the identifier, the name the table is known by in SQL.
    pub fn full_name(&self) -> Vec<String> {
        std::iter::once(self.catalog.clone())
            .chain(self.identifier.iter().cloned())
            .collect()
    }
}

#[derive(Clone, Debug)]
struct AttachedCatalog {
    name: String,
    /// The namespace of one-part table names.
    default_namespace: Option<Vec<String>>,
}

/// The catalogs attached to a context.
#[derive(Clone, Debug, Default)]
pub(crate) struct CatalogRegistry {
    catalogs: Vec<AttachedCatalog>,
    /// The catalog of table names that don't start with an attached catalog.
    default: Option<String>,
}

impl CatalogRegistry {
    pub(crate) fn is_empty(&self) -> bool {
        self.catalogs.is_empty()
    }

    /// Attach a catalog; the same name replaces it, with the new `default_namespace`, and keeps
    /// its default status (`default` only ever sets the default). A name that differs from an
    /// attached one only in case is an error, so matching names ignoring case is never ambiguous.
    pub(crate) fn register(
        &mut self,
        name: &str,
        default_namespace: Option<Vec<String>>,
        default: bool,
    ) -> PolarsResult<()> {
        polars_ensure!(!name.is_empty(), InvalidOperation: "catalog name must not be empty");
        polars_ensure!(
            default_namespace.as_ref().is_none_or(|ns| !ns.is_empty() && ns.iter().all(|p| !p.is_empty())),
            InvalidOperation: "the default namespace of catalog '{}' must have non-empty parts", name
        );
        if let Some(other) = self
            .catalogs
            .iter()
            .find(|c| c.name != name && c.name.eq_ignore_ascii_case(name))
        {
            polars_bail!(
                InvalidOperation: "catalog '{}' differs from attached catalog '{}' only in case; use that name, or unregister it first",
                name, other.name
            );
        }
        let keep_default = self.default.as_deref() == Some(name);
        self.unregister(name);
        self.catalogs.push(AttachedCatalog {
            name: name.to_string(),
            default_namespace,
        });
        if default || keep_default {
            self.default = Some(name.to_string());
        }
        Ok(())
    }

    /// The name of the default catalog.
    pub(crate) fn default_catalog(&self) -> Option<&str> {
        self.default.as_deref()
    }

    /// Make an attached catalog (exact name, else ignoring case) the default.
    pub(crate) fn set_default(&mut self, name: &str) -> PolarsResult<()> {
        let Some(name) = self.catalog_name(name).map(str::to_string) else {
            polars_bail!(SQLInterface: "catalog '{}' is not attached", name);
        };
        self.default = Some(name);
        Ok(())
    }

    /// Detach a catalog (exact name, else ignoring case), returning the name it had.
    pub(crate) fn unregister(&mut self, name: &str) -> Option<String> {
        let idx = self.position(name)?;
        let removed = self.catalogs.remove(idx).name;
        if self.default.as_deref() == Some(removed.as_str()) {
            self.default = None;
        }
        Some(removed)
    }

    fn position(&self, name: &str) -> Option<usize> {
        self.catalogs
            .iter()
            .position(|c| c.name == name)
            .or_else(|| {
                self.catalogs
                    .iter()
                    .position(|c| c.name.eq_ignore_ascii_case(name))
            })
    }

    /// The attached catalog called `name` (exact, else ignoring case).
    pub(crate) fn catalog_name(&self, name: &str) -> Option<&str> {
        self.position(name).map(|i| self.catalogs[i].name.as_str())
    }

    /// The attached catalog a table name explicitly starts with.
    pub(crate) fn named_catalog(&self, parts: &[String]) -> Option<&str> {
        match parts {
            [first, _, ..] => self.catalog_name(first),
            _ => None,
        }
    }

    /// The catalog table a table name refers to: a leading part naming an attached catalog
    /// selects it, else the default catalog; a name without namespace parts gets the catalog's
    /// default namespace. `None` if no catalog applies.
    pub(crate) fn resolve(&self, parts: &[String]) -> Option<CatalogTable> {
        let named = match parts {
            [first, rest @ ..] if !rest.is_empty() => self.position(first).map(|idx| (idx, rest)),
            _ => None,
        };
        let (idx, rest) = match named {
            Some(named) => named,
            None => (self.position(self.default.as_deref()?)?, parts),
        };
        let catalog = &self.catalogs[idx];
        let identifier = match rest {
            [table] => {
                let mut identifier = catalog.default_namespace.clone()?;
                identifier.push(table.clone());
                identifier
            },
            _ => rest.to_vec(),
        };
        Some(CatalogTable {
            catalog: catalog.name.clone(),
            identifier,
        })
    }

    /// Why a table name that nothing resolved is not a catalog table, if a catalog could have
    /// applied.
    pub(crate) fn unresolved_hint(&self, parts: &[String]) -> Option<String> {
        if self.is_empty() || self.resolve(parts).is_some() {
            return None;
        }
        let catalog = match self.named_catalog(parts) {
            Some(named) => named,
            None => match &self.default {
                Some(default) => default.as_str(),
                None => return Some("; no default catalog is set".to_string()),
            },
        };
        Some(format!("; catalog '{catalog}' has no default namespace"))
    }
}

/// The identifier parts of a table name, or `None` if a part is not an identifier.
pub(crate) fn name_parts(name: &ObjectName) -> Option<Vec<String>> {
    name.0
        .iter()
        .map(|part| part.as_ident().map(|ident| ident.value.clone()))
        .collect()
}

/// A table name as a statement wrote it, and the full name of the catalog table it refers to.
pub(crate) type RewrittenName = (Vec<String>, Vec<String>);

/// The names a statement wrote for its catalog tables, by the position of each rewritten name
/// (its first part's span start), so a table is also known by the name written there.
pub(crate) type CatalogSpellings = PlHashMap<(u64, u64), Vec<String>>;

/// The position a rewritten name is known by in `CatalogSpellings`.
pub(crate) fn name_position(name: &ObjectName) -> Option<(u64, u64)> {
    let start = name.0.first()?.as_ident()?.span.start;
    Some((start.line, start.column))
}

/// Rewrite the names of catalog tables in `stmt` to their full names, `lineitem` to
/// `onelake.tpch.lineitem`, so the rest of the translation knows them by that name. Names of
/// CTEs in scope and of local tables (`is_local`) are left alone. Returns each rewritten name
/// as written with its full name, and where each name was written.
pub(crate) fn rewrite_catalog_names(
    stmt: &mut Statement,
    resolve: impl Fn(&[String]) -> Option<Vec<String>>,
    is_local: impl Fn(&str) -> bool,
) -> PolarsResult<(Vec<RewrittenName>, CatalogSpellings)> {
    let mut registered = RegisteredAliases(Vec::new());
    let _ = dyn_visit::visit(&*stmt, &mut registered);
    let mut rewriter = Rewriter {
        resolve,
        is_local,
        registered_aliases: registered.0,
        cte_scopes: Vec::new(),
        with_stack: Vec::new(),
        rewritten: Vec::new(),
        spellings: CatalogSpellings::default(),
        desugared: 0,
    };
    if let ControlFlow::Break(err) = dyn_visit::visit_mut(stmt, &mut rewriter) {
        return Err(err);
    }
    Ok((rewriter.rewritten, rewriter.spellings))
}

/// The aliases of the statement's derived tables, table functions and `UNNEST`s, which are
/// registered as tables while the statement runs (#30070), wherever they are written.
struct RegisteredAliases(Vec<String>);

impl Visitor for RegisteredAliases {
    type Break = ();

    fn pre_visit_table_factor(&mut self, factor: &TableFactor) -> ControlFlow<()> {
        let alias = match factor {
            TableFactor::Table {
                args: Some(_),
                alias,
                ..
            }
            | TableFactor::Derived { alias, .. }
            | TableFactor::UNNEST { alias, .. } => alias.as_ref(),
            _ => None,
        };
        self.0.extend(alias.map(|alias| alias.name.value.clone()));
        ControlFlow::Continue(())
    }
}

struct Rewriter<R, L> {
    resolve: R,
    is_local: L,
    /// Aliases that tables of the same name would be looked up as (see [`RegisteredAliases`]).
    registered_aliases: Vec<String>,
    /// The CTE names visible in each enclosing query.
    cte_scopes: Vec<Vec<String>>,
    /// The `WITH` clauses of the queries being visited, taken out while their bodies are visited.
    with_stack: Vec<Option<With>>,
    rewritten: Vec<RewrittenName>,
    spellings: CatalogSpellings,
    /// The number of `TABLE x` desugared, which numbers their (made-up) name positions.
    desugared: u64,
}

impl<R, L> Rewriter<R, L>
where
    R: Fn(&[String]) -> Option<Vec<String>>,
    L: Fn(&str) -> bool,
{
    fn is_cte(&self, name: &str) -> bool {
        self.cte_scopes
            .iter()
            .flatten()
            .any(|cte| cte.eq_ignore_ascii_case(name))
    }

    /// The full name of the catalog table that `parts` refers to: not a CTE or a local table.
    fn resolve_name(&self, parts: &[String]) -> Option<Vec<String>> {
        let written = parts.join(".");
        if self.is_cte(&written) || (self.is_local)(&written) {
            return None;
        }
        (self.resolve)(parts)
    }

    /// Record `parts` as referring to the catalog table `full`. Tables are looked up by their
    /// dotted name, so it must be the catalog table's alone.
    fn record(&mut self, parts: &[String], full: &[String]) -> PolarsResult<()> {
        let key = full.join(".");
        let is_alias = self
            .registered_aliases
            .iter()
            .any(|alias| alias.eq_ignore_ascii_case(&key));
        if is_alias || self.is_cte(&key) || (self.is_local)(&key) {
            polars_bail!(
                SQLInterface: "table '{}' refers to catalog table '{}', which is also the name of a registered table, CTE or alias; rename one of them",
                display_name(parts), display_name(full)
            );
        }
        if let Some((_, other)) = self
            .rewritten
            .iter()
            .find(|(_, other)| other.join(".") == key && other != full)
        {
            polars_bail!(
                SQLInterface: "catalog tables {} and {} have the same dotted name, which SQL cannot tell apart",
                display_name(other), display_name(full)
            );
        }
        if !self.rewritten.iter().any(|(written, _)| written == parts) {
            self.rewritten.push((parts.to_vec(), full.to_vec()));
        }
        Ok(())
    }

    /// Replace `TABLE x` of a catalog table by `SELECT * FROM x`, whose table name the visit
    /// then rewrites (sqlparser's `TABLE` holds at most two name parts).
    fn desugar_table(&mut self, body: &mut SetExpr) {
        match body {
            SetExpr::SetOperation { left, right, .. } => {
                self.desugar_table(left);
                self.desugar_table(right);
            },
            SetExpr::Table(table) => {
                let parts: Vec<String> = table
                    .schema_name
                    .iter()
                    .chain(&table.table_name)
                    .cloned()
                    .collect();
                if !parts.is_empty() && self.resolve_name(&parts).is_some() {
                    // Line 0 is never a position in the query text.
                    self.desugared += 1;
                    let at = Location::new(0, self.desugared);
                    *body = select_star_from(&parts, Span::new(at, at));
                }
            },
            _ => {},
        }
    }
}

impl<R, L> VisitorMut for Rewriter<R, L>
where
    R: Fn(&[String]) -> Option<Vec<String>>,
    L: Fn(&str) -> bool,
{
    type Break = PolarsError;

    fn pre_visit_query(&mut self, query: &mut Query) -> ControlFlow<PolarsError> {
        // A non-recursive CTE is visible after its definition, as `register_ctes` registers them.
        let mut with = query.with.take();
        let mut names = Vec::new();
        if let Some(with) = with.as_mut() {
            let all: Vec<String> = with
                .cte_tables
                .iter()
                .map(|cte| cte.alias.name.value.clone())
                .collect();
            let recursive = with.recursive;
            for cte in &mut with.cte_tables {
                self.cte_scopes.push(if recursive {
                    all.clone()
                } else {
                    names.clone()
                });
                let flow = dyn_visit::visit_mut(&mut *cte.query, self);
                self.cte_scopes.pop();
                flow?;
                names.push(cte.alias.name.value.clone());
            }
        }
        self.cte_scopes.push(names);
        self.with_stack.push(with);
        self.desugar_table(&mut query.body);
        ControlFlow::Continue(())
    }

    fn post_visit_query(&mut self, query: &mut Query) -> ControlFlow<PolarsError> {
        self.cte_scopes.pop();
        query.with = self.with_stack.pop().flatten();
        ControlFlow::Continue(())
    }

    fn pre_visit_table_factor(&mut self, factor: &mut TableFactor) -> ControlFlow<PolarsError> {
        // A name with arguments is a table function.
        let TableFactor::Table {
            name, args: None, ..
        } = factor
        else {
            return ControlFlow::Continue(());
        };
        let Some(parts) = name_parts(name) else {
            return ControlFlow::Continue(());
        };
        if let Some(full) = self.resolve_name(&parts) {
            if let Err(err) = self.record(&parts, &full) {
                return ControlFlow::Break(err);
            }
            let span = name.0[0]
                .as_ident()
                .map_or_else(Span::empty, |ident| ident.span);
            if let Some(position) = name_position(name) {
                self.spellings.insert(position, parts);
            }
            *name = object_name(&full, span);
        }
        ControlFlow::Continue(())
    }
}

/// A table name whose parts all have `span`, so the name keeps the position it was written at.
fn object_name(parts: &[String], span: Span) -> ObjectName {
    let ident = |part: &String| Ident {
        value: part.clone(),
        quote_style: None,
        span,
    };
    ObjectName(
        parts
            .iter()
            .map(|part| ObjectNamePart::Identifier(ident(part)))
            .collect(),
    )
}

/// `SELECT * FROM <parts>`, with the name at `span`.
fn select_star_from(parts: &[String], span: Span) -> SetExpr {
    let mut stmts = Parser::parse_sql(&GenericDialect, "SELECT * FROM t").unwrap();
    let Statement::Query(query) = stmts.remove(0) else {
        unreachable!()
    };
    let mut body = *query.body;
    if let SetExpr::Select(select) = &mut body
        && let TableFactor::Table { name, .. } = &mut select.from[0].relation
    {
        *name = object_name(parts, span);
    }
    body
}

/// A table name for messages, with parts containing dots quoted.
pub(crate) fn display_name(parts: &[String]) -> String {
    let parts: Vec<String> = parts
        .iter()
        .map(|part| match part.contains('.') {
            true => format!("\"{part}\""),
            false => part.clone(),
        })
        .collect();
    parts.join(".")
}
