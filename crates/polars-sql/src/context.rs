use std::cell::RefCell;
use std::ops::Deref;

use polars_core::frame::row::Row;
use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_ops::frame::JoinCoalesce;
use polars_plan::dsl::function_expr::StructFunction;
use polars_plan::prelude::*;
use sqlparser::ast::{
    Distinct, ExcludeSelectItem, Expr as SQLExpr, FunctionArg, GroupByExpr, Ident, JoinConstraint,
    JoinOperator, ObjectName, ObjectType, Offset, OrderByExpr, Query, RenameSelectItem, Select,
    SelectItem, SetExpr, SetOperator, SetQuantifier, Statement, TableAlias, TableFactor,
    TableWithJoins, UnaryOperator, Value as SQLValue, Values, WildcardAdditionalOptions,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::{Parser, ParserOptions};

use crate::function_registry::{DefaultFunctionRegistry, FunctionRegistry};
use crate::sql_expr::{
    parse_sql_array, parse_sql_expr, process_join_constraint, resolve_compound_identifier,
    to_sql_interface_err,
};
use crate::table_functions::PolarsTableFunctions;

/// The SQLContext is the main entry point for executing SQL queries.
#[derive(Clone)]
pub struct SQLContext {
    pub(crate) table_map: PlHashMap<String, LazyFrame>,
    pub(crate) function_registry: Arc<dyn FunctionRegistry>,
    pub(crate) lp_arena: Arena<IR>,
    pub(crate) expr_arena: Arena<AExpr>,

    cte_map: RefCell<PlHashMap<String, LazyFrame>>,
    table_aliases: RefCell<PlHashMap<String, String>>,
    joined_aliases: RefCell<PlHashMap<String, PlHashMap<String, String>>>,
}

impl Default for SQLContext {
    fn default() -> Self {
        Self {
            function_registry: Arc::new(DefaultFunctionRegistry {}),
            table_map: Default::default(),
            cte_map: Default::default(),
            table_aliases: Default::default(),
            joined_aliases: Default::default(),
            lp_arena: Default::default(),
            expr_arena: Default::default(),
        }
    }
}

impl SQLContext {
    /// Create a new SQLContext.
    /// ```rust
    /// # use polars_sql::SQLContext;
    /// # fn main() {
    /// let ctx = SQLContext::new();
    /// # }
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the names of all registered tables, in sorted order.
    pub fn get_tables(&self) -> Vec<String> {
        let mut tables = Vec::from_iter(self.table_map.keys().cloned());
        tables.sort_unstable();
        tables
    }

    /// Register a [`LazyFrame`] as a table in the SQLContext.
    /// ```rust
    /// # use polars_sql::SQLContext;
    /// # use polars_core::prelude::*;
    /// # use polars_lazy::prelude::*;
    /// # fn main() {
    ///
    /// let mut ctx = SQLContext::new();
    /// let df = df! {
    ///    "a" =>  [1, 2, 3],
    /// }.unwrap().lazy();
    ///
    /// ctx.register("df", df);
    /// # }
    ///```
    pub fn register(&mut self, name: &str, lf: LazyFrame) {
        self.table_map.insert(name.to_owned(), lf);
    }

    /// Unregister a [`LazyFrame`] table from the [`SQLContext`].
    pub fn unregister(&mut self, name: &str) {
        self.table_map.remove(&name.to_owned());
    }

    /// Execute a SQL query, returning a [`LazyFrame`].
    /// ```rust
    /// # use polars_sql::SQLContext;
    /// # use polars_core::prelude::*;
    /// # use polars_lazy::prelude::*;
    /// # fn main() {
    ///
    /// let mut ctx = SQLContext::new();
    /// let df = df! {
    ///    "a" =>  [1, 2, 3],
    /// }
    /// .unwrap();
    ///
    /// ctx.register("df", df.clone().lazy());
    /// let sql_df = ctx.execute("SELECT * FROM df").unwrap().collect().unwrap();
    /// assert!(sql_df.equals(&df));
    /// # }
    ///```
    pub fn execute(&mut self, query: &str) -> PolarsResult<LazyFrame> {
        let mut parser = Parser::new(&GenericDialect);
        parser = parser.with_options(ParserOptions {
            trailing_commas: true,
            ..Default::default()
        });

        let ast = parser
            .try_with_sql(query)
            .map_err(to_sql_interface_err)?
            .parse_statements()
            .map_err(to_sql_interface_err)?;

        polars_ensure!(ast.len() == 1, SQLInterface: "one (and only one) statement can be parsed at a time");
        let res = self.execute_statement(ast.first().unwrap())?;

        // Ensure the result uses the proper arenas.
        // This will instantiate new arenas with a new version.
        let lp_arena = std::mem::take(&mut self.lp_arena);
        let expr_arena = std::mem::take(&mut self.expr_arena);
        res.set_cached_arena(lp_arena, expr_arena);

        // Every execution should clear the statement-level maps.
        self.cte_map.borrow_mut().clear();
        self.table_aliases.borrow_mut().clear();
        self.joined_aliases.borrow_mut().clear();

        Ok(res)
    }

    /// add a function registry to the SQLContext
    /// the registry provides the ability to add custom functions to the SQLContext
    pub fn with_function_registry(mut self, function_registry: Arc<dyn FunctionRegistry>) -> Self {
        self.function_registry = function_registry;
        self
    }

    /// Get the function registry of the SQLContext
    pub fn registry(&self) -> &Arc<dyn FunctionRegistry> {
        &self.function_registry
    }

    /// Get a mutable reference to the function registry of the SQLContext
    pub fn registry_mut(&mut self) -> &mut dyn FunctionRegistry {
        Arc::get_mut(&mut self.function_registry).unwrap()
    }
}

impl SQLContext {
    pub(crate) fn execute_statement(&mut self, stmt: &Statement) -> PolarsResult<LazyFrame> {
        let ast = stmt;
        Ok(match ast {
            Statement::Query(query) => self.execute_query(query)?,
            stmt @ Statement::ShowTables { .. } => self.execute_show_tables(stmt)?,
            stmt @ Statement::CreateTable { .. } => self.execute_create_table(stmt)?,
            stmt @ Statement::Drop {
                object_type: ObjectType::Table,
                ..
            } => self.execute_drop_table(stmt)?,
            stmt @ Statement::Explain { .. } => self.execute_explain(stmt)?,
            stmt @ Statement::Truncate { .. } => self.execute_truncate_table(stmt)?,
            _ => polars_bail!(
                SQLInterface: "statement type {:?} is not supported", ast,
            ),
        })
    }

    pub(crate) fn execute_query(&mut self, query: &Query) -> PolarsResult<LazyFrame> {
        self.register_ctes(query)?;
        self.execute_query_no_ctes(query)
    }

    pub(crate) fn execute_query_no_ctes(&mut self, query: &Query) -> PolarsResult<LazyFrame> {
        let lf = self.process_query(&query.body, query)?;
        self.process_limit_offset(lf, &query.limit, &query.offset)
    }

    pub(super) fn get_table_from_current_scope(&self, name: &str) -> Option<LazyFrame> {
        let table_name = self.table_map.get(name).cloned();
        table_name
            .or_else(|| self.cte_map.borrow().get(name).cloned())
            .or_else(|| {
                self.table_aliases
                    .borrow()
                    .get(name)
                    .and_then(|alias| self.table_map.get(alias).cloned())
            })
    }

    fn expr_or_ordinal(
        &mut self,
        e: &SQLExpr,
        exprs: &[Expr],
        selected: Option<&[Expr]>,
        schema: Option<&Schema>,
        clause: &str,
    ) -> PolarsResult<Expr> {
        match e {
            SQLExpr::UnaryOp {
                op: UnaryOperator::Minus,
                expr,
            } if matches!(**expr, SQLExpr::Value(SQLValue::Number(_, _))) => {
                if let SQLExpr::Value(SQLValue::Number(ref idx, _)) = **expr {
                    Err(polars_err!(
                    SQLSyntax:
                    "negative ordinal values are invalid for {}; found -{}",
                    clause,
                    idx
                    ))
                } else {
                    unreachable!()
                }
            },
            SQLExpr::Value(SQLValue::Number(idx, _)) => {
                // note: sql queries are 1-indexed
                let idx = idx.parse::<usize>().map_err(|_| {
                    polars_err!(
                        SQLSyntax:
                        "negative ordinal values are invalid for {}; found {}",
                        clause,
                        idx
                    )
                })?;
                // note: "selected" cols represent final projection order, so we use those for
                // ordinal resolution. "exprs" may include cols that are subsequently dropped.
                let cols = if let Some(cols) = selected {
                    cols
                } else {
                    exprs
                };
                Ok(cols
                    .get(idx - 1)
                    .ok_or_else(|| {
                        polars_err!(
                            SQLInterface:
                            "{} ordinal value must refer to a valid column; found {}",
                            clause,
                            idx
                        )
                    })?
                    .clone())
            },
            SQLExpr::Value(v) => Err(polars_err!(
                SQLSyntax:
                "{} requires a valid expression or positive ordinal; found {}", clause, v,
            )),
            _ => parse_sql_expr(e, self, schema),
        }
    }

    pub(super) fn resolve_name(&self, tbl_name: &str, column_name: &str) -> String {
        if self.joined_aliases.borrow().contains_key(tbl_name) {
            self.joined_aliases
                .borrow()
                .get(tbl_name)
                .and_then(|aliases| aliases.get(column_name))
                .cloned()
                .unwrap_or_else(|| column_name.to_string())
        } else {
            column_name.to_string()
        }
    }

    fn process_query(&mut self, expr: &SetExpr, query: &Query) -> PolarsResult<LazyFrame> {
        match expr {
            SetExpr::Select(select_stmt) => self.execute_select(select_stmt, query),
            SetExpr::Query(query) => self.execute_query_no_ctes(query),
            SetExpr::SetOperation {
                op: SetOperator::Union,
                set_quantifier,
                left,
                right,
            } => self.process_union(left, right, set_quantifier, query),

            #[cfg(feature = "semi_anti_join")]
            SetExpr::SetOperation {
                op: SetOperator::Intersect | SetOperator::Except,
                set_quantifier,
                left,
                right,
            } => self.process_except_intersect(left, right, set_quantifier, query),

            SetExpr::Values(Values {
                explicit_row: _,
                rows,
            }) => self.process_values(rows),

            SetExpr::Table(tbl) => {
                if tbl.table_name.is_some() {
                    let table_name = tbl.table_name.as_ref().unwrap();
                    self.get_table_from_current_scope(table_name)
                        .ok_or_else(|| {
                            polars_err!(
                                SQLInterface: "no table or alias named '{}' found",
                                tbl
                            )
                        })
                } else {
                    polars_bail!(SQLInterface: "'TABLE' requires valid table name")
                }
            },
            op => {
                let op = match op {
                    SetExpr::SetOperation { op, .. } => op,
                    _ => unreachable!(),
                };
                polars_bail!(SQLInterface: "'{}' operation is currently unsupported", op)
            },
        }
    }

    #[cfg(feature = "semi_anti_join")]
    fn process_except_intersect(
        &mut self,
        left: &SetExpr,
        right: &SetExpr,
        quantifier: &SetQuantifier,
        query: &Query,
    ) -> PolarsResult<LazyFrame> {
        let (join_type, op_name) = match *query.body {
            SetExpr::SetOperation {
                op: SetOperator::Except,
                ..
            } => (JoinType::Anti, "EXCEPT"),
            _ => (JoinType::Semi, "INTERSECT"),
        };
        let mut lf = self.process_query(left, query)?;
        let mut rf = self.process_query(right, query)?;
        let join = lf
            .clone()
            .join_builder()
            .with(rf.clone())
            .how(join_type)
            .join_nulls(true);

        let lf_schema = lf.schema_with_arenas(&mut self.lp_arena, &mut self.expr_arena)?;
        let lf_cols: Vec<_> = lf_schema.iter_names().map(|nm| col(nm)).collect();
        let joined_tbl = match quantifier {
            SetQuantifier::ByName | SetQuantifier::AllByName => {
                // note: 'BY NAME' is pending https://github.com/sqlparser-rs/sqlparser-rs/pull/1309
                join.on(lf_cols).finish()
            },
            SetQuantifier::Distinct | SetQuantifier::None => {
                let rf_schema = rf.schema_with_arenas(&mut self.lp_arena, &mut self.expr_arena)?;
                let rf_cols: Vec<_> = rf_schema.iter_names().map(|nm| col(nm)).collect();
                if lf_cols.len() != rf_cols.len() {
                    polars_bail!(SQLInterface: "{} requires equal number of columns in each table (use '{} BY NAME' to combine mismatched tables)", op_name, op_name)
                }
                join.left_on(lf_cols).right_on(rf_cols).finish()
            },
            _ => {
                polars_bail!(SQLInterface: "'{} {}' is not supported", op_name, quantifier.to_string())
            },
        };
        Ok(joined_tbl.unique(None, UniqueKeepStrategy::Any))
    }

    fn process_union(
        &mut self,
        left: &SetExpr,
        right: &SetExpr,
        quantifier: &SetQuantifier,
        query: &Query,
    ) -> PolarsResult<LazyFrame> {
        let mut lf = self.process_query(left, query)?;
        let mut rf = self.process_query(right, query)?;
        let opts = UnionArgs {
            parallel: true,
            to_supertypes: true,
            ..Default::default()
        };
        match quantifier {
            // UNION [ALL | DISTINCT]
            SetQuantifier::All | SetQuantifier::Distinct | SetQuantifier::None => {
                let lf_schema = lf.schema_with_arenas(&mut self.lp_arena, &mut self.expr_arena)?;
                let rf_schema = rf.schema_with_arenas(&mut self.lp_arena, &mut self.expr_arena)?;
                if lf_schema.len() != rf_schema.len() {
                    polars_bail!(SQLInterface: "UNION requires equal number of columns in each table (use 'UNION BY NAME' to combine mismatched tables)")
                }
                let concatenated = polars_lazy::dsl::concat(vec![lf, rf], opts);
                match quantifier {
                    SetQuantifier::Distinct | SetQuantifier::None => {
                        concatenated.map(|lf| lf.unique(None, UniqueKeepStrategy::Any))
                    },
                    _ => concatenated,
                }
            },
            // UNION ALL BY NAME
            #[cfg(feature = "diagonal_concat")]
            SetQuantifier::AllByName => concat_lf_diagonal(vec![lf, rf], opts),
            // UNION [DISTINCT] BY NAME
            #[cfg(feature = "diagonal_concat")]
            SetQuantifier::ByName | SetQuantifier::DistinctByName => {
                let concatenated = concat_lf_diagonal(vec![lf, rf], opts);
                concatenated.map(|lf| lf.unique(None, UniqueKeepStrategy::Any))
            },
            #[allow(unreachable_patterns)]
            _ => polars_bail!(SQLInterface: "'UNION {}' is not currently supported", quantifier),
        }
    }

    fn process_values(&mut self, values: &[Vec<SQLExpr>]) -> PolarsResult<LazyFrame> {
        let frame_rows: Vec<Row> = values.iter().map(|row| {
            let row_data: Result<Vec<_>, _> = row.iter().map(|expr| {
                let expr = parse_sql_expr(expr, self, None)?;
                match expr {
                    Expr::Literal(value) => {
                        value.to_any_value()
                            .ok_or_else(|| polars_err!(SQLInterface: "invalid literal value: {:?}", value))
                            .map(|av| av.into_static().unwrap())
                    },
                    _ => polars_bail!(SQLInterface: "VALUES clause expects literals; found {}", expr),
                }
            }).collect();
            row_data.map(Row::new)
        }).collect::<Result<_, _>>()?;

        Ok(DataFrame::from_rows(frame_rows.as_ref())?.lazy())
    }

    // EXPLAIN SELECT * FROM DF
    fn execute_explain(&mut self, stmt: &Statement) -> PolarsResult<LazyFrame> {
        match stmt {
            Statement::Explain { statement, .. } => {
                let lf = self.execute_statement(statement)?;
                let plan = lf.describe_optimized_plan()?;
                let plan = plan
                    .split('\n')
                    .collect::<Series>()
                    .with_name("Logical Plan");
                let df = DataFrame::new(vec![plan])?;
                Ok(df.lazy())
            },
            _ => unreachable!(),
        }
    }

    // SHOW TABLES
    fn execute_show_tables(&mut self, _: &Statement) -> PolarsResult<LazyFrame> {
        let tables = Series::new("name", self.get_tables());
        let df = DataFrame::new(vec![tables])?;
        Ok(df.lazy())
    }

    fn execute_drop_table(&mut self, stmt: &Statement) -> PolarsResult<LazyFrame> {
        match stmt {
            Statement::Drop { names, .. } => {
                names.iter().for_each(|name| {
                    self.table_map.remove(&name.to_string());
                });
                Ok(DataFrame::empty().lazy())
            },
            _ => unreachable!(),
        }
    }

    fn execute_truncate_table(&mut self, stmt: &Statement) -> PolarsResult<LazyFrame> {
        if let Statement::Truncate {
            table_name,
            partitions,
            ..
        } = stmt
        {
            match partitions {
                None => {
                    let tbl = table_name.to_string();
                    if let Some(lf) = self.table_map.get_mut(&tbl) {
                        *lf = DataFrame::from(
                            lf.schema_with_arenas(&mut self.lp_arena, &mut self.expr_arena)
                                .unwrap()
                                .as_ref(),
                        )
                        .lazy();
                        Ok(lf.clone())
                    } else {
                        polars_bail!(SQLInterface: "table '{}' does not exist", tbl);
                    }
                },
                _ => {
                    polars_bail!(SQLInterface: "TRUNCATE does not support use of 'partitions'")
                },
            }
        } else {
            unreachable!()
        }
    }

    fn register_cte(&mut self, name: &str, lf: LazyFrame) {
        self.cte_map.borrow_mut().insert(name.to_owned(), lf);
    }

    fn register_ctes(&mut self, query: &Query) -> PolarsResult<()> {
        if let Some(with) = &query.with {
            if with.recursive {
                polars_bail!(SQLInterface: "recursive CTEs are not supported")
            }
            for cte in &with.cte_tables {
                let cte_name = cte.alias.name.value.clone();
                let mut lf = self.execute_query(&cte.query)?;
                lf = self.rename_columns_from_table_alias(lf, &cte.alias)?;
                self.register_cte(&cte_name, lf);
            }
        }
        Ok(())
    }

    /// execute the 'FROM' part of the query
    fn execute_from_statement(&mut self, tbl_expr: &TableWithJoins) -> PolarsResult<LazyFrame> {
        let (l_name, mut lf) = self.get_table(&tbl_expr.relation)?;
        if !tbl_expr.joins.is_empty() {
            for tbl in &tbl_expr.joins {
                let (r_name, mut rf) = self.get_table(&tbl.relation)?;
                let left_schema =
                    lf.schema_with_arenas(&mut self.lp_arena, &mut self.expr_arena)?;
                let right_schema =
                    rf.schema_with_arenas(&mut self.lp_arena, &mut self.expr_arena)?;

                lf = match &tbl.join_operator {
                    JoinOperator::FullOuter(constraint) => {
                        self.process_join(lf, rf, constraint, &l_name, &r_name, JoinType::Full)?
                    },
                    JoinOperator::Inner(constraint) => {
                        self.process_join(lf, rf, constraint, &l_name, &r_name, JoinType::Inner)?
                    },
                    JoinOperator::LeftOuter(constraint) => {
                        self.process_join(lf, rf, constraint, &l_name, &r_name, JoinType::Left)?
                    },
                    #[cfg(feature = "semi_anti_join")]
                    JoinOperator::LeftAnti(constraint) => {
                        self.process_join(lf, rf, constraint, &l_name, &r_name, JoinType::Anti)?
                    },
                    #[cfg(feature = "semi_anti_join")]
                    JoinOperator::LeftSemi(constraint) => {
                        self.process_join(lf, rf, constraint, &l_name, &r_name, JoinType::Semi)?
                    },
                    #[cfg(feature = "semi_anti_join")]
                    JoinOperator::RightAnti(constraint) => {
                        self.process_join(rf, lf, constraint, &l_name, &r_name, JoinType::Anti)?
                    },
                    #[cfg(feature = "semi_anti_join")]
                    JoinOperator::RightSemi(constraint) => {
                        self.process_join(rf, lf, constraint, &l_name, &r_name, JoinType::Semi)?
                    },
                    JoinOperator::CrossJoin => lf.cross_join(rf, Some(format!(":{}", r_name))),
                    join_type => {
                        polars_bail!(
                            SQLInterface:
                            "join type '{:?}' not currently supported", join_type
                        );
                    },
                };

                // track join-aliased columns so we can resolve them later
                let joined_schema =
                    lf.schema_with_arenas(&mut self.lp_arena, &mut self.expr_arena)?;
                self.joined_aliases.borrow_mut().insert(
                    r_name.to_string(),
                    right_schema
                        .iter_names()
                        .filter_map(|name| {
                            // col exists in both tables and is aliased in the joined result
                            let aliased_name = format!("{}:{}", name, r_name);
                            if left_schema.contains(name)
                                && joined_schema.contains(aliased_name.as_str())
                            {
                                Some((name.to_string(), aliased_name))
                            } else {
                                None
                            }
                        })
                        .collect::<PlHashMap<String, String>>(),
                );
            }
        };
        Ok(lf)
    }

    /// Execute the 'SELECT' part of the query.
    fn execute_select(&mut self, select_stmt: &Select, query: &Query) -> PolarsResult<LazyFrame> {
        let mut lf = if select_stmt.from.is_empty() {
            DataFrame::empty().lazy()
        } else {
            // Note: implicit joins need more work to support properly,
            // explicit joins are preferred for now (ref: #16662)
            let from = select_stmt.clone().from;
            if from.len() > 1 {
                polars_bail!(SQLInterface: "multiple tables in FROM clause are not currently supported (found {}); use explicit JOIN syntax instead", from.len())
            }
            self.execute_from_statement(from.first().unwrap())?
        };

        // Filter expression (WHERE clause)
        let schema = lf.schema_with_arenas(&mut self.lp_arena, &mut self.expr_arena)?;
        lf = self.process_where(lf, &select_stmt.selection)?;

        // 'SELECT *' modifiers
        let mut excluded_cols = vec![];
        let mut replace_exprs = vec![];
        let mut rename_cols = (&mut vec![], &mut vec![]);

        // Column projections (SELECT clause)
        let projections: Vec<Expr> = select_stmt
            .projection
            .iter()
            .map(|select_item| {
                Ok(match select_item {
                    SelectItem::UnnamedExpr(expr) => {
                        vec![parse_sql_expr(expr, self, Some(schema.deref()))?]
                    },
                    SelectItem::ExprWithAlias { expr, alias } => {
                        let expr = parse_sql_expr(expr, self, Some(schema.deref()))?;
                        vec![expr.alias(&alias.value)]
                    },
                    SelectItem::QualifiedWildcard(obj_name, wildcard_options) => self
                        .process_qualified_wildcard(
                            obj_name,
                            wildcard_options,
                            &mut excluded_cols,
                            &mut rename_cols,
                            &mut replace_exprs,
                            Some(schema.deref()),
                        )?,
                    SelectItem::Wildcard(wildcard_options) => {
                        let cols = schema
                            .iter_names()
                            .map(|name| col(name))
                            .collect::<Vec<_>>();

                        self.process_wildcard_additional_options(
                            cols,
                            wildcard_options,
                            &mut excluded_cols,
                            &mut rename_cols,
                            &mut replace_exprs,
                            Some(schema.deref()),
                        )?
                    },
                })
            })
            .collect::<PolarsResult<Vec<Vec<_>>>>()?
            .into_iter()
            .flatten()
            .collect();

        // Check for "GROUP BY ..." (after determining projections)
        let mut group_by_keys: Vec<Expr> = Vec::new();
        match &select_stmt.group_by {
            // Standard "GROUP BY x, y, z" syntax (also recognising ordinal values)
            GroupByExpr::Expressions(group_by_exprs) => {
                // translate the group expressions, allowing ordinal values
                group_by_keys = group_by_exprs
                    .iter()
                    .map(|e| {
                        self.expr_or_ordinal(
                            e,
                            &projections,
                            None,
                            Some(schema.deref()),
                            "GROUP BY",
                        )
                    })
                    .collect::<PolarsResult<_>>()?
            },
            // "GROUP BY ALL" syntax; automatically adds expressions that do not contain
            // nested agg/window funcs to the group key (also ignores literals).
            GroupByExpr::All => {
                projections.iter().for_each(|expr| match expr {
                    // immediately match the most common cases (col|agg|len|lit, optionally aliased).
                    Expr::Agg(_) | Expr::Len | Expr::Literal(_) => (),
                    Expr::Column(_) => group_by_keys.push(expr.clone()),
                    Expr::Alias(e, _)
                        if matches!(&**e, Expr::Agg(_) | Expr::Len | Expr::Literal(_)) => {},
                    Expr::Alias(e, _) if matches!(&**e, Expr::Column(_)) => {
                        if let Expr::Column(name) = &**e {
                            group_by_keys.push(col(name));
                        }
                    },
                    _ => {
                        // If not quick-matched, add if no nested agg/window expressions
                        if !has_expr(expr, |e| {
                            matches!(e, Expr::Agg(_))
                                || matches!(e, Expr::Len)
                                || matches!(e, Expr::Window { .. })
                        }) {
                            group_by_keys.push(expr.clone())
                        }
                    },
                });
            },
        };

        lf = if group_by_keys.is_empty() {
            lf = if query.order_by.is_empty() {
                // No sort, select cols as given
                lf.select(projections)
            } else {
                // Add projections to the base frame as any of the
                // original columns may be required for the sort
                lf = lf.with_columns(projections.clone());

                // Final/selected cols (also ensures accurate ordinal position refs)
                let retained_cols = projections
                    .iter()
                    .map(|e| {
                        col(e
                            .to_field(schema.deref(), Context::Default)
                            .unwrap()
                            .name
                            .as_str())
                    })
                    .collect::<Vec<_>>();

                lf = self.process_order_by(lf, &query.order_by, Some(&retained_cols))?;
                lf.select(retained_cols)
            };
            // Discard any excluded cols
            if !excluded_cols.is_empty() {
                lf.drop(excluded_cols)
            } else {
                lf
            }
        } else {
            lf = self.process_group_by(lf, &group_by_keys, &projections)?;
            lf = self.process_order_by(lf, &query.order_by, None)?;

            // Apply optional 'having' clause, post-aggregation.
            let schema = Some(lf.schema_with_arenas(&mut self.lp_arena, &mut self.expr_arena)?);
            match select_stmt.having.as_ref() {
                Some(expr) => lf.filter(parse_sql_expr(expr, self, schema.as_deref())?),
                None => lf,
            }
        };

        // Apply optional DISTINCT clause.
        lf = match &select_stmt.distinct {
            Some(Distinct::Distinct) => lf.unique_stable(None, UniqueKeepStrategy::Any),
            Some(Distinct::On(exprs)) => {
                // TODO: support exprs in `unique` see https://github.com/pola-rs/polars/issues/5760
                let schema = Some(lf.schema_with_arenas(&mut self.lp_arena, &mut self.expr_arena)?);
                let cols = exprs
                    .iter()
                    .map(|e| {
                        let expr = parse_sql_expr(e, self, schema.as_deref())?;
                        if let Expr::Column(name) = expr {
                            Ok(name.to_string())
                        } else {
                            Err(polars_err!(SQLSyntax:"DISTINCT ON only supports column names"))
                        }
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;

                // DISTINCT ON has to apply the ORDER BY before the operation.
                if !query.order_by.is_empty() {
                    lf = self.process_order_by(lf, &query.order_by, None)?;
                }
                return Ok(lf.unique_stable(Some(cols), UniqueKeepStrategy::First));
            },
            None => lf,
        };

        // Apply final 'SELECT *' modifiers
        if !replace_exprs.is_empty() {
            lf = lf.with_columns(replace_exprs);
        }
        if !rename_cols.0.is_empty() {
            lf = lf.rename(rename_cols.0, rename_cols.1);
        }
        Ok(lf)
    }

    fn process_where(
        &mut self,
        mut lf: LazyFrame,
        expr: &Option<SQLExpr>,
    ) -> PolarsResult<LazyFrame> {
        if let Some(expr) = expr {
            let schema = Some(lf.schema_with_arenas(&mut self.lp_arena, &mut self.expr_arena)?);
            let mut filter_expression = parse_sql_expr(expr, self, schema.as_deref())?;
            lf = self.process_subqueries(lf, vec![&mut filter_expression]);
            lf = lf.filter(filter_expression);
        }
        Ok(lf)
    }

    pub(super) fn process_join(
        &self,
        left_tbl: LazyFrame,
        right_tbl: LazyFrame,
        constraint: &JoinConstraint,
        tbl_name: &str,
        join_tbl_name: &str,
        join_type: JoinType,
    ) -> PolarsResult<LazyFrame> {
        let (left_on, right_on) = process_join_constraint(constraint, tbl_name, join_tbl_name)?;

        let joined_tbl = left_tbl
            .clone()
            .join_builder()
            .with(right_tbl.clone())
            .left_on(left_on)
            .right_on(right_on)
            .how(join_type)
            .suffix(format!(":{}", join_tbl_name))
            .coalesce(JoinCoalesce::KeepColumns)
            .finish();

        Ok(joined_tbl)
    }

    fn process_subqueries(&self, lf: LazyFrame, exprs: Vec<&mut Expr>) -> LazyFrame {
        let mut contexts = vec![];
        for expr in exprs {
            *expr = expr.clone().map_expr(|e| match e {
                Expr::SubPlan(lp, names) => {
                    contexts.push(<LazyFrame>::from((**lp).clone()));
                    if names.len() == 1 {
                        Expr::Column(names[0].as_str().into())
                    } else {
                        Expr::SubPlan(lp, names)
                    }
                },
                e => e,
            })
        }

        if contexts.is_empty() {
            lf
        } else {
            lf.with_context(contexts)
        }
    }

    fn execute_create_table(&mut self, stmt: &Statement) -> PolarsResult<LazyFrame> {
        if let Statement::CreateTable {
            if_not_exists,
            name,
            query,
            ..
        } = stmt
        {
            let tbl_name = name.0.first().unwrap().value.as_str();
            // CREATE TABLE IF NOT EXISTS
            if *if_not_exists && self.table_map.contains_key(tbl_name) {
                polars_bail!(SQLInterface: "relation '{}' already exists", tbl_name);
                // CREATE OR REPLACE TABLE
            }
            if let Some(query) = query {
                let lf = self.execute_query(query)?;
                self.register(tbl_name, lf);
                let out = df! {
                    "Response" => ["CREATE TABLE"]
                }
                .unwrap()
                .lazy();
                Ok(out)
            } else {
                polars_bail!(SQLInterface: "only `CREATE TABLE AS SELECT ...` is currently supported");
            }
        } else {
            unreachable!()
        }
    }

    fn get_table(&mut self, relation: &TableFactor) -> PolarsResult<(String, LazyFrame)> {
        match relation {
            TableFactor::Table {
                name, alias, args, ..
            } => {
                if let Some(args) = args {
                    return self.execute_table_function(name, alias, args);
                }
                let tbl_name = name.0.first().unwrap().value.as_str();
                if let Some(lf) = self.get_table_from_current_scope(tbl_name) {
                    match alias {
                        Some(alias) => {
                            self.table_aliases
                                .borrow_mut()
                                .insert(alias.name.value.clone(), tbl_name.to_string());
                            Ok((alias.to_string(), lf))
                        },
                        None => Ok((tbl_name.to_string(), lf)),
                    }
                } else {
                    polars_bail!(SQLInterface: "relation '{}' was not found", tbl_name);
                }
            },
            TableFactor::Derived {
                lateral,
                subquery,
                alias,
            } => {
                polars_ensure!(!(*lateral), SQLInterface: "LATERAL not supported");
                if let Some(alias) = alias {
                    let mut lf = self.execute_query_no_ctes(subquery)?;
                    lf = self.rename_columns_from_table_alias(lf, alias)?;
                    self.table_map.insert(alias.name.value.clone(), lf.clone());
                    Ok((alias.name.value.clone(), lf))
                } else {
                    polars_bail!(SQLSyntax: "derived tables must have aliases");
                }
            },
            TableFactor::UNNEST {
                alias,
                array_exprs,
                with_offset,
                with_offset_alias: _,
            } => {
                if let Some(alias) = alias {
                    let table_name = alias.name.value.clone();
                    let column_names: Vec<Option<&str>> = alias
                        .columns
                        .iter()
                        .map(|c| {
                            if c.value.is_empty() {
                                None
                            } else {
                                Some(c.value.as_str())
                            }
                        })
                        .collect();

                    let column_values: Vec<Series> = array_exprs
                        .iter()
                        .map(|arr| parse_sql_array(arr, self))
                        .collect::<Result<_, _>>()?;

                    polars_ensure!(!column_names.is_empty(),
                        SQLSyntax:
                        "UNNEST table alias must also declare column names, eg: {} (a,b,c)", alias.name.to_string()
                    );
                    if column_names.len() != column_values.len() {
                        let plural = if column_values.len() > 1 { "s" } else { "" };
                        polars_bail!(
                            SQLSyntax:
                            "UNNEST table alias requires {} column name{}, found {}", column_values.len(), plural, column_names.len()
                        );
                    }
                    let column_series: Vec<Series> = column_values
                        .iter()
                        .zip(column_names.iter())
                        .map(|(s, name)| {
                            if let Some(name) = name {
                                s.clone().with_name(name)
                            } else {
                                s.clone()
                            }
                        })
                        .collect();

                    let lf = DataFrame::new(column_series)?.lazy();
                    if *with_offset {
                        // TODO: make a PR to `sqlparser-rs` to support 'ORDINALITY'
                        //  (note that 'OFFSET' is BigQuery-specific syntax, not PostgreSQL)
                        polars_bail!(SQLInterface: "UNNEST tables do not (yet) support WITH OFFSET/ORDINALITY");
                    }
                    self.table_map.insert(table_name.clone(), lf.clone());
                    Ok((table_name.clone(), lf))
                } else {
                    polars_bail!(SQLSyntax: "UNNEST table must have an alias");
                }
            },
            // Support bare table, optionally with an alias, for now
            _ => polars_bail!(SQLInterface: "not yet implemented: {}", relation),
        }
    }

    fn execute_table_function(
        &mut self,
        name: &ObjectName,
        alias: &Option<TableAlias>,
        args: &[FunctionArg],
    ) -> PolarsResult<(String, LazyFrame)> {
        let tbl_fn = name.0.first().unwrap().value.as_str();
        let read_fn = tbl_fn.parse::<PolarsTableFunctions>()?;
        let (tbl_name, lf) = read_fn.execute(args)?;
        #[allow(clippy::useless_asref)]
        let tbl_name = alias
            .as_ref()
            .map(|a| a.name.value.clone())
            .unwrap_or_else(|| tbl_name);

        self.table_map.insert(tbl_name.clone(), lf.clone());
        Ok((tbl_name, lf))
    }

    fn process_order_by(
        &mut self,
        mut lf: LazyFrame,
        order_by: &[OrderByExpr],
        selected: Option<&[Expr]>,
    ) -> PolarsResult<LazyFrame> {
        let mut by = Vec::with_capacity(order_by.len());
        let mut descending = Vec::with_capacity(order_by.len());
        let mut nulls_last = Vec::with_capacity(order_by.len());

        let schema = Some(lf.schema_with_arenas(&mut self.lp_arena, &mut self.expr_arena)?);
        let columns = schema
            .clone()
            .unwrap()
            .iter_names()
            .map(|e| col(e))
            .collect::<Vec<_>>();

        for ob in order_by {
            // note: if not specified 'NULLS FIRST' is default for DESC, 'NULLS LAST' otherwise
            // https://www.postgresql.org/docs/current/queries-order.html
            let desc_order = !ob.asc.unwrap_or(true);
            nulls_last.push(!ob.nulls_first.unwrap_or(desc_order));
            descending.push(desc_order);

            // translate order expression, allowing ordinal values
            by.push(self.expr_or_ordinal(
                &ob.expr,
                &columns,
                selected,
                schema.as_deref(),
                "ORDER BY",
            )?)
        }
        Ok(lf.sort_by_exprs(
            &by,
            SortMultipleOptions::default()
                .with_order_descending_multi(descending)
                .with_nulls_last_multi(nulls_last)
                .with_maintain_order(true),
        ))
    }

    fn process_group_by(
        &mut self,
        mut lf: LazyFrame,
        group_by_keys: &[Expr],
        projections: &[Expr],
    ) -> PolarsResult<LazyFrame> {
        let schema_before = lf.schema_with_arenas(&mut self.lp_arena, &mut self.expr_arena)?;
        let group_by_keys_schema =
            expressions_to_schema(group_by_keys, &schema_before, Context::Default)?;

        // Remove the group_by keys as polars adds those implicitly.
        let mut aggregation_projection = Vec::with_capacity(projections.len());
        let mut projection_overrides = PlHashMap::with_capacity(projections.len());
        let mut projection_aliases = PlHashSet::new();
        let mut group_key_aliases = PlHashSet::new();

        for mut e in projections {
            // `Len` represents COUNT(*) so we treat as an aggregation here.
            let is_agg_or_window = has_expr(e, |e| {
                matches!(e, Expr::Agg(_) | Expr::Len | Expr::Window { .. })
            });

            // Note: if simple aliased expression we defer aliasing until after the group_by.
            if let Expr::Alias(expr, alias) = e {
                if e.clone().meta().is_simple_projection() {
                    group_key_aliases.insert(alias.as_ref());
                    e = expr
                } else if let Expr::Function {
                    function: FunctionExpr::StructExpr(StructFunction::FieldByName(name)),
                    ..
                } = expr.deref()
                {
                    projection_overrides.insert(alias.as_ref(), col(name).alias(alias));
                } else if !is_agg_or_window && !group_by_keys_schema.contains(alias) {
                    projection_aliases.insert(alias.as_ref());
                }
            }
            let field = e.to_field(&schema_before, Context::Default)?;
            if group_by_keys_schema.get(&field.name).is_none() && is_agg_or_window {
                let mut e = e.clone();
                if let Expr::Agg(AggExpr::Implode(expr)) = &e {
                    e = (**expr).clone();
                } else if let Expr::Alias(expr, name) = &e {
                    if let Expr::Agg(AggExpr::Implode(expr)) = expr.as_ref() {
                        e = (**expr).clone().alias(name.as_ref());
                    }
                }
                aggregation_projection.push(e);
            } else if let Expr::Column(_)
            | Expr::Function {
                function: FunctionExpr::StructExpr(StructFunction::FieldByName(_)),
                ..
            } = e
            {
                // Non-aggregated columns must be part of the GROUP BY clause
                if !group_by_keys_schema.contains(&field.name) {
                    polars_bail!(SQLSyntax: "'{}' should participate in the GROUP BY clause or an aggregate function", &field.name);
                }
            }
        }
        let aggregated = lf.group_by(group_by_keys).agg(&aggregation_projection);
        let projection_schema =
            expressions_to_schema(projections, &schema_before, Context::Default)?;

        // A final projection to get the proper order and any deferred transforms/aliases.
        let final_projection = projection_schema
            .iter_names()
            .zip(projections)
            .map(|(name, projection_expr)| {
                if let Some(expr) = projection_overrides.get(name.as_str()) {
                    expr.clone()
                } else if group_by_keys_schema.get(name).is_some()
                    || projection_aliases.contains(name.as_str())
                    || group_key_aliases.contains(name.as_str())
                {
                    projection_expr.clone()
                } else {
                    col(name)
                }
            })
            .collect::<Vec<_>>();

        Ok(aggregated.select(&final_projection))
    }

    fn process_limit_offset(
        &self,
        lf: LazyFrame,
        limit: &Option<SQLExpr>,
        offset: &Option<Offset>,
    ) -> PolarsResult<LazyFrame> {
        match (offset, limit) {
            (
                Some(Offset {
                    value: SQLExpr::Value(SQLValue::Number(offset, _)),
                    ..
                }),
                Some(SQLExpr::Value(SQLValue::Number(limit, _))),
            ) => Ok(lf.slice(
                offset
                    .parse()
                    .map_err(|e| polars_err!(SQLInterface: "OFFSET conversion error: {}", e))?,
                limit
                    .parse()
                    .map_err(|e| polars_err!(SQLInterface: "LIMIT conversion error: {}", e))?,
            )),
            (
                Some(Offset {
                    value: SQLExpr::Value(SQLValue::Number(offset, _)),
                    ..
                }),
                None,
            ) => Ok(lf.slice(
                offset
                    .parse()
                    .map_err(|e| polars_err!(SQLInterface: "OFFSET conversion error: {}", e))?,
                IdxSize::MAX,
            )),
            (None, Some(SQLExpr::Value(SQLValue::Number(limit, _)))) => Ok(lf.limit(
                limit
                    .parse()
                    .map_err(|e| polars_err!(SQLInterface: "LIMIT conversion error: {}", e))?,
            )),
            (None, None) => Ok(lf),
            _ => polars_bail!(
                SQLSyntax: "non-numeric arguments for LIMIT/OFFSET are not supported",
            ),
        }
    }

    fn process_qualified_wildcard(
        &mut self,
        ObjectName(idents): &ObjectName,
        options: &WildcardAdditionalOptions,
        excluded_cols: &mut Vec<String>,
        rename_cols: &mut (&mut Vec<String>, &mut Vec<String>),
        replace_exprs: &mut Vec<Expr>,
        schema: Option<&Schema>,
    ) -> PolarsResult<Vec<Expr>> {
        let mut new_idents = idents.clone();
        new_idents.push(Ident::new("*"));

        let expr = resolve_compound_identifier(self, new_idents.deref(), schema);
        self.process_wildcard_additional_options(
            expr?,
            options,
            excluded_cols,
            rename_cols,
            replace_exprs,
            schema,
        )
    }

    fn process_wildcard_additional_options(
        &mut self,
        exprs: Vec<Expr>,
        options: &WildcardAdditionalOptions,
        excluded_cols: &mut Vec<String>,
        rename_cols: &mut (&mut Vec<String>, &mut Vec<String>),
        replace_exprs: &mut Vec<Expr>,
        schema: Option<&Schema>,
    ) -> PolarsResult<Vec<Expr>> {
        // bail on (currently) unsupported wildcard options
        if options.opt_except.is_some() {
            polars_bail!(SQLInterface: "EXCEPT wildcard option is unsupported (use EXCLUDE instead)")
        } else if options.opt_ilike.is_some() {
            polars_bail!(SQLInterface: "ILIKE wildcard option is currently unsupported")
        } else if options.opt_rename.is_some() && options.opt_replace.is_some() {
            // pending an upstream fix: https://github.com/sqlparser-rs/sqlparser-rs/pull/1321
            polars_bail!(SQLInterface: "RENAME and REPLACE wildcard options cannot (yet) be used simultaneously")
        }

        if let Some(items) = &options.opt_exclude {
            *excluded_cols = match items {
                ExcludeSelectItem::Single(ident) => vec![ident.value.clone()],
                ExcludeSelectItem::Multiple(idents) => {
                    idents.iter().map(|i| i.value.clone()).collect()
                },
            };
        }
        if let Some(items) = &options.opt_rename {
            match items {
                RenameSelectItem::Single(rename) => {
                    rename_cols.0.push(rename.ident.value.clone());
                    rename_cols.1.push(rename.alias.value.clone());
                },
                RenameSelectItem::Multiple(renames) => {
                    for rn in renames {
                        rename_cols.0.push(rn.ident.value.clone());
                        rename_cols.1.push(rn.alias.value.clone());
                    }
                },
            }
        }
        if let Some(replacements) = &options.opt_replace {
            for rp in &replacements.items {
                let replacement_expr = parse_sql_expr(&rp.expr, self, schema);
                replace_exprs.push(replacement_expr?.alias(rp.column_name.value.as_str()));
            }
        }
        Ok(exprs)
    }

    fn rename_columns_from_table_alias(
        &mut self,
        mut frame: LazyFrame,
        alias: &TableAlias,
    ) -> PolarsResult<LazyFrame> {
        if alias.columns.is_empty() {
            Ok(frame)
        } else {
            let schema = frame.schema_with_arenas(&mut self.lp_arena, &mut self.expr_arena)?;
            if alias.columns.len() != schema.len() {
                polars_bail!(
                    SQLSyntax: "number of columns ({}) in alias '{}' does not match the number of columns in the table/query ({})",
                    alias.columns.len(), alias.name.value, schema.len()
                )
            } else {
                let existing_columns: Vec<_> = schema.iter_names().collect();
                let new_columns: Vec<_> = alias.columns.iter().map(|c| c.value.clone()).collect();
                Ok(frame.rename(existing_columns, new_columns))
            }
        }
    }
}

impl SQLContext {
    /// Get internal table map. For internal use only.
    pub fn get_table_map(&self) -> PlHashMap<String, LazyFrame> {
        self.table_map.clone()
    }

    /// Create a new SQLContext from a table map. For internal use only
    pub fn new_from_table_map(table_map: PlHashMap<String, LazyFrame>) -> Self {
        Self {
            table_map,
            ..Default::default()
        }
    }
}
