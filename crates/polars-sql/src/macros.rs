//! SQL macros: `CREATE MACRO f(a, b := 5) AS a + b` and `CREATE MACRO t(n) AS TABLE SELECT ...`.
//!
//! A macro call is replaced by the macro body, with its parameters replaced by the arguments,
//! before the statement is planned.
use std::cell::Cell;
use std::ops::ControlFlow;
use std::sync::{Arc, RwLock};

use polars_core::prelude::{PolarsError, PolarsResult, polars_bail, polars_ensure, polars_err};
use polars_utils::aliases::{PlHashMap, PlHashSet};
use sqlparser::ast::{
    AccessExpr, CastKind, Expr as SQLExpr, Function as SQLFunction, FunctionArg, FunctionArgExpr,
    FunctionArgumentList, FunctionArguments, Ident, ObjectName, OrderByExpr, Query, Select,
    SelectItem, SetExpr, Statement, TableAlias, TableFactor, TableWithJoins, Value as SQLValue,
    Visit, VisitMut, Visitor, VisitorMut,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::keywords::Keyword;
use sqlparser::parser::{Parser, ParserError};
use sqlparser::tokenizer::{Token, TokenWithSpan};

use crate::context::{RelationName, match_qualifier, relation_name};
use crate::dyn_visit;
use crate::lambda::{lambda_arg, shift_lambda_elements};
use crate::params::first_param;
use crate::sql_expr::{parser_of, to_sql_interface_err};
use crate::sql_visitors::expr_references_any_column;
use crate::subquery::strip_nested;

#[derive(Clone, Debug)]
pub(crate) enum MacroBody {
    Expr(Box<SQLExpr>),
    Table(Box<Query>),
}

#[derive(Clone, Debug)]
pub(crate) struct MacroParam {
    pub(crate) name: Ident,
    pub(crate) default: Option<SQLExpr>,
}

/// One definition of a macro, for a number of arguments.
#[derive(Clone, Debug)]
pub(crate) struct MacroOverload {
    pub(crate) params: Vec<MacroParam>,
    pub(crate) body: MacroBody,
}

#[derive(Clone, Debug)]
pub(crate) enum MacroStatement {
    Create {
        or_replace: bool,
        name: Ident,
        table: bool,
        overloads: Vec<MacroOverload>,
    },
    Drop {
        table: bool,
        if_exists: bool,
        name: Ident,
    },
}

/// Visits the bodies and parameter defaults of a created macro.
impl VisitMut for MacroStatement {
    fn visit<V: VisitorMut>(&mut self, visitor: &mut V) -> ControlFlow<V::Break> {
        let MacroStatement::Create { overloads, .. } = self else {
            return ControlFlow::Continue(());
        };
        for overload in overloads {
            for default in overload
                .params
                .iter_mut()
                .filter_map(|p| p.default.as_mut())
            {
                default.visit(visitor)?;
            }
            match &mut overload.body {
                MacroBody::Expr(expr) => expr.visit(visitor)?,
                MacroBody::Table(query) => query.visit(visitor)?,
            }
        }
        ControlFlow::Continue(())
    }
}

/// The macros of a context, by lowercase name.
#[derive(Clone, Default)]
pub(crate) struct Macros {
    scalar: Arc<RwLock<PlHashMap<String, Arc<[MacroOverload]>>>>,
    table: Arc<RwLock<PlHashMap<String, Arc<[MacroOverload]>>>>,
}

impl Macros {
    fn map(&self, table: bool) -> &RwLock<PlHashMap<String, Arc<[MacroOverload]>>> {
        if table { &self.table } else { &self.scalar }
    }

    pub(crate) fn create(
        &self,
        name: &Ident,
        table: bool,
        or_replace: bool,
        overloads: Vec<MacroOverload>,
    ) -> PolarsResult<()> {
        let mut map = self.map(table).write().unwrap();
        let key = name.value.to_lowercase();
        polars_ensure!(
            or_replace || !map.contains_key(&key),
            SQLInterface: "macro '{}' already exists", name.value
        );
        map.insert(key, overloads.into());
        Ok(())
    }

    pub(crate) fn drop(&self, name: &Ident, table: bool, if_exists: bool) -> PolarsResult<()> {
        let removed = self
            .map(table)
            .write()
            .unwrap()
            .remove(&name.value.to_lowercase());
        polars_ensure!(
            removed.is_some() || if_exists,
            SQLInterface: "{}macro '{}' does not exist", if table { "table " } else { "" }, name.value
        );
        Ok(())
    }

    fn get(&self, name: &ObjectName, table: bool) -> Option<(String, Arc<[MacroOverload]>)> {
        let [part] = name.0.as_slice() else {
            return None;
        };
        let name = &part.as_ident()?.value;
        let map = self.map(table).read().unwrap();
        if map.is_empty() {
            return None;
        }
        let overloads = map.get(&name.to_lowercase())?.clone();
        Some((name.clone(), overloads))
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.scalar.read().unwrap().is_empty() && self.table.read().unwrap().is_empty()
    }

    /// Replace the macro calls in `node` by the macro bodies. Returns whether there were any.
    pub(crate) fn expand<T: Expandable>(&self, node: &mut T) -> PolarsResult<bool> {
        if self.is_empty() {
            return Ok(false);
        }
        let mut expander = MacroExpander {
            macros: self,
            expanding: vec![],
            expanded: false,
            size: ExpandedSize::default(),
            lambdas: LambdaScopes::default(),
        };
        if let ControlFlow::Break(err) = dyn_visit::visit_mut(node, &mut expander) {
            return Err(err);
        }
        if expander.expanded {
            // The expanded SQL may not be nested deeper than the parser accepts.
            let sql = node.to_string();
            let reparsed = Parser::new(&GenericDialect)
                .try_with_sql(&sql)
                .and_then(|mut parser| T::reparse(&mut parser));
            polars_ensure!(
                !matches!(reparsed, Err(ParserError::RecursionLimitExceeded)),
                SQLInterface: "the query is nested too deeply after expanding its macros"
            );
        }
        Ok(expander.expanded)
    }

    /// `node` with its macro calls replaced by the macro bodies, or `None` if it has none.
    pub(crate) fn expanded<T: Expandable + Clone>(&self, node: &T) -> PolarsResult<Option<T>> {
        if self.is_empty() {
            return Ok(None);
        }
        let mut node = node.clone();
        Ok(self.expand(&mut node)?.then_some(node))
    }
}

/// A statement or query whose macros can be expanded.
pub(crate) trait Expandable: Visit + VisitMut + std::fmt::Display {
    fn reparse(parser: &mut Parser) -> Result<(), ParserError>;
}

impl Expandable for Statement {
    fn reparse(parser: &mut Parser) -> Result<(), ParserError> {
        parser.parse_statement().map(|_| ())
    }
}

impl Expandable for Query {
    fn reparse(parser: &mut Parser) -> Result<(), ParserError> {
        parser.parse_query().map(|_| ())
    }
}

impl Expandable for SQLExpr {
    fn reparse(parser: &mut Parser) -> Result<(), ParserError> {
        parser.parse_expr().map(|_| ())
    }
}

impl Expandable for OrderByExpr {
    fn reparse(parser: &mut Parser) -> Result<(), ParserError> {
        parser.parse_order_by_expr().map(|_| ())
    }
}

/// The macro statement at the start of `parser`; if there is none, `parser` stays at the start.
pub(crate) fn parse_macro_statement(parser: &mut Parser) -> PolarsResult<Option<MacroStatement>> {
    let head = parser
        .maybe_parse(|p| {
            while p.consume_token(&Token::SemiColon) {}
            match p.parse_one_of_keywords(&[Keyword::CREATE, Keyword::DROP]) {
                Some(Keyword::CREATE) => {
                    let or_replace = p.parse_keywords(&[Keyword::OR, Keyword::REPLACE]);
                    let _ = p.parse_one_of_keywords(&[Keyword::TEMP, Keyword::TEMPORARY]);
                    p.expect_one_of_keywords(&[Keyword::MACRO, Keyword::FUNCTION])?;
                    Ok((true, or_replace, false))
                },
                Some(_) => {
                    let table =
                        match p.expect_one_of_keywords(&[Keyword::MACRO, Keyword::FUNCTION])? {
                            Keyword::MACRO => p.parse_keyword(Keyword::TABLE),
                            _ => false,
                        };
                    Ok((false, false, table))
                },
                None => Err(ParserError::ParserError(String::new())),
            }
        })
        .map_err(to_sql_interface_err)?;
    let Some((create, or_replace, table)) = head else {
        return Ok(None);
    };
    let stmt = if create {
        parse_create_macro(parser, or_replace)
    } else {
        parse_drop_macro(parser, table)
    }
    .map_err(to_sql_interface_err)?;
    while parser.consume_token(&Token::SemiColon) {}
    if parser.peek_token().token != Token::EOF {
        polars_bail!(SQLInterface: "one (and only one) statement can be parsed at a time");
    }
    Ok(Some(stmt))
}

fn parse_create_macro(
    parser: &mut Parser,
    or_replace: bool,
) -> Result<MacroStatement, ParserError> {
    let name = single_name(parser.parse_object_name(false)?)?;
    let mut overloads = vec![];
    let mut table = false;
    loop {
        parser.expect_token(&Token::LParen)?;
        let mut params = vec![];
        if !parser.consume_token(&Token::RParen) {
            loop {
                let name = parser.parse_identifier()?;
                let default = if parser.consume_token(&Token::Assignment)
                    || parser.consume_token(&Token::RArrow)
                {
                    Some(parser.parse_expr()?)
                } else {
                    None
                };
                params.push(MacroParam { name, default });
                if !parser.consume_token(&Token::Comma) {
                    break;
                }
            }
            parser.expect_token(&Token::RParen)?;
        }
        parser.expect_keyword_is(Keyword::AS)?;
        let body = if parser.parse_keyword(Keyword::TABLE) {
            MacroBody::Table(parse_table_body(parser)?)
        } else {
            MacroBody::Expr(Box::new(parser.parse_expr()?))
        };
        let is_table = matches!(body, MacroBody::Table(_));
        if !overloads.is_empty() && is_table != table {
            return Err(ParserError::ParserError(
                "the definitions of a macro must all be scalar or all be TABLE".into(),
            ));
        }
        table = is_table;
        overloads.push(MacroOverload { params, body });
        if !parser.consume_token(&Token::Comma) {
            break;
        }
    }
    Ok(MacroStatement::Create {
        or_replace,
        name,
        table,
        overloads,
    })
}

/// The query of a TABLE definition, which ends before the `, (...) AS TABLE` of the next
/// definition.
fn parse_table_body(parser: &mut Parser) -> Result<Box<Query>, ParserError> {
    // The parser is past `AS TABLE`, so the next token is at the current index + 1.
    let tokens: Vec<&TokenWithSpan> = (parser.get_current_index() + 1..)
        .map(|i| parser.token_at(i))
        .take_while(|t| !matches!(t.token, Token::EOF | Token::SemiColon))
        .filter(|t| !matches!(t.token, Token::Whitespace(_)))
        .collect();
    let len = next_definition(&tokens).unwrap_or(tokens.len());
    let mut body_parser = parser_of(tokens[..len].iter().map(|&t| t.clone()).collect());
    let query = body_parser.parse_query()?;
    let rest = body_parser.peek_token();
    if rest.token != Token::EOF {
        return body_parser.expected("the end of the macro body", rest);
    }
    for _ in 0..len {
        parser.next_token();
    }
    Ok(query)
}

/// The position of the comma that starts the next TABLE definition in `tokens`.
fn next_definition(tokens: &[&TokenWithSpan]) -> Option<usize> {
    let is_keyword = |i: usize, keywords: &[Keyword]| matches!(tokens.get(i).map(|t| &t.token), Some(Token::Word(w)) if keywords.contains(&w.keyword));
    let mut depth = 0usize;
    for (i, token) in tokens.iter().enumerate() {
        match token.token {
            Token::LParen | Token::LBracket | Token::LBrace => depth += 1,
            Token::RParen | Token::RBracket | Token::RBrace => depth = depth.saturating_sub(1),
            Token::Comma if depth == 0 && tokens.get(i + 1)?.token == Token::LParen => {
                // `, (params) AS TABLE` and a query; `(a) AS table` can also be a SELECT item.
                let mut params_depth = 0usize;
                let close = (i + 1..tokens.len()).find(|&j| {
                    match tokens[j].token {
                        Token::LParen => params_depth += 1,
                        Token::RParen => params_depth -= 1,
                        _ => {},
                    }
                    params_depth == 0
                })?;
                if is_keyword(close + 1, &[Keyword::AS])
                    && is_keyword(close + 2, &[Keyword::TABLE])
                    && (is_keyword(
                        close + 3,
                        &[Keyword::SELECT, Keyword::WITH, Keyword::VALUES],
                    ) || tokens.get(close + 3).map(|t| &t.token) == Some(&Token::LParen))
                {
                    return Some(i);
                }
            },
            _ => {},
        }
    }
    None
}

fn parse_drop_macro(parser: &mut Parser, table: bool) -> Result<MacroStatement, ParserError> {
    let if_exists = parser.parse_keywords(&[Keyword::IF, Keyword::EXISTS]);
    let name = single_name(parser.parse_object_name(false)?)?;
    Ok(MacroStatement::Drop {
        table,
        if_exists,
        name,
    })
}

fn single_name(name: ObjectName) -> Result<Ident, ParserError> {
    match name.0.as_slice() {
        [part] => part
            .as_ident()
            .cloned()
            .ok_or_else(|| ParserError::ParserError(format!("invalid macro name {name}"))),
        _ => Err(ParserError::ParserError(format!(
            "a macro name cannot be qualified; found {name}"
        ))),
    }
}

/// Check the definitions of macro `name`.
pub(crate) fn check_macro(name: &Ident, overloads: &[MacroOverload]) -> PolarsResult<()> {
    for overload in overloads {
        let params = &overload.params;
        let param = match &overload.body {
            MacroBody::Expr(body) => first_param(body.as_ref()),
            MacroBody::Table(query) => first_param(query.as_ref()),
        };
        if let Some(p) = param.or_else(|| params.iter().find_map(|p| first_param(&p.default))) {
            polars_bail!(
                SQLInterface: "query parameters are not supported in CREATE MACRO; found {}", p
            );
        }
        for (i, param) in params.iter().enumerate() {
            polars_ensure!(
                !params[..i].iter().any(|p| same_name(&p.name, &param.name.value)),
                SQLInterface: "macro '{}' has a duplicate parameter '{}'", name.value, param.name.value
            );
            polars_ensure!(
                param.default.as_ref().is_none_or(|d| !expr_references_any_column(d)),
                SQLInterface: "the default of parameter '{}' of macro '{}' cannot read columns",
                param.name.value, name.value
            );
            polars_ensure!(
                param.default.is_some() || params[..i].iter().all(|p| p.default.is_none()),
                SQLInterface: "in macro '{}', parameter '{}' without a default follows a parameter with a default",
                name.value, param.name.value
            );
        }
        if let MacroBody::Expr(body) = &overload.body {
            let mut check = FreeIdentifiers {
                params,
                query_depth: 0,
            };
            if let ControlFlow::Break(ident) = dyn_visit::visit(body, &mut check) {
                polars_bail!(
                    SQLInterface: "the body of macro '{}' can only use its parameters; found '{}'",
                    name.value, ident
                );
            }
        }
    }
    Ok(())
}

/// Whether a call of `name` parses as a call of a function (or table function), rather than
/// as special syntax like `TRIM(...)` or `UNNEST(...)`, which a macro cannot replace.
pub(crate) fn is_callable_name(name: &Ident, table: bool) -> bool {
    let sql = if table {
        format!("SELECT * FROM {name}(1)")
    } else {
        format!("SELECT {name}(1)")
    };
    let Ok(statements) = Parser::parse_sql(&GenericDialect, &sql) else {
        return false;
    };
    let [Statement::Query(query)] = statements.as_slice() else {
        return false;
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return false;
    };
    if table {
        matches!(
            select.from.as_slice(),
            [TableWithJoins {
                relation: TableFactor::Table { args: Some(_), .. },
                ..
            }]
        )
    } else {
        matches!(
            select.projection.as_slice(),
            [SelectItem::UnnamedExpr(SQLExpr::Function(_))]
        )
    }
}

fn same_name(param: &Ident, name: &str) -> bool {
    param.value.eq_ignore_ascii_case(name)
}

/// Finds an identifier outside subqueries that is not a parameter.
struct FreeIdentifiers<'a> {
    params: &'a [MacroParam],
    query_depth: usize,
}

impl Visitor for FreeIdentifiers<'_> {
    type Break = String;

    fn pre_visit_query(&mut self, _query: &Query) -> ControlFlow<String> {
        self.query_depth += 1;
        ControlFlow::Continue(())
    }

    fn post_visit_query(&mut self, _query: &Query) -> ControlFlow<String> {
        self.query_depth -= 1;
        ControlFlow::Continue(())
    }

    fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<String> {
        if self.query_depth > 0 {
            return ControlFlow::Continue(());
        }
        let first = match expr {
            SQLExpr::Identifier(ident) => ident,
            SQLExpr::CompoundIdentifier(idents) => &idents[0],
            _ => return ControlFlow::Continue(()),
        };
        if self.params.iter().any(|p| same_name(&p.name, &first.value)) {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(expr.to_string())
        }
    }
}

/// The deepest nesting of macro calls in macro bodies.
const MAX_MACRO_DEPTH: usize = 64;

/// The most nodes that expanding the macros of a statement may copy.
const MAX_EXPANDED_EXPRS: usize = 100_000;

/// The number of nodes copied by an expansion.
#[derive(Default)]
struct ExpandedSize(Cell<usize>);

impl ExpandedSize {
    /// Counts the expressions, queries and relations of `node`, which is about to be copied.
    fn add<T: Visit>(&self, node: &T) -> ControlFlow<PolarsError> {
        struct Count(usize);
        impl Visitor for Count {
            type Break = ();
            fn pre_visit_expr(&mut self, _expr: &SQLExpr) -> ControlFlow<()> {
                self.0 += 1;
                ControlFlow::Continue(())
            }

            fn pre_visit_query(&mut self, _query: &Query) -> ControlFlow<()> {
                self.0 += 1;
                ControlFlow::Continue(())
            }

            fn pre_visit_select(&mut self, _select: &Select) -> ControlFlow<()> {
                self.0 += 1;
                ControlFlow::Continue(())
            }

            fn pre_visit_table_factor(&mut self, _factor: &TableFactor) -> ControlFlow<()> {
                self.0 += 1;
                ControlFlow::Continue(())
            }
        }
        let mut count = Count(0);
        let _ = dyn_visit::visit(node, &mut count);
        let size = self.0.get() + count.0;
        if size > MAX_EXPANDED_EXPRS {
            return ControlFlow::Break(polars_err!(
                SQLInterface: "the query is too large after expanding its macros"
            ));
        }
        self.0.set(size);
        ControlFlow::Continue(())
    }
}

struct MacroExpander<'a> {
    macros: &'a Macros,
    /// The macros whose bodies are being expanded, as (lowercase name, whether a table macro).
    expanding: Vec<(String, bool)>,
    expanded: bool,
    size: ExpandedSize,
    lambdas: LambdaScopes,
}

/// The lambdas around the visited expression.
#[derive(Default)]
struct LambdaScopes {
    depth: usize,
    /// The bodies of the lambda arguments of the functions being visited.
    bodies: Vec<*const SQLExpr>,
}

impl LambdaScopes {
    /// Called before the children of `expr` are visited.
    fn enter(&mut self, expr: &SQLExpr) {
        if self.bodies.contains(&(expr as *const SQLExpr)) {
            self.depth += 1;
        }
        if let SQLExpr::Function(func) = expr {
            self.bodies
                .extend(lambda_args(func).map(|body| body as *const SQLExpr));
        }
    }

    /// Called after the children of `expr` are visited. Returns whether `expr` is a lambda
    /// body, which `leave_body` leaves once `expr` is visited.
    fn leave_children(&mut self, expr: &SQLExpr) -> bool {
        if let SQLExpr::Function(func) = expr {
            let n_lambdas = lambda_args(func).count();
            self.bodies.truncate(self.bodies.len() - n_lambdas);
        }
        self.bodies.contains(&(expr as *const SQLExpr))
    }

    fn leave_body(&mut self, is_body: bool) {
        if is_body {
            self.depth -= 1;
        }
    }
}

impl MacroExpander<'_> {
    /// An error if macro `name` is being expanded: a macro that calls itself never ends.
    fn check_not_expanding(&self, name: &str, table: bool) -> ControlFlow<PolarsError> {
        let key = (name.to_lowercase(), table);
        if self.expanding.contains(&key) {
            return ControlFlow::Break(polars_err!(
                SQLInterface: "macro '{}' calls itself", name
            ));
        }
        ControlFlow::Continue(())
    }

    /// Expand the macro calls in the arguments of a call of macro `name`.
    fn expand_args(
        &mut self,
        bound: Vec<BoundArg>,
        name: &str,
        table: bool,
    ) -> ControlFlow<PolarsError, Vec<(Ident, SQLExpr)>> {
        let mut params = Vec::with_capacity(bound.len());
        for mut arg in bound {
            if !arg.is_default {
                // An argument of the call is expanded where the call is.
                dyn_visit::visit_mut(&mut arg.value, self)?;
            } else {
                // The lambdas of a default count from the call.
                shift_lambda_elements(&mut arg.value, 0, self.lambdas.depth);
                self.expand_nested(&mut arg.value, name, table)?;
                // As written in the definition, a default cannot read columns.
                if expr_references_any_column(&arg.value) {
                    return ControlFlow::Break(polars_err!(
                        SQLInterface: "the default of parameter '{}' of macro '{}' cannot read columns",
                        arg.name.value, name
                    ));
                }
            }
            params.push((arg.name, arg.value));
        }
        ControlFlow::Continue(params)
    }

    /// Expand the macro calls in the body of macro `name`, which replaced a call.
    fn expand_nested<T: VisitMut>(
        &mut self,
        node: &mut T,
        name: &str,
        table: bool,
    ) -> ControlFlow<PolarsError> {
        if self.expanding.len() >= MAX_MACRO_DEPTH {
            return ControlFlow::Break(polars_err!(
                SQLInterface: "macro '{}' is nested more than {} macros deep", name, MAX_MACRO_DEPTH
            ));
        }
        self.expanding.push((name.to_lowercase(), table));
        let res = dyn_visit::visit_mut(node, self);
        self.expanding.pop();
        res
    }
}

impl VisitorMut for MacroExpander<'_> {
    type Break = PolarsError;

    fn pre_visit_table_factor(&mut self, factor: &mut TableFactor) -> ControlFlow<PolarsError> {
        let (macro_name, lateral) = match &*factor {
            TableFactor::Table {
                name,
                args: Some(_),
                ..
            } => (name, false),
            TableFactor::Function { name, lateral, .. } => (name, *lateral),
            _ => return ControlFlow::Continue(()),
        };
        let Some((name, overloads)) = self.macros.get(macro_name, true) else {
            return ControlFlow::Continue(());
        };
        let default_alias = macro_name
            .0
            .last()
            .and_then(|part| part.as_ident())
            .cloned();
        self.check_not_expanding(&name, true)?;
        let (args, alias, sample) = match factor {
            TableFactor::Table {
                alias,
                args: Some(args),
                sample,
                version,
                with_hints,
                partitions,
                json_path,
                index_hints,
                with_ordinality,
                ..
            } if version.is_none()
                && with_hints.is_empty()
                && partitions.is_empty()
                && json_path.is_none()
                && index_hints.is_empty()
                && !*with_ordinality =>
            {
                (&args.args, alias, sample.take())
            },
            TableFactor::Function {
                args,
                alias,
                with_ordinality: false,
                ..
            } => (&*args, alias, None),
            _ => {
                return ControlFlow::Break(polars_err!(
                    SQLInterface: "table macro '{}' takes no table modifiers", name
                ));
            },
        };
        let (overload, bound) = flow(bind_args(&name, &overloads, args))?;
        let alias = alias.take();
        let bound = self.expand_args(bound, &name, true)?;
        let MacroBody::Table(query) = &overload.body else {
            unreachable!()
        };
        self.size.add(&**query)?;
        let mut subquery = query.clone();
        dyn_visit::visit_mut(
            &mut *subquery,
            &mut ParamReplacer::new(self.macros, &name, &bound, &self.size, self.lambdas.depth),
        )?;
        self.expand_nested(&mut subquery, &name, true)?;
        let alias = alias.unwrap_or_else(|| TableAlias {
            explicit: false,
            name: default_alias.unwrap_or_else(|| Ident::new(name)),
            columns: vec![],
            at: None,
        });
        *factor = TableFactor::Derived {
            lateral,
            subquery,
            alias: Some(alias),
            sample,
        };
        self.expanded = true;
        ControlFlow::Continue(())
    }

    fn post_visit_table_factor(&mut self, factor: &mut TableFactor) -> ControlFlow<PolarsError> {
        unparenthesize_table_args(factor, self.macros);
        ControlFlow::Continue(())
    }

    fn pre_visit_expr(&mut self, expr: &mut SQLExpr) -> ControlFlow<PolarsError> {
        self.lambdas.enter(expr);
        ControlFlow::Continue(())
    }

    fn post_visit_expr(&mut self, expr: &mut SQLExpr) -> ControlFlow<PolarsError> {
        let is_body = self.lambdas.leave_children(expr);
        self.expand_call(expr)?;
        self.lambdas.leave_body(is_body);
        ControlFlow::Continue(())
    }
}

impl MacroExpander<'_> {
    /// Replaces `expr` by the body of the macro it calls, if it calls one.
    fn expand_call(&mut self, expr: &mut SQLExpr) -> ControlFlow<PolarsError> {
        unparenthesize_operands(expr, self.macros);
        let SQLExpr::Function(func) = expr else {
            return ControlFlow::Continue(());
        };
        let Some((name, overloads)) = self.macros.get(&func.name, false) else {
            return ControlFlow::Continue(());
        };
        self.check_not_expanding(&name, false)?;
        let args = flow(macro_call_args(&name, func))?;
        let (overload, bound) = flow(bind_args(&name, &overloads, args))?;
        let bound = self.expand_args(bound, &name, false)?;
        let MacroBody::Expr(body) = &overload.body else {
            unreachable!()
        };
        self.size.add(body.as_ref())?;
        let mut body = body.as_ref().clone();
        shift_lambda_elements(&mut body, 0, self.lambdas.depth);
        let mut replacer =
            ParamReplacer::new(self.macros, &name, &bound, &self.size, self.lambdas.depth);
        dyn_visit::visit_mut(&mut body, &mut replacer)?;
        self.expand_nested(&mut body, &name, false)?;
        // The value of a call is an expression, not a position or keyword in ORDER BY or
        // GROUP BY.
        *expr = match body {
            SQLExpr::Value(_) => SQLExpr::Nested(Box::new(body)),
            body => parenthesize(body),
        };
        self.expanded = true;
        ControlFlow::Continue(())
    }
}

/// The bodies of the lambda arguments of `func`.
fn lambda_args(func: &SQLFunction) -> impl Iterator<Item = &SQLExpr> {
    let args = match &func.args {
        FunctionArguments::List(list) => list.args.as_slice(),
        _ => &[],
    };
    args.iter().filter_map(|arg| Some(lambda_arg(arg)?.1))
}

/// `res` as the result of a visit.
fn flow<T>(res: PolarsResult<T>) -> ControlFlow<PolarsError, T> {
    match res {
        Ok(value) => ControlFlow::Continue(value),
        Err(err) => ControlFlow::Break(err),
    }
}

/// The arguments of the scalar macro call `func`.
fn macro_call_args<'a>(name: &str, func: &'a SQLFunction) -> PolarsResult<&'a [FunctionArg]> {
    polars_ensure!(
        func.over.is_none()
            && func.filter.is_none()
            && func.within_group.is_empty()
            && func.null_treatment.is_none()
            && matches!(func.parameters, FunctionArguments::None),
        SQLInterface: "macro '{}' does not support OVER, FILTER, WITHIN GROUP or IGNORE/RESPECT NULLS", name
    );
    match &func.args {
        FunctionArguments::None => Ok(&[]),
        FunctionArguments::List(FunctionArgumentList {
            args,
            duplicate_treatment: None,
            clauses,
        }) if clauses.is_empty() => Ok(args),
        _ => polars_bail!(
            SQLInterface: "macro '{}' takes only arguments, without DISTINCT, ORDER BY or other clauses", name
        ),
    }
}

/// The definition of macro `name` for `args`, with each parameter and its value.
fn bind_args<'a>(
    name: &str,
    overloads: &'a [MacroOverload],
    args: &[FunctionArg],
) -> PolarsResult<(&'a MacroOverload, Vec<BoundArg>)> {
    let mut positional = vec![];
    let mut named: Vec<(&Ident, &SQLExpr)> = vec![];
    for arg in args {
        match arg {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => {
                polars_ensure!(
                    named.is_empty(),
                    SQLInterface: "in the call of macro '{}', a positional argument follows a named argument", name
                );
                positional.push(e);
            },
            FunctionArg::Named {
                name: arg_name,
                arg: FunctionArgExpr::Expr(e),
                ..
            } => named.push((arg_name, e)),
            _ => polars_bail!(SQLInterface: "invalid argument of macro '{}': {}", name, arg),
        }
    }
    let mut given_twice = None;
    let mut bind = |params: &[MacroParam]| -> Option<Vec<BoundArg>> {
        if positional.len() > params.len() {
            return None;
        }
        let mut values: Vec<Option<SQLExpr>> = positional
            .iter()
            .map(|e| Some((*e).clone()))
            .chain(std::iter::repeat_n(None, params.len() - positional.len()))
            .collect();
        for (arg_name, e) in &named {
            let i = params
                .iter()
                .position(|p| same_name(&p.name, &arg_name.value))?;
            if values[i].is_some() {
                given_twice.get_or_insert(arg_name.value.clone());
                return None;
            }
            values[i] = Some((*e).clone());
        }
        params
            .iter()
            .zip(values)
            .map(|(param, value)| {
                let (value, is_default) = match value {
                    Some(value) => (value, false),
                    None => (param.default.clone()?, true),
                };
                Some(BoundArg {
                    name: param.name.clone(),
                    value,
                    is_default,
                })
            })
            .collect()
    };
    let mut matches = overloads
        .iter()
        .filter_map(|overload| Some((overload, bind(&overload.params)?)));
    let Some(binding) = matches.next() else {
        if let Some(param) = given_twice {
            polars_bail!(
                SQLInterface: "in the call of macro '{}', parameter '{}' is given twice", name, param
            );
        }
        polars_bail!(
            SQLInterface: "macro '{}' has no definition for the given arguments", name
        )
    };
    polars_ensure!(
        matches.next().is_none(),
        SQLInterface: "the call of macro '{}' matches more than one of its definitions", name
    );
    Ok(binding)
}

/// A parameter of a macro call and its value.
struct BoundArg {
    name: Ident,
    value: SQLExpr,
    /// Whether the value is the parameter's default.
    is_default: bool,
}

/// The names of the relations of a query or SELECT, with each part of a qualified name.
struct Relations {
    names: Vec<RelationName>,
    /// A CTE or derived table, which does not see the relations of the query it is part of.
    starts_scope: bool,
    is_query: bool,
}

/// Replaces the parameters of a macro body by their values.
struct ParamReplacer<'a> {
    macros: &'a Macros,
    name: &'a str,
    params: &'a [(Ident, SQLExpr)],
    size: &'a ExpandedSize,
    /// The number of lambdas around the call, which the elements of the values count.
    call_depth: usize,
    /// The lambdas of the body around the visited expression.
    lambdas: LambdaScopes,
    /// The relations of each query and SELECT around the visited expression.
    relations: Vec<Relations>,
    /// The CTEs and derived tables of the visited queries.
    scoped_queries: PlHashSet<*const Query>,
}

impl<'a> ParamReplacer<'a> {
    fn new(
        macros: &'a Macros,
        name: &'a str,
        params: &'a [(Ident, SQLExpr)],
        size: &'a ExpandedSize,
        call_depth: usize,
    ) -> Self {
        Self {
            macros,
            name,
            params,
            size,
            call_depth,
            lambdas: LambdaScopes::default(),
            relations: Vec::new(),
            scoped_queries: PlHashSet::default(),
        }
    }

    /// Whether a leading run of `idents` names a relation that the visited expression sees.
    fn names_relation(&self, idents: &[Ident]) -> bool {
        let mut hidden = false;
        for relations in self.relations.iter().rev() {
            if hidden {
                // The relations of an enclosing query can still be correlated, unless the
                // query is itself a CTE or derived table.
                if relations.is_query {
                    hidden = relations.starts_scope;
                }
                continue;
            }
            if match_qualifier(idents, relations.names.iter()).is_some() {
                return true;
            }
            hidden = relations.starts_scope;
        }
        false
    }

    fn value(&self, name: &str) -> ControlFlow<PolarsError, Option<SQLExpr>> {
        let Some((_, value)) = self.params.iter().find(|(p, _)| same_name(p, name)) else {
            return ControlFlow::Continue(None);
        };
        // A column of the caller would be read from the tables of the subquery.
        if !self.relations.is_empty() && expr_references_any_column(value) {
            return ControlFlow::Break(polars_err!(
                SQLInterface: "macro '{}' uses parameter '{}' in a query, where its value cannot read columns",
                self.name, name
            ));
        }
        self.size.add(value)?;
        let mut value = parenthesize(value.clone());
        // The lambdas of the value are placed in the lambdas of the body around it.
        shift_lambda_elements(&mut value, self.call_depth, self.lambdas.depth);
        ControlFlow::Continue(Some(value))
    }
}

/// A compound expression in parentheses, so that it keeps its meaning where it is placed.
fn parenthesize(expr: SQLExpr) -> SQLExpr {
    match expr {
        SQLExpr::Value(_)
        | SQLExpr::Interval(_)
        | SQLExpr::Identifier(_)
        | SQLExpr::CompoundIdentifier(_)
        | SQLExpr::Function(_)
        | SQLExpr::Nested(_)
        | SQLExpr::Subquery(_) => expr,
        _ => SQLExpr::Nested(Box::new(expr)),
    }
}

impl VisitorMut for ParamReplacer<'_> {
    type Break = PolarsError;

    fn pre_visit_select(&mut self, select: &mut Select) -> ControlFlow<PolarsError> {
        // A SELECT item that is just a parameter keeps the parameter name.
        for item in &mut select.projection {
            if let SelectItem::UnnamedExpr(SQLExpr::Identifier(ident)) = item
                && self.params.iter().any(|(p, _)| same_name(p, &ident.value))
            {
                let alias = ident.clone();
                *item = SelectItem::ExprWithAlias {
                    expr: SQLExpr::Identifier(ident.clone()),
                    alias,
                };
            }
        }
        let mut names = Vec::new();
        for table in &select.from {
            table_names(table, &mut names);
        }
        self.relations.push(Relations {
            names,
            starts_scope: false,
            is_query: false,
        });
        ControlFlow::Continue(())
    }

    fn post_visit_select(&mut self, _select: &mut Select) -> ControlFlow<PolarsError> {
        self.relations.pop();
        ControlFlow::Continue(())
    }

    fn pre_visit_query(&mut self, query: &mut Query) -> ControlFlow<PolarsError> {
        let starts_scope = self.scoped_queries.remove(&(&*query as *const Query));
        if let Some(with) = &query.with {
            self.scoped_queries.extend(
                with.cte_tables
                    .iter()
                    .map(|cte| &*cte.query as *const Query),
            );
        }
        let mut names = Vec::new();
        relation_names(&query.body, &mut names);
        self.relations.push(Relations {
            names,
            starts_scope,
            is_query: true,
        });
        ControlFlow::Continue(())
    }

    fn post_visit_query(&mut self, _query: &mut Query) -> ControlFlow<PolarsError> {
        self.relations.pop();
        ControlFlow::Continue(())
    }

    fn pre_visit_table_factor(&mut self, factor: &mut TableFactor) -> ControlFlow<PolarsError> {
        if let TableFactor::Derived {
            lateral: false,
            subquery,
            ..
        } = factor
        {
            self.scoped_queries.insert(&**subquery as *const Query);
        }
        // The conditions inside an aliased join see the relations that the alias hides, but
        // not the alias or the other relations of the FROM.
        if let Some(names) = hidden_by_alias(factor) {
            self.relations.push(Relations {
                names,
                starts_scope: true,
                is_query: false,
            });
        }
        ControlFlow::Continue(())
    }

    fn pre_visit_expr(&mut self, expr: &mut SQLExpr) -> ControlFlow<PolarsError> {
        self.lambdas.enter(expr);
        ControlFlow::Continue(())
    }

    fn post_visit_expr(&mut self, expr: &mut SQLExpr) -> ControlFlow<PolarsError> {
        let is_body = self.lambdas.leave_children(expr);
        self.replace_param(expr)?;
        self.lambdas.leave_body(is_body);
        ControlFlow::Continue(())
    }

    fn post_visit_table_factor(&mut self, factor: &mut TableFactor) -> ControlFlow<PolarsError> {
        if hidden_by_alias(factor).is_some() {
            self.relations.pop();
        }
        unparenthesize_table_args(factor, self.macros);
        ControlFlow::Continue(())
    }
}

impl ParamReplacer<'_> {
    fn replace_param(&mut self, expr: &mut SQLExpr) -> ControlFlow<PolarsError> {
        match expr {
            SQLExpr::Identifier(ident) => {
                if let Some(value) = self.value(&ident.value)? {
                    *expr = value;
                }
            },
            // A qualifier naming a relation of the body reads that relation.
            SQLExpr::CompoundIdentifier(idents) if self.names_relation(idents) => {},
            SQLExpr::CompoundIdentifier(idents) => {
                if let Some(value) = self.value(&idents[0].value)? {
                    // The fields of the value, also when it names a column like a table.
                    *expr = SQLExpr::CompoundFieldAccess {
                        root: Box::new(SQLExpr::Nested(Box::new(value))),
                        access_chain: idents[1..]
                            .iter()
                            .map(|ident| AccessExpr::Dot(SQLExpr::Identifier(ident.clone())))
                            .collect(),
                    };
                }
            },
            _ => unparenthesize_operands(expr, self.macros),
        }
        ControlFlow::Continue(())
    }
}

/// The names of the relations inside `factor` that its alias hides.
fn hidden_by_alias(factor: &TableFactor) -> Option<Vec<RelationName>> {
    let mut names = Vec::new();
    match factor {
        TableFactor::NestedJoin {
            table_with_joins,
            alias: Some(_),
        } => table_names(table_with_joins, &mut names),
        TableFactor::Pivot {
            table,
            alias: Some(_),
            ..
        }
        | TableFactor::Unpivot {
            table,
            alias: Some(_),
            ..
        } => factor_names(table, &mut names),
        _ => return None,
    }
    Some(names)
}

/// Adds the names of the relations that the ORDER BY of a query with this body can qualify
/// columns with to `names`. Each SELECT of a set
/// operation has its own relations.
fn relation_names(body: &SetExpr, names: &mut Vec<RelationName>) {
    match body {
        SetExpr::Select(select) => {
            for table in &select.from {
                table_names(table, names);
            }
        },
        SetExpr::Query(query) => relation_names(&query.body, names),
        _ => {},
    }
}

fn table_names(table: &TableWithJoins, names: &mut Vec<RelationName>) {
    for factor in std::iter::once(&table.relation).chain(table.joins.iter().map(|j| &j.relation)) {
        factor_names(factor, names);
    }
}

fn factor_names(factor: &TableFactor, names: &mut Vec<RelationName>) {
    match factor {
        // An alias hides the names of the relations inside.
        TableFactor::NestedJoin {
            table_with_joins,
            alias: None,
        } => table_names(table_with_joins, names),
        TableFactor::Pivot {
            table, alias: None, ..
        }
        | TableFactor::Unpivot {
            table, alias: None, ..
        } => factor_names(table, names),
        TableFactor::Function { name, alias, .. } => names.push(RelationName::new(match alias {
            Some(alias) => vec![alias.name.value.clone()],
            None => name
                .0
                .iter()
                .filter_map(|p| Some(p.as_ident()?.value.clone()))
                .collect(),
        })),
        _ => {},
    }
    names.extend(relation_name(factor));
}

/// Drops the parentheses around function arguments. A substituted value keeps its
/// parentheses only where it is an operand.
fn unparenthesize_operands(expr: &mut SQLExpr, macros: &Macros) {
    match expr {
        // The value of a macro argument keeps its meaning, as a number that is no position.
        SQLExpr::Function(function) if macros.get(&function.name, false).is_some() => {},
        SQLExpr::Cast { kind, expr, .. } if *kind != CastKind::DoubleColon => unparenthesize(expr),
        SQLExpr::Interval(interval)
            if matches!(strip_nested(&interval.value), SQLExpr::Value(_)) =>
        {
            unparenthesize(&mut interval.value)
        },
        SQLExpr::Function(function) => {
            if let FunctionArguments::List(list) = &mut function.args {
                unparenthesize_args(&mut list.args);
            }
        },
        _ => {},
    }
}

/// Drops the parentheses around the arguments of a table function or `UNNEST`.
fn unparenthesize_table_args(factor: &mut TableFactor, macros: &Macros) {
    match factor {
        TableFactor::Table { name, .. } if macros.get(name, true).is_some() => {},
        TableFactor::Table {
            args: Some(args), ..
        } => unparenthesize_args(&mut args.args),
        TableFactor::UNNEST { array_exprs, .. } => array_exprs.iter_mut().for_each(unparenthesize),
        _ => {},
    }
}

fn unparenthesize_args(args: &mut [FunctionArg]) {
    for arg in args {
        if let FunctionArg::Unnamed(FunctionArgExpr::Expr(expr))
        | FunctionArg::Named {
            arg: FunctionArgExpr::Expr(expr),
            ..
        }
        | FunctionArg::ExprNamed {
            arg: FunctionArgExpr::Expr(expr),
            ..
        } = arg
        {
            unparenthesize(expr);
        }
    }
}

fn unparenthesize(expr: &mut SQLExpr) {
    while let SQLExpr::Nested(inner) = expr {
        *expr = std::mem::replace(inner.as_mut(), SQLExpr::value(SQLValue::Null));
    }
}
