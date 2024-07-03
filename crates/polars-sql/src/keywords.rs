//! Keywords that are supported by Polars SQL
//!
//! This is useful for syntax highlighting
//!
//! This module defines:
//! - all Polars SQL keywords [`all_keywords`]
//! - all of polars SQL functions [`all_functions`]
use crate::functions::PolarsSQLFunctions;
use crate::table_functions::PolarsTableFunctions;

/// Get all keywords that are supported by Polars SQL
pub fn all_keywords() -> Vec<&'static str> {
    let mut keywords = vec![];
    keywords.extend_from_slice(PolarsTableFunctions::keywords());
    keywords.extend_from_slice(PolarsSQLFunctions::keywords());

    use sqlparser::keywords;
    let sql_keywords = &[
        keywords::AND,
        keywords::ANTI,
        keywords::ARRAY,
        keywords::AS,
        keywords::ASC,
        keywords::BOOLEAN,
        keywords::BY,
        keywords::CASE,
        keywords::CREATE,
        keywords::DATE,
        keywords::DATETIME,
        keywords::DESC,
        keywords::DISTINCT,
        keywords::DOUBLE,
        keywords::DROP,
        keywords::EXCEPT,
        keywords::EXCLUDE,
        keywords::FLOAT,
        keywords::FROM,
        keywords::FULL,
        keywords::GROUP,
        keywords::HAVING,
        keywords::IN,
        keywords::INNER,
        keywords::INT,
        keywords::INTERSECT,
        keywords::INTERVAL,
        keywords::JOIN,
        keywords::LEFT,
        keywords::LIMIT,
        keywords::NOT,
        keywords::NULL,
        keywords::OFFSET,
        keywords::ON,
        keywords::OR,
        keywords::ORDER,
        keywords::OUTER,
        keywords::REGEXP,
        keywords::RENAME,
        keywords::REPLACE,
        keywords::RIGHT,
        keywords::RLIKE,
        keywords::SELECT,
        keywords::SEMI,
        keywords::SHOW,
        keywords::TABLE,
        keywords::TABLES,
        keywords::THEN,
        keywords::TIME,
        keywords::TRUNCATE,
        keywords::UNION,
        keywords::USING,
        keywords::VARCHAR,
        keywords::WHEN,
        keywords::WHERE,
        keywords::WITH,
    ];
    keywords.extend_from_slice(sql_keywords);
    keywords
}

/// Get a list of all function names that are supported by Polars SQL
pub fn all_functions() -> Vec<&'static str> {
    let mut functions = vec![];
    functions.extend_from_slice(PolarsTableFunctions::keywords());
    functions.extend_from_slice(PolarsSQLFunctions::keywords());
    functions
}
