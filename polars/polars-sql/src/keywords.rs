//! Keywords that are supported by Polars SQL
//!
//! This is useful for syntax highlighting
//!
//! This module defines:
//! - all Polars SQL keywords [`all_keywords`]
//! - all of polars SQL functions [`all_functions`]
use crate::functions::PolarsSqlFunctions;
use crate::table_functions::PolarsTableFunctions;

/// Get all keywords that are supported by Polars SQL
pub fn all_keywords() -> Vec<&'static str> {
    let mut keywords = vec![];
    keywords.extend_from_slice(PolarsTableFunctions::keywords());
    keywords.extend_from_slice(PolarsSqlFunctions::keywords());
    use sqlparser::keywords;
    let sql_keywords = &[
        keywords::SELECT,
        keywords::FROM,
        keywords::WHERE,
        keywords::GROUP,
        keywords::BY,
        keywords::ORDER,
        keywords::LIMIT,
        keywords::OFFSET,
        keywords::AND,
        keywords::OR,
        keywords::AS,
        keywords::ON,
        keywords::INNER,
        keywords::LEFT,
        keywords::RIGHT,
        keywords::FULL,
        keywords::OUTER,
        keywords::JOIN,
        keywords::CREATE,
        keywords::TABLE,
        keywords::SHOW,
        keywords::TABLES,
        keywords::AS,
        keywords::VARCHAR,
        keywords::INT,
        keywords::FLOAT,
        keywords::DOUBLE,
        keywords::BOOLEAN,
        keywords::DATE,
        keywords::TIME,
        keywords::DATETIME,
        keywords::ARRAY,
        keywords::ASC,
        keywords::DESC,
        keywords::NULL,
        keywords::NOT,
        keywords::IN,
        keywords::WITH,
    ];
    keywords.extend_from_slice(sql_keywords);
    keywords
}

/// Get a list of all function names that are supported by Polars SQL
pub fn all_functions() -> Vec<&'static str> {
    let mut functions = vec![];
    functions.extend_from_slice(PolarsTableFunctions::keywords());
    functions.extend_from_slice(PolarsSqlFunctions::keywords());
    functions
}
