#[cfg(feature = "highlight")]
mod highlighter;
mod prompt;
use std::borrow::Cow;
use std::env;
use std::ffi::OsStr;
use std::io::{self, BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[cfg(feature = "highlight")]
use highlighter::SQLHighlighter;
use polars_core::prelude::*;
use polars_lazy::frame::LazyFrame;
use polars_sql::SQLContext;
use reedline::{
    DefaultPrompt, FileBackedHistory, Prompt, PromptHistorySearchStatus, Reedline, Signal,
};
use sqlparser::ast::{Select, SetExpr, Statement, TableFactor, TableWithJoins};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;
#[cfg(feature = "highlight")]
use syntect::easy::HighlightLines;
#[cfg(feature = "highlight")]
use syntect::highlighting::{Style, ThemeSet};
#[cfg(feature = "highlight")]
use syntect::parsing::SyntaxSet;
#[cfg(feature = "highlight")]
use syntect::util::{as_24_bit_terminal_escaped, LinesWithEndings};

use crate::cli::prompt::SQLPrompt;

// Command: /? | help
fn print_help() {
    println!("List of all client commands:");
    for (name, short, desc) in
        vec![(".help", "?", "Display this help."), (".exit", "q", "Exit")].iter()
    {
        println!("{:10}\\{:10} {}", name, short, desc);
    }
}

fn get_extension_from_filename(filename: &str) -> Option<&str> {
    Path::new(filename).extension().and_then(OsStr::to_str)
}

fn get_home_dir() -> PathBuf {
    match env::var("HOME") {
        Ok(path) => PathBuf::from(path),
        Err(_) => match env::var("USERPROFILE") {
            Ok(path) => PathBuf::from(path),
            Err(_) => panic!("Failed to get home directory"),
        },
    }
}

fn get_history_path() -> PathBuf {
    let mut home_dir = get_home_dir();
    home_dir.push(".polars");
    home_dir.push("history.txt");
    home_dir
}

fn execute_query(query: &str, ctx: &mut SQLContext) -> PolarsResult<DataFrame> {
    ctx.execute(query)?.collect()
}

fn execute_query_tty(query: &str, ctx: &mut SQLContext) -> String {
    match execute_query(query, ctx) {
        Ok(df) => format!("{df}"),
        Err(e) => format!("Error: {}", e),
    }
}

pub fn run_tty() -> std::io::Result<()> {
    // let ps = SyntaxSet::load_defaults_newlines();
    // let ts = ThemeSet::load_defaults();
    let mut context = SQLContext::new();
    let mut input: Vec<u8> = Vec::with_capacity(1024);
    let mut stdin = std::io::stdin();
    let history = Box::new(
        FileBackedHistory::with_file(20, get_history_path())
            .expect("Error configuring history with file"),
    );

    let mut line_editor = Reedline::create().with_history(history);

    #[cfg(feature = "highlight")]
    {
        let sql_highlighter = SQLHighlighter {};
        line_editor = line_editor.with_highlighter(Box::new(sql_highlighter));
    }

    let mut stdout = io::stdout();
    let mut context = SQLContext::new();
    let version = env!("CARGO_PKG_VERSION");

    println!("Polars CLI v{}", version);
    println!("Welcome to Polars CLI. Commands end with ; or \\n");
    println!("Type .help or \\? for help.");

    let prompt = SQLPrompt {};
    let mut is_exit_cmd = false;
    let mut scratch = String::with_capacity(1024);

    loop {
        let sig = line_editor.read_line(&prompt);
        match sig {
            Ok(Signal::Success(buffer)) => {
                is_exit_cmd = false;
                match buffer.as_ref() {
                    ".help" | "\\?" => print_help(),
                    ".exit" | "\\q" => {
                        break;
                    }
                    _ => {
                        let mut parts = buffer.splitn(2, ';');
                        let first = parts.next().unwrap();
                        scratch.push_str(first);

                        let second = parts.next();
                        if second.is_some() {
                            let res = execute_query_tty(&scratch, &mut context);
                            println!("{res}");
                            scratch.clear();
                        } else {
                            scratch.push(' ');
                        }
                    }
                }
            }
            Ok(Signal::CtrlD) | Ok(Signal::CtrlC) => {
                if is_exit_cmd {
                    break;
                } else {
                    is_exit_cmd = true;
                }
            }
            x => {
                is_exit_cmd = false;
                println!("Event: {:?}", x);
            }
        }
    }
    Ok(())
}

pub fn run() -> io::Result<()> {
    if atty::is(atty::Stream::Stdin) {
        return run_tty();
    }

    let mut context = SQLContext::new();
    let mut input: Vec<u8> = Vec::with_capacity(1024);
    let mut stdin = std::io::stdin();

    loop {
        input.clear();
        stdin.lock().read_until(b';', &mut input);

        let query = std::str::from_utf8(&input).unwrap_or("").trim();
        if query.is_empty() {
            break;
        }

        match execute_query(query, &mut context) {
            Ok(lf) => println!("{}", lf),
            Err(e) => eprintln!("{}", e),
        }
    }

    Ok(())
}
