use std::env;
use std::io::Cursor;
use std::path::PathBuf;

use clap::crate_version;
use once_cell::sync::Lazy;
use polars::df;
use polars::prelude::{DataFrame, *};
use polars::sql::SQLContext;
use reedline::{FileBackedHistory, Reedline, Signal};

#[cfg(feature = "highlight")]
use crate::highlighter::SQLHighlighter;
use crate::prompt::SQLPrompt;
use crate::{OutputMode, SerializableContext};

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

static HELP_DF: Lazy<DataFrame> = Lazy::new(|| {
    df! {
        "command" => [
            ".exit",
            ".quit",
            ".save FILE",
            ".open FILE",
            ".help",
      ],
        "description" => [
            "Exit this program",
            "Exit this program (alias for .exit)",
            "Save the current state of the database to FILE",
            "Open a database from FILE",
            "Display this help.",
      ]
    }
    .unwrap()
});

// Command: help
fn print_help() {
    let _tmp_env = (
        tmp_env::set_var("POLARS_FMT_TABLE_FORMATTING", "NOTHING"),
        tmp_env::set_var("POLARS_FMT_TABLE_HIDE_COLUMN_DATA_TYPES", "1"),
        tmp_env::set_var("POLARS_FMT_TABLE_HIDE_DATAFRAME_SHAPE_INFORMATION", "1"),
        tmp_env::set_var("POLARS_FMT_TABLE_HIDE_COLUMN_NAMES", "1"),
        tmp_env::set_var("POLARS_FMT_STR_LEN", "80"),
    );
    let df: &DataFrame = &HELP_DF;
    println!("{}", df);
}

pub(super) enum PolarsCommand {
    Help,
    Exit,
    Save(PathBuf),
    Open(PathBuf),
    Unknown(String),
}

impl PolarsCommand {
    fn execute_and_print(&self, ctx: &mut SQLContext) {
        match self.execute(ctx) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }
    }
    fn execute(&self, ctx: &mut SQLContext) -> std::io::Result<()> {
        match self {
            PolarsCommand::Help => {
                print_help();
                Ok(())
            }
            PolarsCommand::Exit => Ok(()),
            PolarsCommand::Save(buf) => {
                let serializable_ctx: SerializableContext = ctx.into();
                let mut w: Vec<u8> = vec![];
                ciborium::ser::into_writer(&serializable_ctx, &mut w).map_err(|e| {
                    std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Serialization error: {}", e),
                    )
                })?;

                *ctx = serializable_ctx.into();
                std::fs::write(buf, w)?;
                Ok(())
            }
            PolarsCommand::Open(buf) => {
                let db = std::fs::read(buf)?;
                let cursor = Cursor::new(db);
                let serializable_ctx: SerializableContext = ciborium::de::from_reader(cursor)
                    .map_err(|e| {
                        std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("Deserialization error: {}", e),
                        )
                    })?;
                *ctx = serializable_ctx.into();
                Ok(())
            }
            PolarsCommand::Unknown(cmd) => {
                println!(r#"Unknown command: "{cmd}".  Enter ".help" for help"#);
                Ok(())
            }
        }
    }
}

impl TryFrom<(&str, &str)> for PolarsCommand {
    type Error = std::io::Error;

    fn try_from(value: (&str, &str)) -> Result<Self, Self::Error> {
        let (cmd, arg) = value;

        match cmd {
            ".help" | "?" => Ok(PolarsCommand::Help),
            ".exit" | ".quit" => Ok(PolarsCommand::Exit),
            ".save" => Ok(PolarsCommand::Save(arg.trim().into())),
            ".open" => Ok(PolarsCommand::Open(arg.trim().into())),
            unknown => Ok(PolarsCommand::Unknown(unknown.to_string())),
        }
    }
}

pub(super) fn run_tty(output_mode: OutputMode) -> std::io::Result<()> {
    let history = Box::new(
        FileBackedHistory::with_file(100, get_history_path())
            .expect("Error configuring history with file"),
    );

    let mut line_editor = Reedline::create().with_history(history);

    #[cfg(feature = "highlight")]
    {
        let sql_highlighter = SQLHighlighter {};
        line_editor = line_editor.with_highlighter(Box::new(sql_highlighter));
    }

    let mut context = SQLContext::new();

    println!("Polars CLI v{}", crate_version!());
    println!("Type .help for help.");

    let prompt = SQLPrompt {};
    let mut is_exit_cmd = false;
    let mut scratch = String::with_capacity(1024);

    loop {
        let sig = line_editor.read_line(&prompt);
        match sig {
            Ok(Signal::Success(buffer)) => {
                is_exit_cmd = false;
                match buffer.as_str() {
                    special_cmd if buffer.starts_with('.') => {
                        let cmd: PolarsCommand = special_cmd
                            .split_at(special_cmd.find(' ').unwrap_or(special_cmd.len()))
                            .try_into()?;

                        if let PolarsCommand::Exit = cmd {
                            break;
                        };

                        cmd.execute_and_print(&mut context)
                    }
                    _ => {
                        let mut parts = buffer.splitn(2, ';');
                        let first = parts.next().unwrap();
                        scratch.push_str(first);

                        let second = parts.next();
                        if second.is_some() {
                            output_mode.execute_query(&scratch, &mut context);
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
            _ => {
                is_exit_cmd = false;
            }
        }
    }
    Ok(())
}

impl PolarsCommand {
    pub(super) fn keywords() -> Vec<&'static str> {
        vec!["exit", "quit", "save", "open", "help"]
    }
}
