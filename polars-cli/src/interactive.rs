#[cfg(feature = "highlight")]
use crate::highlighter::SQLHighlighter;
use crate::OutputMode;
#[global_allocator]
#[cfg(target_os = "linux")]
static ALLOC: Jemalloc = Jemalloc;

use std::env;
use std::path::PathBuf;

use clap::crate_version;
use polars::df;
use polars::sql::SQLContext;
use reedline::{FileBackedHistory, Reedline, Signal};

use crate::prompt::SQLPrompt;

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

// Command: help
fn print_help() {
    use polars::prelude::*;
    let df = df! {
        "command" => [
          ".help",
          ".exit"
      ],
        "description" => [
          "Display this help.",
          "Exit this program",
      ]
    }
    .unwrap();
    println!("{}", df);
}

enum PolarsCommand {
    Help,
    Exit,
}

impl PolarsCommand {
    fn execute(&self) -> std::io::Result<()> {
        match self {
            PolarsCommand::Help => {
                print_help();
                Ok(())
            }
            PolarsCommand::Exit => Ok(()),
        }
    }
}
impl TryFrom<(&str, &str)> for PolarsCommand {
    type Error = std::io::Error;

    fn try_from(value: (&str, &str)) -> Result<Self, Self::Error> {
        let (cmd, _) = value;
        match cmd {
            ".help" | "?" => Ok(PolarsCommand::Help),
            ".exit" | "q" => Ok(PolarsCommand::Exit),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Unknown command",
            )),
        }
    }
}

pub(super) fn run_tty(output_mode: OutputMode) -> std::io::Result<()> {
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

                        cmd.execute()?
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
