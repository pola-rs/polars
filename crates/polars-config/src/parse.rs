use polars_error::polars_warn;

use crate::Engine;

pub fn parse_bool(var: &str, val: &str) -> Option<bool> {
    match val.trim_ascii() {
        "0" | "false" => Some(false),
        "1" | "true" => Some(true),
        _ => {
            polars_warn!("illegal value '{val}' found while parsing option '{var}'");
            None
        },
    }
}

pub fn parse_u64(var: &str, val: &str) -> Option<u64> {
    let ret = val.trim_ascii().parse::<u64>().ok();
    if ret.is_none() {
        polars_warn!("illegal value '{val}' found while parsing option '{var}'");
    }
    ret
}

pub fn parse_engine(var: &str, val: &str) -> Option<Engine> {
    match val.trim_ascii().parse::<Engine>() {
        Ok(x) => Some(x),
        Err(e) => {
            polars_warn!("illegal value '{val}' found while parsing option '{var}' ({e})");
            None
        },
    }
}
