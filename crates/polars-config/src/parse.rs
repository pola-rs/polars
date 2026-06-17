use polars_error::polars_warn;

use crate::{Engine, ResolveMode, SpillFormat};

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

pub fn parse_f64(var: &str, val: &str) -> Option<f64> {
    let ret = val.trim_ascii().parse::<f64>().ok();
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

pub fn parse_spill_format(var: &str, val: &str) -> Option<SpillFormat> {
    match val.trim_ascii().parse::<SpillFormat>() {
        Ok(x) => Some(x),
        Err(e) => {
            polars_warn!("illegal value '{val}' found while parsing option '{var}' ({e})");
            None
        },
    }
}

pub fn parse_resolve_mode(var: &str, val: &str) -> Option<ResolveMode> {
    match val.trim_ascii().parse::<ResolveMode>() {
        Ok(x) => Some(x),
        Err(e) => {
            polars_warn!("illegal value '{val}' found while parsing option '{var}' ({e})");
            None
        },
    }
}
