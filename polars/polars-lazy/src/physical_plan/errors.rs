macro_rules! expression_err {
    ($msg:expr, $source:expr, $error:ident) => {{
        let msg = format!(
            "{}\n\n> Error originated in expression: '{:?}'",
            $msg, $source
        );
        PolarsError::$error(msg.into())
    }};
}

pub(super) use expression_err;
