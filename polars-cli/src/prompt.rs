use std::borrow::Cow;

use reedline::{Prompt, PromptHistorySearchStatus};

pub(super) struct SQLPrompt {}

impl Prompt for SQLPrompt {
    fn render_prompt_left(&self) -> std::borrow::Cow<str> {
        Cow::Borrowed("")
    }

    fn render_prompt_right(&self) -> std::borrow::Cow<str> {
        Cow::Borrowed("")
    }

    fn render_prompt_indicator(
        &self,
        _prompt_mode: reedline::PromptEditMode,
    ) -> std::borrow::Cow<str> {
        "〉".into()
    }

    fn render_prompt_multiline_indicator(&self) -> std::borrow::Cow<str> {
        Cow::Borrowed("::: ")
    }

    fn render_prompt_history_search_indicator(
        &self,
        history_search: reedline::PromptHistorySearch,
    ) -> std::borrow::Cow<str> {
        let prefix = match history_search.status {
            PromptHistorySearchStatus::Passing => "",
            PromptHistorySearchStatus::Failing => "failing ",
        };
        Cow::Owned(format!(
            "({}reverse-search: {}) ",
            prefix, history_search.term
        ))
    }
}
