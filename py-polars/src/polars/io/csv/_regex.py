from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from re import Pattern


def normalize_regex_csv(
    text: str,
    separator: Pattern[str],
    quote_char: str | None,
    eol_char: str,
    comment_prefix: str | None,
) -> tuple[str, str]:
    """Replace regex separators outside quoted fields with an unused ASCII byte."""
    if separator.search("") is not None:
        msg = "CSV regex separators must not match an empty string"
        raise ValueError(msg)

    excluded = "\r\n" + eol_char + (quote_char or "") + (comment_prefix or "")
    for code in (31, *range(1, 128)):
        delimiter = chr(code)
        if delimiter not in text and delimiter not in excluded:
            break
    else:
        msg = "No unused ASCII character is available for the regex CSV separator"
        raise ValueError(msg)

    trim = separator.pattern == r"\s+"
    quoted = False
    field_start = True
    result: list[str] = []
    for line in text.split(eol_char):
        if (
            not quoted
            and comment_prefix is not None
            and line.startswith(comment_prefix)
        ):
            result.append(line)
            continue

        carriage_return = "\r" if eol_char == "\n" and line.endswith("\r") else ""
        body = line[:-1] if carriage_return else line
        parts: list[str] = []
        cursor = position = 0
        while position <= len(body):
            if quoted and quote_char:
                end_quote = body.find(quote_char, position)
                if end_quote == -1:
                    break
                position = end_quote + 1
                if position < len(body) and body[position] == quote_char:
                    position += 1
                    continue
                quoted = False
                field_start = False
                continue
            if (
                field_start
                and quote_char
                and position < len(body)
                and body[position] == quote_char
            ):
                quoted = True
                field_start = False
                position += 1
                continue
            match = separator.search(body, position)
            if match is None:
                break
            start, end = match.span()
            if start == end:
                msg = "CSV regex separators must not match an empty string"
                raise ValueError(msg)
            parts.append(body[cursor:start])
            parts.append("" if trim and (start == 0 or end == len(body)) else delimiter)
            cursor = position = end
            field_start = True
        parts.append(body[cursor:])
        result.append("".join(parts) + carriage_return)
        if not quoted:
            field_start = True

    return eol_char.join(result), delimiter
