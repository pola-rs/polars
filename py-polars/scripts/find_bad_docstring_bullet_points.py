"""
Find cases where docstring bullet points will not render correctly in the documentation.

Do this:
```
Here is a list of things:
- thing 1
- thing 2
```
Not this:
```
Here is a list of things:
- thing 1
- thing 2
```.

Source: https://gist.github.com/MarcoGorelli/6d95b32436e43fa37288b1dd3030425d
"""
import ast
import subprocess
import sys
from pathlib import Path


def _find_bad_docstring_bullet_points(file_path: str) -> bool:
    content = Path(file_path).read_text(encoding="utf-8")
    parsed_tree = ast.parse(content)
    has_issue = False

    for node in ast.walk(parsed_tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            function_or_class_body = node.body
            if (
                len(function_or_class_body) > 0
                and isinstance(function_or_class_body[0], ast.Expr)
                and isinstance(function_or_class_body[0].value, ast.Constant)
            ):
                docstring = function_or_class_body[0].value.value

                if isinstance(docstring, str):
                    lines = docstring.split("\n")
                    for i, line in enumerate(lines[1:], start=1):
                        previous_line = lines[i - 1]
                        current_line_indentation = len(line) - len(line.lstrip())
                        previous_line_indentation = len(lines[i - 1]) - len(
                            lines[i - 1].lstrip()
                        )

                        if (
                            current_line_indentation == previous_line_indentation
                            and not previous_line.lstrip().startswith("*")
                            and not previous_line.lstrip().startswith("-")
                            and (
                                line.lstrip().startswith("- ")
                                or line.lstrip().startswith("* ")
                            )
                        ):
                            line_number = (
                                lines.index(line)
                                + function_or_class_body[0].value.lineno
                            )
                            print(
                                f"{file_path}:{line_number}:9: Found docstring bullet points which will not render."
                            )
                            has_issue = True

    return has_issue


if __name__ == "__main__":
    git_files = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True
    ).stdout.split()
    has_issues = False

    for file_path in git_files:
        if file_path.endswith(".py"):
            has_issues |= _find_bad_docstring_bullet_points(file_path)

    if has_issues:
        sys.exit(1)
    else:
        sys.exit(0)
