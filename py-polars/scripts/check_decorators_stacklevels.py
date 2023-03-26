"""
Check that warnings decorators are set with the correct stacklevel.

By default, `deprecated_nonkeyword_arguments` has stacklevel 2, and `deprecated_alias`
stacklevel 3. If called together, then the stacklevel may need setting manually for
some.
"""
import ast
import subprocess
import sys
from ast import NodeVisitor

DEFAULT_STACKLEVEL = {
    "deprecate_nonkeyword_arguments": 2,
    "deprecated_alias": 3,
}


def _stacklevel_is_correct(keywords, expected_stacklevel):
    for keyword in keywords:
        if (
            keyword.arg == "stacklevel"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == expected_stacklevel
        ):
            return True
    return False


class StackLevelChecker(NodeVisitor):
    def __init__(self, file) -> None:
        self.file = file
        self.violations = set()

    def _check_decorator_stacklevel(self, decorator: ast.Call, idx: int):
        if (
            isinstance(decorator.func, ast.Name)
            and decorator.func.id in DEFAULT_STACKLEVEL
            and idx != 0
        ):
            decorator_name = decorator.func.id
            expected_stacklevel = idx + DEFAULT_STACKLEVEL[decorator_name]
            if not _stacklevel_is_correct(decorator.keywords, expected_stacklevel):
                self.violations.add(
                    f"{self.file}:{decorator.lineno}:{decorator.col_offset}: "
                    f"found incorrect stacklevel. "
                    f"Please set `{decorator_name}`'s stacklevel to {expected_stacklevel}"
                )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        call_decorators = [
            decorator
            for decorator in node.decorator_list
            if isinstance(decorator, ast.Call)
        ]
        for idx, decorator in enumerate(call_decorators):
            self._check_decorator_stacklevel(decorator, idx)
        self.generic_visit(node)


if __name__ == "__main__":
    files = subprocess.run(
        ["git", "ls-files", "polars"], capture_output=True, text=True
    ).stdout.split()
    ret = 0
    for file in files:
        if not file.endswith(".py"):
            continue
        with open(file) as fd:
            content = fd.read()
        tree = ast.parse(content)
        stacklevel_checker = StackLevelChecker(file)
        stacklevel_checker.visit(tree)
        for violation in stacklevel_checker.violations:
            print(violation)
            ret |= 1
    sys.exit(ret)
