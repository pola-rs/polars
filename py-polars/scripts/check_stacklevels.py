"""
Check that warnings decorators are set with the correct stacklevel.

Use `find_stacklevel()` instead of setting it manually.
"""
import ast
import subprocess
import sys
from ast import NodeVisitor

# Files in which it's OK to set the stacklevel manually.
# `git ls-files` lists files with forwards-slashes
# even on Windows, so it's OK to list them like that.
EXCLUDE = frozenset(["polars/utils/polars_version.py"])


class StackLevelChecker(NodeVisitor):
    def __init__(self, file) -> None:
        self.file = file
        self.violations = set()

    def visit_Call(self, node: ast.Call) -> None:
        for keyword in node.keywords:
            if keyword.arg == "stacklevel" and isinstance(keyword.value, ast.Constant):
                self.violations.add(
                    f"{self.file}:{keyword.lineno}:{keyword.col_offset}: "
                    f"stacklevel set manually. "
                    f"Please use `find_stacklevel()` instead."
                )
        self.generic_visit(node)


if __name__ == "__main__":
    files = subprocess.run(
        ["git", "ls-files", "polars"], capture_output=True, text=True
    ).stdout.split()
    ret = 0
    for file in files:
        if file in EXCLUDE:
            continue
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
