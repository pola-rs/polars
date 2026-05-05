# /// script
# dependencies = [
#   "GitPython",
# ]
# ///

import git
import re

REGEX = re.compile("target|source")

repo = git.Repo(search_parent_directories=True)
patch = repo.git.diff("main", "--unified=0", "--no-color")

current_file = None
old_file = None
old_line = new_line = 0

for raw in patch.splitlines():
    if raw.startswith("diff --git "):
        current_file = None
        old_file = None
        continue

    if raw.startswith("--- "):
        old_file = raw[6:] if raw.startswith("--- a/") else raw[4:]
        continue

    if raw.startswith("+++ "):
        current_file = raw[6:] if raw.startswith("+++ b/") else raw[4:]
        continue

    if raw.startswith("@@ "):
        m = re.search(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
        if not m:
            continue
        old_line = int(m.group(1))
        new_line = int(m.group(2))
        continue

    if raw.startswith("+") and not raw.startswith("+++"):
        if REGEX.search(raw[1:]):
            print(f"{current_file}:{new_line}:1:+{raw[1:]}")
        new_line += 1
    elif raw.startswith("-") and not raw.startswith("---"):
        target = old_file or current_file
        # print(f"{target}:{old_line}:1:-{raw[1:]}")
        old_line += 1
    elif raw.startswith(" "):
        old_line += 1
        new_line += 1
