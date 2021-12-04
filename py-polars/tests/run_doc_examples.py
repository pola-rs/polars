"""
Run all doctest examples inside the `polars` module using Python's built-in doctest module.

How to check examples: run this script, if exits with code 0, all is good. Otherwise, the errors will be reported.

How to modify behaviour for doctests:
1. if you would like code to be run and output checked: add the output below the code block
2. if you would like code to be run (and thus checked whether it actually not fails), but output not be checked: add
`# doctest: +IGNORE_RESULT` to the code block. You may still add example output.
3. if you would not like code to run: add `#doctest: +SKIP`. You may still add example output.

Notes:
* Doctest does not have a built-in IGNORE_RESULT directive. We have a number of tests where we want to ensure that the
code runs, but the output may be random by design, or not interesting for us to check. To allow for this behaviour, a
custom output checker has been created, see below.
* The doctests depend on the exact string representation staying the same. This may not be true in the future. For
instance, in the past, the printout of dataframes has changed from rounded corners to less rounded corners. To
facilitate such a change, whilst not immediately having to add IGNORE_RESULT directives everywhere or changing all
outputs, set `IGNORE_RESULT_ALL=True` below. Do note that this does mean no output is being checked anymore.
* This script will always take the code from this repo (see `src_dir` below), but the module used to run the code is
determined by the import below (see `import polars as pl`). For example, in CI, the import will import the installed
package, not the code in the repo. This is similar to how pytest works.
"""
import doctest
import sys
from pathlib import Path
from typing import Any

import polars

print(polars.__file__)
if __name__ == "__main__":
    # set to True to just run the code, and do not check any output. Will still report errors if the code is invalid
    IGNORE_RESULT_ALL = False

    # Below the implementation if the IGNORE_RESULT directive
    # You can ignore the result of a doctest by adding "doctest: +IGNORE_RESULT" into the code block
    # The difference with SKIP is that if the code errors on running, that will still be reported.
    IGNORE_RESULT = doctest.register_optionflag("IGNORE_RESULT")

    OutputChecker = doctest.OutputChecker

    class CustomOutputChecker(OutputChecker):
        def check_output(self, want: str, got: str, optionflags: Any) -> bool:
            if IGNORE_RESULT_ALL:
                return True
            if IGNORE_RESULT & optionflags:
                return True
            else:
                return OutputChecker.check_output(self, want, got, optionflags)

    doctest.OutputChecker = CustomOutputChecker  # type: ignore

    # We want to be relaxed about whitespace, but strict on True vs 1
    doctest.NORMALIZE_WHITESPACE = True
    doctest.DONT_ACCEPT_TRUE_FOR_1 = True

    # If REPORT_NDIFF is turned on, it will report on line by line, character by character, differences
    # The disadvantage is that you cannot just copy the output directly into the docstring
    # doctest.REPORT_NDIFF = True

    results_list = []
    src_dir = Path(polars.__file__).parent  # __file__ returns the __init__.py
    print(src_dir)
    for file in src_dir.rglob("*.py"):
        pretty_file_name = file.relative_to(src_dir)
        print(file)
        print(f"===== Testing {pretty_file_name} =====")
        # The globs arg means we do not have to do `import polars as pl` on each example
        # optionflags=1 enables the NORMALIZE_WHITESPACE and other options above
        res = doctest.testfile(
            str(file), module_relative=False, globs={"pl": polars}, optionflags=1
        )
        results_list.append(
            {
                "name": str(pretty_file_name),
                "attempted": res.attempted,
                "failed": res.failed,
            }
        )

    results = polars.DataFrame(results_list)
    print(results.sort("attempted", reverse=True))

    # we define success as no failures, and at least one doctest having run
    success_flag = (results["failed"].sum() == 0) and (results["attempted"].sum() > 0)

    sys.exit(int(not success_flag))
