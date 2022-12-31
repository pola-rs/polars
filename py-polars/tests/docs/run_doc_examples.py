"""
Run all doctest examples of the `polars` module using Python's built-in doctest module.

How to check examples: run this script, if exits with code 0, all is good. Otherwise,
the errors will be reported.

How to modify behaviour for doctests:
1. if you would like code to be run and output checked: add the output below the code
   block
2. if you would like code to be run (and thus checked whether it actually not fails),
   but output not be checked: add `# doctest: +IGNORE_RESULT` to the code block. You may
   still add example output.
3. if you would not like code to run: add `#doctest: +SKIP`. You may still add example
   output.

Notes
-----
* Doctest does not have a built-in IGNORE_RESULT directive. We have a number of tests
  where we want to ensure that the code runs, but the output may be random by design, or
  not interesting for us to check. To allow for this behaviour, a custom output checker
  has been created, see below.
* The doctests depend on the exact string representation staying the same. This may not
  be true in the future. For instance, in the past, the printout of dataframes has
  changed from rounded corners to less rounded corners. To facilitate such a change,
  whilst not immediately having to add IGNORE_RESULT directives everywhere or changing
  all outputs, set `IGNORE_RESULT_ALL=True` below. Do note that this does mean no output
  is being checked anymore.

"""
from __future__ import annotations

import doctest
import importlib
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any, Iterator

import polars


def doctest_teardown(d: doctest.DocTest) -> None:
    # don't let config changes leak between tests
    polars.Config.restore_defaults()


def modules_in_path(p: Path) -> Iterator[ModuleType]:
    for file in p.rglob("*.py"):
        # Construct path as string for import, for instance "internals.frame"
        # The -3 drops the ".py"
        file_name_import = ".".join(file.relative_to(p).parts)[:-3]
        temp_module = importlib.import_module(p.name + "." + file_name_import)
        yield temp_module


if __name__ == "__main__":
    # set to True to just run the code, and do not check any output.
    # Will still report errors if the code is invalid
    IGNORE_RESULT_ALL = False

    # Below the implementation of the IGNORE_RESULT directive
    # You can ignore the result of a doctest by adding "doctest: +IGNORE_RESULT" into
    # the code block. The difference with SKIP is that if the code errors on running,
    # that will still be reported.
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

    doctest.OutputChecker = CustomOutputChecker  # type: ignore[misc]

    # We want to be relaxed about whitespace, but strict on True vs 1
    doctest.NORMALIZE_WHITESPACE = True
    doctest.DONT_ACCEPT_TRUE_FOR_1 = True

    # If REPORT_NDIFF is turned on, it will report on line by line, character by
    # character, differences. The disadvantage is that you cannot just copy the output
    # directly into the docstring.
    # doctest.REPORT_NDIFF = True

    # __file__ returns the __init__.py, so grab the parent
    src_dir = Path(polars.__file__).parent

    with TemporaryDirectory() as tmpdir:
        # collect all tests
        tests = [
            doctest.DocTestSuite(
                m,
                extraglobs={"pl": polars, "dirpath": Path(tmpdir)},
                optionflags=1,
                tearDown=doctest_teardown,
            )
            for m in modules_in_path(src_dir)
        ]
        test_suite = unittest.TestSuite(tests)

        # Ensure that we clean up any artifacts produced by the doctests
        # with patch(polars.DataFrame.write_csv):
        # run doctests and report
        result = unittest.TextTestRunner().run(test_suite)
        success_flag = (result.testsRun > 0) & (len(result.failures) == 0)
        sys.exit(int(not success_flag))
