import doctest
import sys
from pathlib import Path

import polars

if __name__ == "__main__":
    # set to True to just run the code, and do not check any output. Will still report errors if the code is invalid
    IGNORE_RESULT_ALL = False

    # Below the implementation if the IGNORE_RESULT directive
    # You can ignore the result of a doctest by adding "doctest: +IGNORE_RESULT" into the code block
    # The difference with SKIP is that if the code errors on running, that will still be reported.
    IGNORE_RESULT = doctest.register_optionflag('IGNORE_RESULT')

    OutputChecker = doctest.OutputChecker
    class CustomOutputChecker(OutputChecker):
        def check_output(self, want, got, optionflags):
            if IGNORE_RESULT_ALL:
                return True
            if IGNORE_RESULT & optionflags:
                return True
            else:
                return OutputChecker.check_output(self, want, got, optionflags)


    doctest.OutputChecker = CustomOutputChecker

    # We want to be relaxed about whitespace, but strict on True vs 1
    doctest.NORMALIZE_WHITESPACE = True
    doctest.DONT_ACCEPT_TRUE_FOR_1 = True

    # If REPORT_NDIFF is turned on, it will report on line by line, character by character, differences
    # The disadvantage is that you cannot just copy the output directly into the docstring
    # doctest.REPORT_NDIFF = True

    results_list = []
    src_dir = Path(__file__).parent / "polars"
    for file in src_dir.rglob("*.py"):
        pretty_file_name = file.relative_to(src_dir)

        print(f"===== Testing {pretty_file_name} =====")
        res = doctest.testfile(
            str(file), globs={"pl": polars}, optionflags=1
        )  # optionflags=1 enables the NORMALIZE_WHITESPACE and other options above
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
