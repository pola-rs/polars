import doctest
import sys
from pathlib import Path

import polars

if __name__ == "__main__":
    results_list = []
    src_dir = Path(__file__).parent / "polars"
    for file in src_dir.rglob("*.py"):
        pretty_file_name = file.relative_to(src_dir)

        print(f"===== Testing {pretty_file_name} =====")
        res = doctest.testfile(str(file), globs={"pl": polars})
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
