#!/bin/sh
# For `release-polars-cli.yml`. POSIX sh, for Alpine.
#
# Usage: smoke-test-polars-cli.sh <polars executable> <commit it was built from>
set -eu

polars=$1
commit=$2
cd "$(mktemp -d)"

fail() {
  echo "smoke test failed: $*" >&2
  exit 1
}

version=$("$polars" --version)
echo "$version"
case "$version" in
  *"($(printf '%s' "$commit" | cut -c 1-10))") ;;
  *) fail "--version does not name commit $commit" ;;
esac

[ "$("$polars" --format csv -c 'SELECT 1 AS a')" = "$(printf 'a\n1')" ] || fail "SELECT 1"

printf 'a,b\n1,x\n2,y\n' > in.csv
"$polars" -c "COPY (SELECT * FROM 'in.csv') TO 'out.parquet'"
[ "$("$polars" --format csv -c "SELECT sum(a) AS s FROM 'out.parquet'")" = "$(printf 's\n3')" ] ||
  fail "COPY to Parquet"

"$polars" --format parquet -c 'SELECT 1 AS a' > a.parquet
[ "$(head -c 4 a.parquet)" = PAR1 ] || fail "--format parquet"

# TLS and root certificates. Retried: the server can refuse a burst of requests.
url="https://raw.githubusercontent.com/pola-rs/polars/$commit/examples/datasets/foods1.csv"
tries=1
until [ "$("$polars" --format csv -c "SELECT count(*) AS n FROM '$url'")" = "$(printf 'n\n27')" ]; do
  [ "$tries" -lt 3 ] || fail "reading $url"
  tries=$((tries + 1))
  sleep 10
done

status=0
"$polars" -c 'SELECT nope' 2> /dev/null || status=$?
[ "$status" -eq 1 ] || fail "a failing statement exits with $status, not 1"

status=0
"$polars" --bogus 2> /dev/null || status=$?
[ "$status" -eq 2 ] || fail "a wrong command line exits with $status, not 2"

echo "smoke test passed"
