#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 ]]; then
    echo "usage: $0 BASE_CHECKOUT LAZY_CHECKOUT OUTPUT_DIR" >&2
    exit 2
fi
base=$(cd "$1" && pwd)
lazy=$(cd "$2" && pwd)
out=$(mkdir -p "$3" && cd "$3" && pwd)
here=$(cd "$(dirname "$0")" && pwd)

diff_cases=${DIFF_CASES:-20000}
oracle_cases=${ORACLE_CASES:-5000}
asan_positive_cases=${ASAN_POSITIVE_CASES:-3500}
asan_zero_cases=${ASAN_ZERO_CASES:-4000}

"$here/compile_core.sh" "$base" "$out/differential-base" "$here/validate_differential.cc" -O3 -DNDEBUG
"$here/compile_core.sh" "$lazy" "$out/differential-lazy" "$here/validate_differential.cc" -O3 -DNDEBUG -DPM_LAZY_TOP
for mode in positive zero; do
    "$out/differential-base" 12345 "$diff_cases" "$mode" > "$out/base-$mode.csv"
    "$out/differential-lazy" 12345 "$diff_cases" "$mode" > "$out/lazy-$mode.csv"
    "$here/compare_differential.py" "$out/base-$mode.csv" "$out/lazy-$mode.csv"
done

"$here/compile_core.sh" "$lazy" "$out/oracle-lazy" "$here/validate_oracle.cc" -O2 -DNDEBUG -DPM_LAZY_TOP
"$out/oracle-lazy" 67890 "$oracle_cases"

"$here/compile_core.sh" "$lazy" "$out/differential-lazy-asan" "$here/validate_differential.cc" \
    -O1 -g -DPM_LAZY_TOP -fsanitize=address,undefined -fno-omit-frame-pointer
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    "$out/differential-lazy-asan" 24680 "$asan_positive_cases" positive > "$out/asan-positive.csv"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    "$out/differential-lazy-asan" 13579 "$asan_zero_cases" zero > "$out/asan-zero.csv"

echo "validation complete"
