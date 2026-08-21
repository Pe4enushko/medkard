#!/usr/bin/env bash
# Run standalone e2e tests whose paths match a caller-supplied grep pattern.
# Preserve each test's output and continue after failures. The final exit code
# is non-zero if any test failed.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
E2E_TESTS_DIR="$ROOT/e2e/tests"
E2E_PYTHON_BIN="${E2E_PYTHON_BIN:-python}"

usage() {
    echo "Usage: $0 [--list] <grep-regex>" >&2
}

list_only=false
if [[ "${1:-}" == "--list" ]]; then
    list_only=true
    shift
fi

if (( $# != 1 )) || [[ -z "$1" ]]; then
    usage
    exit 2
fi

pattern="$1"

# grep exits with 2 for an invalid regular expression.
grep -E -- "$pattern" /dev/null >/dev/null 2>&1
pattern_status=$?
if (( pattern_status == 2 )); then
    echo "Invalid grep regular expression: $pattern" >&2
    exit 2
fi

mapfile -t TESTS < <(
    cd "$ROOT" || exit
    find "${E2E_TESTS_DIR#"$ROOT/"}" -type f -name 'test_*.py' \
        | grep -E -- "$pattern" \
        | sort
)

if (( ${#TESTS[@]} == 0 )); then
    echo "No e2e tests matched grep regular expression: $pattern" >&2
    exit 2
fi

if [[ "$list_only" == true ]]; then
    printf '%s\n' "${TESTS[@]}"
    exit 0
fi

passed=0
failed=0
failed_tests=()

echo "Found ${#TESTS[@]} e2e test(s) matching: $pattern"
echo "Python: $E2E_PYTHON_BIN"

for test_file in "${TESTS[@]}"; do
    echo
    echo "=============================================================================="
    echo "RUN  $test_file"
    echo "=============================================================================="

    "$E2E_PYTHON_BIN" "$ROOT/$test_file"
    status=$?

    if (( status == 0 )); then
        echo "OK   $test_file"
        ((passed += 1))
    else
        echo "FAIL $test_file (exit $status)"
        failed_tests+=("$test_file (exit $status)")
        ((failed += 1))
    fi
done

echo
echo "=============================================================================="
echo "E2e summary: passed=$passed failed=$failed total=${#TESTS[@]}"
if (( failed > 0 )); then
    echo "Failed tests:"
    printf '  - %s\n' "${failed_tests[@]}"
    exit 1
fi

echo "All matched e2e tests passed."
