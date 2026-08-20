#!/usr/bin/env bash
# Run every standalone diagnosis-graph e2e, preserving each test's output and
# continuing after failures. The final exit code is non-zero if any test failed.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUDIT_TESTS_DIR="$ROOT/e2e/tests/audit"
E2E_PYTHON_BIN="${E2E_PYTHON_BIN:-python}"

mapfile -t TESTS < <(
    rg --files "$AUDIT_TESTS_DIR" \
        | grep '/test_diagnosis_graph_.*\.py$' \
        | sort
)

if (( ${#TESTS[@]} == 0 )); then
    echo "No diagnosis-graph e2e tests found in $AUDIT_TESTS_DIR" >&2
    exit 2
fi

if [[ "${1:-}" == "--list" ]]; then
    printf '%s\n' "${TESTS[@]}"
    exit 0
fi

if (( $# > 0 )); then
    echo "Usage: $0 [--list]" >&2
    exit 2
fi

passed=0
failed=0
failed_tests=()

echo "Found ${#TESTS[@]} diagnosis-graph e2e test(s)."
echo "Python: $E2E_PYTHON_BIN"

for test_file in "${TESTS[@]}"; do
    relative_path="${test_file#"$ROOT/"}"
    echo
    echo "=============================================================================="
    echo "RUN  $relative_path"
    echo "=============================================================================="

    "$E2E_PYTHON_BIN" "$test_file"
    status=$?

    if (( status == 0 )); then
        echo "OK   $relative_path"
        ((passed += 1))
    else
        echo "FAIL $relative_path (exit $status)"
        failed_tests+=("$relative_path (exit $status)")
        ((failed += 1))
    fi
done

echo
echo "=============================================================================="
echo "Diagnosis graph e2e summary: passed=$passed failed=$failed total=${#TESTS[@]}"
if (( failed > 0 )); then
    echo "Failed tests:"
    printf '  - %s\n' "${failed_tests[@]}"
    exit 1
fi

echo "All diagnosis-graph e2e tests passed."
