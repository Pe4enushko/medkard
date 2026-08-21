#!/usr/bin/env bash
# Run standalone e2e tests whose paths match a caller-supplied grep pattern.
# Preserve each test's output, continue after failures, and mirror the complete
# terminal stream into a timestamped file under logs/. The final exit code is
# non-zero if any test failed.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
E2E_TESTS_DIR="$ROOT/e2e/tests"
E2E_PYTHON_BIN="${E2E_PYTHON_BIN:-python}"
PARALLEL_TMP_DIR=""

cleanup() {
    if [[ -n "$PARALLEL_TMP_DIR" && -d "$PARALLEL_TMP_DIR" ]]; then
        rm -rf -- "$PARALLEL_TMP_DIR"
    fi
}

trap cleanup EXIT

usage() {
    echo "Usage: $0 [--list] [--parallel [jobs]] <grep-regex>" >&2
}

list_only=false
parallel=false
parallel_jobs=4
args=()

while (( $# > 0 )); do
    case "$1" in
        --list)
            list_only=true
            shift
            ;;
        --parallel)
            parallel=true
            if [[ "${2:-}" =~ ^[0-9]+$ ]]; then
                parallel_jobs="$2"
                shift 2
            else
                shift
            fi
            ;;
        --parallel=*)
            parallel=true
            parallel_jobs="${1#*=}"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            args+=("$@")
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage
            exit 2
            ;;
        *)
            args+=("$1")
            shift
            ;;
    esac
done

if (( ${#args[@]} != 1 )) || [[ -z "${args[0]}" ]]; then
    usage
    exit 2
fi

if ! [[ "$parallel_jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "Parallel jobs must be a positive integer: $parallel_jobs" >&2
    exit 2
fi

pattern="${args[0]}"

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

run_suite() {
    local log_file="$1"
    local passed=0
    local failed=0
    local status=0
    local test_file
    local output_file
    local -a failed_tests=()
    local -a batch_pids=()
    local -a batch_tests=()
    local -a batch_outputs=()

    print_test_header() {
        local file="$1"
        echo
        echo "=============================================================================="
        echo "RUN  $file"
        echo "=============================================================================="
    }

    record_result() {
        local file="$1"
        local result="$2"
        if (( result == 0 )); then
            echo "OK   $file"
            ((passed += 1))
        else
            echo "FAIL $file (exit $result)"
            failed_tests+=("$file (exit $result)")
            ((failed += 1))
        fi
    }

    flush_parallel_batch() {
        local index
        local result
        for index in "${!batch_pids[@]}"; do
            if wait "${batch_pids[$index]}"; then
                result=0
            else
                result=$?
            fi
            print_test_header "${batch_tests[$index]}"
            command cat -- "${batch_outputs[$index]}"
            record_result "${batch_tests[$index]}" "$result"
        done
        batch_pids=()
        batch_tests=()
        batch_outputs=()
    }

    echo "Log: ${log_file#"$ROOT/"}"
    echo "Found ${#TESTS[@]} e2e test(s) matching: $pattern"
    echo "Python: $E2E_PYTHON_BIN"

    if [[ "$parallel" == true ]]; then
        echo "Mode: parallel (jobs=$parallel_jobs)"
        PARALLEL_TMP_DIR="$(mktemp -d)"

        for test_file in "${TESTS[@]}"; do
            output_file="$PARALLEL_TMP_DIR/${#batch_pids[@]}.out"
            "$E2E_PYTHON_BIN" "$ROOT/$test_file" >"$output_file" 2>&1 &
            batch_pids+=("$!")
            batch_tests+=("$test_file")
            batch_outputs+=("$output_file")

            if (( ${#batch_pids[@]} >= parallel_jobs )); then
                flush_parallel_batch
            fi
        done
        if (( ${#batch_pids[@]} > 0 )); then
            flush_parallel_batch
        fi
    else
        echo "Mode: sequential"
        for test_file in "${TESTS[@]}"; do
            print_test_header "$test_file"
            "$E2E_PYTHON_BIN" "$ROOT/$test_file"
            status=$?
            record_result "$test_file" "$status"
        done
    fi

    echo
    echo "=============================================================================="
    echo "E2e summary: passed=$passed failed=$failed total=${#TESTS[@]}"
    if (( failed > 0 )); then
        echo "Failed tests:"
        printf '  - %s\n' "${failed_tests[@]}"
        return 1
    fi

    echo "All matched e2e tests passed."
}

if ! mkdir -p "$ROOT/logs"; then
    echo "Cannot create e2e log directory: $ROOT/logs" >&2
    exit 1
fi
log_stamp="$(date '+%Y-%m-%d_%H-%M-%S')"
log_file="$ROOT/logs/e2e-${log_stamp}-$$.log"

run_suite "$log_file" 2>&1 | tee -- "$log_file"
pipeline_statuses=("${PIPESTATUS[@]}")
if (( pipeline_statuses[1] != 0 )); then
    exit "${pipeline_statuses[1]}"
fi
exit "${pipeline_statuses[0]}"
