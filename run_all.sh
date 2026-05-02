#!/bin/bash
# aricode-ml: run_all.sh — orchestrate the full ML test suite end-to-end.
#
# Two test categories are exercised:
#
#   1. Unit tests in tests/*.ari — recompile each with the current
#      compiler, run the resulting binary, look for a line ending in
#      `_OK` on stdout (case-sensitive).  These cover attention KV
#      semantics, sampling, RoPE edges, codegen quirks, etc.
#
#   2. Example regressions in examples/<name>/run_test.sh — each
#      shell script handles its own pack + compile + run + compare.
#      We delegate to it and grep its combined output for `_OK`.
#
# Per-test bookkeeping:
#
#   * Wall-clock time captured with `date +%s.%N` around each run.
#   * Full stdout/stderr stashed in /tmp/aricode_runall_<test>.log
#     so failing tests can be inspected after the fact.
#   * Hard timeout of 5 min per test (configurable via TIMEOUT_SECS);
#     killed tests are reported as TIMEOUT.
#   * Output is colour-coded: green for pass, red for fail/timeout,
#     yellow for skip.
#
# Skipping rules:
#
#   * distilbert_sst2 is skipped if SKIP_DOWNLOADS=1 or if the HF
#     model cache is absent (it auto-redownloads transformers weights).
#   * gpt2_small is skipped unless its synth.pt (~650 MB) is already
#     present locally — re-downloading + re-keying takes minutes and
#     the user explicitly does not want it triggered.
#   * Anything else where prepare.py fails for missing data is logged
#     as SKIP rather than FAIL.
#
# Usage:
#
#   bash run_all.sh                  # full suite
#   SKIP_DOWNLOADS=1 bash run_all.sh # skip network-dependent tests
#   TIMEOUT_SECS=600  bash run_all.sh # bump the per-test timeout
#
# Idempotent: running it twice produces the same summary modulo the
# wall-clock numbers.

set -uo pipefail
# Note: NO `-e` here — individual failing tests must not abort the run.

# ──────────────────────────────────────────────────────────── config ──

ML_ROOT="$(cd "$(dirname "$0")" && pwd)"
ARIC="${ARIC:-/home/serverbig/git-proyect/aricoderoot/aricode/src/compiler/aric}"
PYTHON="${PYTHON:-/tmp/aricode_venv/bin/python}"
TIMEOUT_SECS="${TIMEOUT_SECS:-300}"
SKIP_DOWNLOADS="${SKIP_DOWNLOADS:-0}"
LOG_DIR="${LOG_DIR:-/tmp}"

export PYTHON   # run_test.sh scripts honour ${PYTHON:-...}.

# ANSI colours — disable if stdout isn't a tty.
if [[ -t 1 ]]; then
    C_GREEN=$'\033[1;32m'
    C_RED=$'\033[1;31m'
    C_YELLOW=$'\033[1;33m'
    C_DIM=$'\033[2m'
    C_RESET=$'\033[0m'
else
    C_GREEN=""; C_RED=""; C_YELLOW=""; C_DIM=""; C_RESET=""
fi

# ───────────────────────────────────────────────────────────── state ──

declare -a NAMES STATUSES TIMES NOTES
PASS=0
FAIL=0
SKIP=0
TIMEOUT=0
SUITE_START=$(date +%s.%N)

# ─────────────────────────────────────────────────────────── helpers ──

# elapsed_secs(start, end)  — pretty-print a `date +%s.%N` delta in
# seconds (2 decimals).  Prefers the project's venv python, falls back
# to system python3.
PY_FOR_TIMING="$PYTHON"
[[ -x "$PY_FOR_TIMING" ]] || PY_FOR_TIMING="python3"

elapsed_secs() {
    "$PY_FOR_TIMING" -c 'import sys; print(f"{float(sys.argv[2]) - float(sys.argv[1]):.2f}")' "$1" "$2"
}

# Records one test's outcome.
# args: name, status (PASS|FAIL|SKIP|TIMEOUT), elapsed_secs, note
record() {
    local name="$1" status="$2" elapsed="$3" note="${4:-}"
    NAMES+=("$name")
    STATUSES+=("$status")
    TIMES+=("$elapsed")
    NOTES+=("$note")

    case "$status" in
        PASS)
            PASS=$((PASS+1))
            printf '%s✓%s %-44s %s%6ss%s  %s\n' \
                "$C_GREEN" "$C_RESET" "$name" "$C_DIM" "$elapsed" "$C_RESET" "$note"
            ;;
        FAIL)
            FAIL=$((FAIL+1))
            printf '%s✗%s %-44s %s%6ss%s  %s%s%s\n' \
                "$C_RED" "$C_RESET" "$name" "$C_DIM" "$elapsed" "$C_RESET" \
                "$C_RED" "$note" "$C_RESET"
            ;;
        TIMEOUT)
            TIMEOUT=$((TIMEOUT+1))
            FAIL=$((FAIL+1))   # timeouts count toward "failing" for grand total.
            printf '%s⏱%s %-44s %s%6ss%s  %sTIMEOUT%s  %s\n' \
                "$C_RED" "$C_RESET" "$name" "$C_DIM" "$elapsed" "$C_RESET" \
                "$C_RED" "$C_RESET" "$note"
            ;;
        SKIP)
            SKIP=$((SKIP+1))
            printf '%s○%s %-44s %s%6ss%s  %sSKIP%s  %s\n' \
                "$C_YELLOW" "$C_RESET" "$name" "$C_DIM" "$elapsed" "$C_RESET" \
                "$C_YELLOW" "$C_RESET" "$note"
            ;;
    esac
}

# Run one shell command with a timeout, redirect output to a log file,
# and return one of: PASS / FAIL / TIMEOUT.  Pass criterion is a line
# matching `_OK$` (case-sensitive) anywhere in stdout/stderr AND exit 0.
#
# args: name, log_file, command...
run_with_check() {
    local name="$1" log="$2"; shift 2

    local start; start=$(date +%s.%N)

    # `timeout` returns 124 on timeout, so we can distinguish.
    timeout --kill-after=10 "$TIMEOUT_SECS" "$@" >"$log" 2>&1
    local rc=$?

    local end; end=$(date +%s.%N)
    local elapsed; elapsed=$(elapsed_secs "$start" "$end")

    local _ok_line
    _ok_line=$(grep -E '_OK( |$)' "$log" | head -1 || true)

    if [[ $rc -eq 124 || $rc -eq 137 ]]; then
        record "$name" TIMEOUT "$elapsed" "killed after ${TIMEOUT_SECS}s — see $log"
        return 1
    fi

    if [[ $rc -eq 0 && -n "$_ok_line" ]]; then
        # Trim leading whitespace on the OK line for a clean note.
        _ok_line="${_ok_line#"${_ok_line%%[![:space:]]*}"}"
        record "$name" PASS "$elapsed" "$_ok_line"
        return 0
    fi

    if [[ $rc -eq 0 && -z "$_ok_line" ]]; then
        record "$name" FAIL "$elapsed" "no _OK marker — see $log"
        return 1
    fi

    # Non-zero exit.  Show the last meaningful line as the note.
    local last
    last=$(grep -v '^$' "$log" | tail -1)
    record "$name" FAIL "$elapsed" "exit $rc — ${last:0:80}"
    return 1
}

# ───────────────────────────────────────────── unit tests in tests/ ──

UNIT_TESTS=(
    test_attention_kv
    test_attention_mh_kv
    test_sampling
    test_kv_reset
    test_kv_overflow
    test_rope_edges
    test_sampling_edges
    test_codegen_quirks
)

run_unit_test() {
    local name="$1"
    local src="$ML_ROOT/tests/$name.ari"
    local bin="$ML_ROOT/tests/$name"
    local log="$LOG_DIR/aricode_runall_${name}.log"

    if [[ ! -f "$src" ]]; then
        record "$name" SKIP "0.00" "source not found: $src"
        return
    fi

    local start; start=$(date +%s.%N)

    # Recompile to make sure the binary reflects current compiler.
    if ! ( cd "$ML_ROOT/tests" && timeout "$TIMEOUT_SECS" "$ARIC" "$name.ari" -o "$name" ) > "$log" 2>&1; then
        local end; end=$(date +%s.%N)
        local elapsed; elapsed=$(elapsed_secs "$start" "$end")
        record "$name" FAIL "$elapsed" "compile failed — see $log"
        return
    fi

    # Compile succeeded.  Now run the binary.  Append run output to log
    # so postmortem has both phases.
    local run_start; run_start=$(date +%s.%N)
    timeout --kill-after=10 "$TIMEOUT_SECS" "$bin" >>"$log" 2>&1
    local rc=$?
    local end; end=$(date +%s.%N)
    local elapsed; elapsed=$(elapsed_secs "$start" "$end")

    if [[ $rc -eq 124 || $rc -eq 137 ]]; then
        record "$name" TIMEOUT "$elapsed" "killed after ${TIMEOUT_SECS}s — see $log"
        return
    fi

    local ok_line
    ok_line=$(grep -E '_OK( |$)' "$log" | head -1 || true)
    if [[ $rc -eq 0 && -n "$ok_line" ]]; then
        ok_line="${ok_line#"${ok_line%%[![:space:]]*}"}"
        record "$name" PASS "$elapsed" "$ok_line"
    elif [[ $rc -eq 0 ]]; then
        record "$name" FAIL "$elapsed" "no _OK marker — see $log"
    else
        local last
        last=$(grep -v '^$' "$log" | tail -1)
        record "$name" FAIL "$elapsed" "exit $rc — ${last:0:80}"
    fi
}

# ───────────────────────────────────────────────── example regressions ──

# (name, run_test_path, optional_skip_reason).  An empty third field
# means "run it"; non-empty means we record a SKIP with that reason.
declare -a EXAMPLES=(
    "attention_min|examples/attention_min/run_test.sh|"
    "distilbert_2block_min|examples/distilbert_2block_min/run_test.sh|"
    "embedding_2byte_min|examples/embedding_2byte_min/run_test.sh|"
    "embedding_min|examples/embedding_min/run_test.sh|"
    "encoder_full_min|examples/encoder_full_min/run_test.sh|"
    "gelu_min|examples/gelu_min/run_test.sh|"
    "layernorm_min|examples/layernorm_min/run_test.sh|"
    "mha_causal_min|examples/mha_causal_min/run_test.sh|"
    "rmsnorm_min|examples/rmsnorm_min/run_test.sh|"
    "swiglu_min|examples/swiglu_min/run_test.sh|"
    "rope_min|examples/rope_min/run_test.sh|"
    "gqa_min|examples/gqa_min/run_test.sh|"
    "rmsnorm_swiglu_decoder|examples/rmsnorm_swiglu_decoder/run_test.sh|"
    "tiny_decoder_min|examples/tiny_decoder_min/run_test.sh|"
    "tiny_decoder_packed|examples/tiny_decoder_packed/run_test.sh|"
    "tiny_decoder_2block|examples/tiny_decoder_2block/run_test.sh|"
    "tiny_llama_min|examples/tiny_llama_min/run_test.sh|"
    "tiny_decoder_packed_int8|examples/tiny_decoder_packed/run_test_int8.sh|"
    "tiny_decoder_2block_int8|examples/tiny_decoder_2block/run_test_int8.sh|"
    # Network / heavyweight downloads — guarded.  Skip dynamically below.
    "distilbert_sst2|examples/distilbert_sst2/run_test.sh|maybe_network"
    "gpt2_small|examples/gpt2_small/run_test.sh|maybe_heavy"
    "gpt2_small_int8|examples/gpt2_small/run_test_int8.sh|maybe_heavy"
)

run_example() {
    local name="$1" relpath="$2" guard="$3"
    local script="$ML_ROOT/$relpath"
    local log="$LOG_DIR/aricode_runall_${name}.log"
    local dir; dir="$(dirname "$script")"

    if [[ ! -f "$script" ]]; then
        record "$name" SKIP "0.00" "missing $relpath"
        return
    fi

    # Guard logic for network / heavy tests.
    if [[ "$guard" == "maybe_network" ]]; then
        if [[ "$SKIP_DOWNLOADS" == "1" ]]; then
            record "$name" SKIP "0.00" "SKIP_DOWNLOADS=1"
            return
        fi
        if [[ ! -d "$HOME/.cache/huggingface/hub/models--distilbert-base-uncased-finetuned-sst-2-english" ]]; then
            record "$name" SKIP "0.00" "HF distilbert cache absent — would need network"
            return
        fi
        # Skip if synth.pt missing — prepare.py would re-pull.  We
        # intentionally don't pre-build here; if it's stale, run it.
    fi
    if [[ "$guard" == "maybe_heavy" ]]; then
        # gpt2_small synth.pt is ~650 MB; only run if cached.
        if [[ "$SKIP_DOWNLOADS" == "1" ]]; then
            record "$name" SKIP "0.00" "SKIP_DOWNLOADS=1"
            return
        fi
        if [[ ! -f "$ML_ROOT/examples/gpt2_small/synth.pt" ]]; then
            record "$name" SKIP "0.00" "gpt2_small/synth.pt absent (>=650 MB) — won't redownload"
            return
        fi
    fi

    local start; start=$(date +%s.%N)
    ( cd "$dir" && timeout --kill-after=10 "$TIMEOUT_SECS" bash "$(basename "$script")" ) >"$log" 2>&1
    local rc=$?
    local end; end=$(date +%s.%N)
    local elapsed; elapsed=$(elapsed_secs "$start" "$end")

    if [[ $rc -eq 124 || $rc -eq 137 ]]; then
        record "$name" TIMEOUT "$elapsed" "killed after ${TIMEOUT_SECS}s — see $log"
        return
    fi

    local ok_line
    ok_line=$(grep -E '_OK( |$)' "$log" | tail -1 || true)
    if [[ $rc -eq 0 && -n "$ok_line" ]]; then
        ok_line="${ok_line#"${ok_line%%[![:space:]]*}"}"
        record "$name" PASS "$elapsed" "$ok_line"
    elif [[ $rc -eq 0 ]]; then
        record "$name" FAIL "$elapsed" "no _OK marker — see $log"
    else
        local last
        last=$(grep -v '^$' "$log" | tail -1)
        record "$name" FAIL "$elapsed" "exit $rc — ${last:0:80}"
    fi
}

# ────────────────────────────────────────────────────────────── main ──

print_header() {
    echo
    echo "════════════════════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════════════════════════"
}

print_header "aricode-ml: full test suite"
echo "  compiler:       $ARIC"
echo "  python:         $PYTHON"
echo "  per-test cap:   ${TIMEOUT_SECS}s"
echo "  logs:           ${LOG_DIR}/aricode_runall_*.log"
echo

print_header "unit tests (tests/*.ari)"
for t in "${UNIT_TESTS[@]}"; do
    run_unit_test "$t"
done

print_header "example regressions (examples/*/run_test*.sh)"
for entry in "${EXAMPLES[@]}"; do
    IFS='|' read -r name relpath guard <<<"$entry"
    run_example "$name" "$relpath" "$guard"
done

# ────────────────────────────────────────────────────────── summary ──

SUITE_END=$(date +%s.%N)
TOTAL=$(elapsed_secs "$SUITE_START" "$SUITE_END")

print_header "summary"

# Two-column per-test breakdown.
total=$((PASS + FAIL + SKIP))
printf "  %-44s %8s %s\n" "TEST" "TIME" "STATUS"
printf "  %s\n" "----------------------------------------------------------------"
for i in "${!NAMES[@]}"; do
    n="${NAMES[$i]}"; s="${STATUSES[$i]}"; t="${TIMES[$i]}"
    case "$s" in
        PASS)    sym="${C_GREEN}PASS${C_RESET}" ;;
        FAIL)    sym="${C_RED}FAIL${C_RESET}" ;;
        TIMEOUT) sym="${C_RED}TIMEOUT${C_RESET}" ;;
        SKIP)    sym="${C_YELLOW}SKIP${C_RESET}" ;;
    esac
    printf "  %-44s %7ss %s\n" "$n" "$t" "$sym"
done
echo

# Top-3 slowest (only PASS or FAIL — SKIP is meaningless here).
print_header "3 slowest tests"
{
    for i in "${!NAMES[@]}"; do
        if [[ "${STATUSES[$i]}" != "SKIP" ]]; then
            printf "%s %s\n" "${TIMES[$i]}" "${NAMES[$i]}"
        fi
    done
} | sort -rn | head -3 | while read -r t n; do
    printf "  %7ss  %s\n" "$t" "$n"
done

echo
print_header "totals"
printf "  passing:   %s%d%s / %d\n" "$C_GREEN" "$PASS" "$C_RESET" "$total"
printf "  failing:   %s%d%s / %d  (timeouts: %d)\n" "$C_RED" "$FAIL" "$C_RESET" "$total" "$TIMEOUT"
printf "  skipped:   %s%d%s / %d\n" "$C_YELLOW" "$SKIP" "$C_RESET" "$total"
printf "  wallclock: %ss\n" "$TOTAL"
echo

# Exit non-zero if any test failed.  Skips don't count.
if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
exit 0
