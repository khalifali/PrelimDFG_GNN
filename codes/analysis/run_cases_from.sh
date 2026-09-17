#!/usr/bin/env bash

# Distribute a master run file, run LIGGGHTS cases sequentially, and install
# the VTK conversion scripts. Place this script beside run, the case_*
# directories, and the converter Python files.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
START_CASE="1"
END_CASE="999999"
DRY_RUN=false
STOP_ON_ERROR=false
CONVERT_ONLY=false

usage() {
    cat <<'EOF'
Usage:
  ./run_cases_from.sh [options]

Options:
  --start N|case_NNN   First case to run (default: 1)
  --end N|case_NNN     Last case to run, inclusive
  --root DIRECTORY     Parent directory containing case_* folders
  --dry-run            Show actions without running or copying
  --convert-only       Skip ./run; copy and execute converters only
  --stop-on-error      Stop after the first failed ./run
  -h, --help           Show this help

Examples:
  ./run_cases_from.sh --start case_003
  ./run_cases_from.sh --start 3 --end 20
  ./run_cases_from.sh --start case_003 --dry-run
  ./run_cases_from.sh --start case_001 --convert-only
EOF
}

case_number() {
    local value="$1"
    value="${value#case_}"
    value="${value%%_*}"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "Invalid case value: $1" >&2
        exit 2
    fi
    echo "$((10#$value))"
}

while (($#)); do
    case "$1" in
        --start) START_CASE="$(case_number "${2:?Missing value after --start}")"; shift 2 ;;
        --end) END_CASE="$(case_number "${2:?Missing value after --end}")"; shift 2 ;;
        --root) ROOT_DIR="$(cd -- "${2:?Missing directory after --root}" && pwd)"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --convert-only) CONVERT_ONLY=true; shift ;;
        --stop-on-error) STOP_ON_ERROR=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

find_first_file() {
    local candidate
    for candidate in "$@"; do
        if [[ -f "$ROOT_DIR/$candidate" ]]; then
            printf '%s\n' "$ROOT_DIR/$candidate"
            return 0
        fi
    done
    return 1
}

POST_SCRIPT="$(find_first_file post_dumptovtk.py post_dumtovtk.py)" || {
    echo "Could not find post_dumptovtk.py or post_dumtovtk.py in $ROOT_DIR" >&2
    exit 1
}
CONT_SCRIPT="$(find_first_file cont_dumptovtk.py contact_dumptovtk.py cont_dumtovtk.py contact_dumtovtk.py cont_dumptovt.py contact_dumptovt.py)" || {
    echo "Could not find a contact converter script in $ROOT_DIR" >&2
    exit 1
}
RUN_TEMPLATE="$ROOT_DIR/run"
if ! $CONVERT_ONLY && [[ ! -f "$RUN_TEMPLATE" ]]; then
    echo "Could not find the master run file: $RUN_TEMPLATE" >&2
    exit 1
fi

mapfile -t CASE_DIRS < <(
    find "$ROOT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'case_*' -printf '%f\n' | sort -V
)

if ((${#CASE_DIRS[@]} == 0)); then
    echo "No case_* directories found in $ROOT_DIR" >&2
    exit 1
fi

SELECTED=()
for case_name in "${CASE_DIRS[@]}"; do
    if [[ "$case_name" =~ ^case_([0-9]+)_ ]]; then
        number="$((10#${BASH_REMATCH[1]}))"
        if ((number >= START_CASE && number <= END_CASE)); then
            SELECTED+=("$case_name")
        fi
    else
        echo "Warning: ignoring directory with unrecognized name: $case_name" >&2
    fi
done

if ((${#SELECTED[@]} == 0)); then
    echo "No cases selected in range $START_CASE to $END_CASE" >&2
    exit 1
fi

LOG_FILE="$ROOT_DIR/batch_run_from_case_$(printf '%03d' "$START_CASE")_$(date +%Y%m%d_%H%M%S).log"
FAILED=()

echo "Root directory : $ROOT_DIR"
echo "All cases      : ${#CASE_DIRS[@]}"
echo "Selected cases : ${#SELECTED[@]}"
echo "First case     : ${SELECTED[0]}"
echo "Last case      : ${SELECTED[-1]}"
echo "Post converter : $(basename "$POST_SCRIPT")"
echo "Cont converter : $(basename "$CONT_SCRIPT")"
echo "Mode           : $($CONVERT_ONLY && echo 'conversion only' || echo 'simulation + conversion')"
$DRY_RUN || echo "Log file       : $LOG_FILE"

# First distribute the same master run file to all case directories. The
# --start/--end range controls execution only, not this distribution step.
if $CONVERT_ONLY; then
    :
elif $DRY_RUN; then
    echo "Would copy master run to all ${#CASE_DIRS[@]} case directories."
else
    echo "Copying master run to all ${#CASE_DIRS[@]} case directories..." | tee -a "$LOG_FILE"
    for case_name in "${CASE_DIRS[@]}"; do
        cp -f -- "$RUN_TEMPLATE" "$ROOT_DIR/$case_name/run"
        chmod +x "$ROOT_DIR/$case_name/run"
    done
    echo "Master run copied successfully to all case directories." | tee -a "$LOG_FILE"
fi

for index in "${!SELECTED[@]}"; do
    case_name="${SELECTED[$index]}"
    case_dir="$ROOT_DIR/$case_name"
    printf '\n[%d/%d] %s\n' "$((index + 1))" "${#SELECTED[@]}" "$case_name"

    if ! $CONVERT_ONLY && [[ ! -f "$case_dir/run" ]]; then
        echo "ERROR: missing $case_dir/run"
        FAILED+=("$case_name: missing run")
        $STOP_ON_ERROR && break
        continue
    fi

    if $DRY_RUN; then
        $CONVERT_ONLY || echo "  would execute: (cd '$case_dir' && ./run)"
        echo "  would copy $(basename "$POST_SCRIPT") -> results/post/"
        echo "  would copy $(basename "$CONT_SCRIPT") -> results/cont/"
        echo "  would execute: (cd results/post && python3 $(basename "$POST_SCRIPT"))"
        echo "  would execute: (cd results/cont && python3 $(basename "$CONT_SCRIPT"))"
        continue
    fi

    run_status=0
    if ! $CONVERT_ONLY; then
        echo "===== $case_name: ./run started $(date --iso-8601=seconds) =====" | tee -a "$LOG_FILE"
        (cd "$case_dir" && ./run) 2>&1 | tee -a "$LOG_FILE"
        run_status=${PIPESTATUS[0]}
    fi

    mkdir -p "$case_dir/results/post" "$case_dir/results/cont"
    cp -f -- "$POST_SCRIPT" "$case_dir/results/post/"
    cp -f -- "$CONT_SCRIPT" "$case_dir/results/cont/"
    echo "Copied converter scripts into results/post and results/cont" | tee -a "$LOG_FILE"

    echo "Running particle VTK conversion..." | tee -a "$LOG_FILE"
    (cd "$case_dir/results/post" && python3 "$(basename "$POST_SCRIPT")") 2>&1 | tee -a "$LOG_FILE"
    post_status=${PIPESTATUS[0]}

    echo "Running contact VTK conversion..." | tee -a "$LOG_FILE"
    (cd "$case_dir/results/cont" && python3 "$(basename "$CONT_SCRIPT")") 2>&1 | tee -a "$LOG_FILE"
    cont_status=${PIPESTATUS[0]}

    if ((run_status != 0 || post_status != 0 || cont_status != 0)); then
        echo "ERROR: $case_name status: run=$run_status, post=$post_status, cont=$cont_status" | tee -a "$LOG_FILE"
        FAILED+=("$case_name: run=$run_status post=$post_status cont=$cont_status")
        $STOP_ON_ERROR && break
    else
        echo "===== $case_name completed $(date --iso-8601=seconds) =====" | tee -a "$LOG_FILE"
    fi
done

if $DRY_RUN; then
    echo "Dry run completed; no simulations were run and no files were copied."
    exit 0
fi

printf '\nBatch finished. Successful: %d, failed: %d\n' \
    "$((${#SELECTED[@]} - ${#FAILED[@]}))" "${#FAILED[@]}" | tee -a "$LOG_FILE"
if ((${#FAILED[@]})); then
    printf '  %s\n' "${FAILED[@]}" | tee -a "$LOG_FILE"
    exit 1
fi
