#!/usr/bin/env bash

set -euo pipefail

#directory in which this bash script is located
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

LOCAL_FILLING_FACTOR_PYTHON_SCRIPT="${SCRIPT_DIR}/LocalFillingFactor/urbassek3d.py"
SYS_SCRIPT="${SCRIPT_DIR}/liggghtsWriter.py"

CONFIG_DIR="${SCRIPT_DIR}/batch_lf/generated_configs_localFillingFactor"
OUTPUT_DIR="${SCRIPT_DIR}/batch_lf/results_csv_localFillingFactor"
SYS_OUTPUT_DIR="${SCRIPT_DIR}/batch_lf/results_sys_localFillingFactor"

#different particle counts to generate
PARTICLE_COUNTS=(10 50 500)

#different detector radius multipliers for the local filling factor method
R_SIGMA_MULT_VALUES=(10.0)

PARTICLE_RADIUS="4.850000000000E-01"
N_FAIL="200"
BASE_SEED="1029"

echo "––––––– Local Filling Factor :¬} –––––––"

CREATE_SYS=False

#cleanup generated files and exit
if [[ "${1:-}" == "--clean" ]]; then
    echo "Cleaning generated Local Filling Factor files..."

    rm -rf "$CONFIG_DIR"
    rm -rf "$OUTPUT_DIR"
    rm -rf "$SYS_OUTPUT_DIR"

    echo "Cleanup complete."
    exit 0
fi

#create LIGGGHTS input files from generated CSV files and exit
if [[ "${1:-}" == "--sys" ]]; then
    CREATE_SYS=true

    if [[ ! -f "$SYS_SCRIPT" ]]; then
        echo "Error: Could not find Python script: $SYS_SCRIPT" >&2
        exit 1
    fi
fi

if [[ ! -f "$LOCAL_FILLING_FACTOR_PYTHON_SCRIPT" ]]; then
    echo "Error: Could not find Python script: $LOCAL_FILLING_FACTOR_PYTHON_SCRIPT" >&2
    exit 1
fi

mkdir -p "$CONFIG_DIR"
mkdir -p "$OUTPUT_DIR"

#create LIGGGHTS input files from generated CSV files and exit
if [[ "$CREATE_SYS" == true ]]; then
    echo "Creating LIGGGHTS input files..."

    mkdir -p "$SYS_OUTPUT_DIR"

    csv_files=("$OUTPUT_DIR"/*.csv)

    if [[ ! -e "${csv_files[0]}" ]]; then
        echo "Error: Could not find CSV files in: $OUTPUT_DIR" >&2
        exit 1
    fi

    for csv_file in "${csv_files[@]}"; do
        csv_name="$(basename "$csv_file" .csv)"
        sys_file="${SYS_OUTPUT_DIR}/${csv_name}.sys"
        generated_sys_file="${csv_file%.csv}.sys"

        echo "========================================"
        echo "Creating SYS file from: ${csv_file}"
        echo "Output: ${sys_file}"
        echo "========================================"

        python3 "$SYS_SCRIPT" "$csv_file"

        if [[ -f "$generated_sys_file" ]]; then
            mv "$generated_sys_file" "$sys_file"
        else
            echo "Warning, did not find SYS file -> Conversion Error?? :¬} : $generated_sys_file" >&2
        fi
    done

    echo "Done."
    echo "Wrote LIGGGHTS input files to: $SYS_OUTPUT_DIR"
    exit 0
fi

for r_sigma_mult in "${R_SIGMA_MULT_VALUES[@]}"; do
    for n in "${PARTICLE_COUNTS[@]}"; do

        config_file="${CONFIG_DIR}/config_delta_${r_sigma_mult}_N_${n}.xml"
        seed=$((BASE_SEED))

        cat > "$config_file" <<EOF
<config>
    <!-- Number of particles in the aggregate -->
    <N_target>${n}</N_target>

    <!-- Maximum number of attachment attempts -->
    <max_try>${N_FAIL}</max_try>

    <!-- Radius of each primary particle -->
    <particle_radius>${PARTICLE_RADIUS}</particle_radius>

    <!-- Detection radius multiplier -->
    <r_sigma_mult>${r_sigma_mult}</r_sigma_mult>

    <!-- Random seed for reproducible aggregate generation -->
    <random_seed>${seed}</random_seed>
</config>
EOF

        echo "========================================"
        echo "Starting r_sigma=${r_sigma_mult} * r_particle, N=${n}"
        echo "Config: ${config_file}"
        echo "========================================"

        (
            cd "$SCRIPT_DIR"

            python3 -u "$LOCAL_FILLING_FACTOR_PYTHON_SCRIPT" "$config_file"
        ) | while IFS= read -r line; do

            if [[ "$line" =~ ^Attached\ particle\ \(\ ([0-9]+)\ /\ ([0-9]+)\ \)$ ]]; then
                current="${BASH_REMATCH[1]}"
                total="${BASH_REMATCH[2]}"

                percent=$((100 * current / total))
                bar_width=40
                filled=$((current * bar_width / total))
                empty=$((bar_width - filled))

                printf -v filled_bar "%*s" "$filled" ""
                printf -v empty_bar "%*s" "$empty" ""

                filled_bar="${filled_bar// /#}"
                empty_bar="${empty_bar// /-}"

                printf "\r[%s%s] %3d%% (%d/%d)" \
                    "$filled_bar" "$empty_bar" "$percent" "$current" "$total"

            else
                printf "\n%s\n" "$line"
            fi
        done

        echo

        generated_csv_file="aggr_delta_${r_sigma_mult}_N_${n}.csv"

        if [[ -f "$generated_csv_file" ]]; then
            mv "$generated_csv_file" "${OUTPUT_DIR}/${generated_csv_file}"
        else
            echo "Warning, did not find CSV file -> Generation Error?? :¬} : $generated_csv_file" >&2
        fi
    done
done

echo "Done."
echo "Wrote results to: $OUTPUT_DIR"

