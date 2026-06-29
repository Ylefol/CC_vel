#!/bin/bash

SNAKEFILE="/home/yohanlefol/A_Projects/CC_vel/Snakefile"
DATA_DIR="/media/yohanlefol/Expansion/CC_vel"

# Configure these
CELL_LINES=(
    "HaCat-Control"
    "A549-Control"
    "A549-CRISPRi-AGO2-EXP"
    "A549-CRISPRi-AGO2-KO"
    "A549-CRISPRi-Dicer-EXP"
    "A549-CRISPRi-Dicer-KO"
    "HaCat-CRISPRi-AGO2-EXP"
    "HaCat-CRISPRi-AGO2-KO"
    "HaCat-CRISPRi-Dicer-EXP"
    "HaCat-CRISPRi-Dicer-KO"
)
REPLICATES=("A" "B")
MERGED="A_B"

BASE_CMD="snakemake --snakefile $SNAKEFILE --directory $DATA_DIR"

set -e

# Build target lists per rule
rule1_targets=()
rule2_targets=()
rule3_targets=()
rule4_targets=()
rule5_targets=()

for cl in "${CELL_LINES[@]}"; do
    for rep in "${REPLICATES[@]}"; do
        rule1_targets+=("data_files/phase_reassigned/CC_${cl}_${rep}.loom")
        rule2_targets+=("data_files/boundary_data/${cl}_${rep}_boundaries.csv")
        rule3_targets+=("data_files/confidence_intervals/${cl}/${rep}/Velocity_iterations")
    done
    rule4_targets+=("data_files/confidence_intervals/${cl}/merged_results/${MERGED}")
    rule5_targets+=("data_files/data_results/rank/${cl}/${MERGED}_ranked_genes.csv")
done

echo "=== Rule 1: QC and phase reassignment ==="
$BASE_CMD "${rule1_targets[@]}" --cores 8

# Qt platform plugin fix for Wayland/Linux environments
export QT_QPA_PLATFORM=xcb
export QT_PLUGIN_PATH=/home/yohanlefol/miniconda3/envs/CC_vel/plugins/


echo "=== Rule 2: Velocyto on cell cycle genes ==="
$BASE_CMD "${rule2_targets[@]}" --cores 8

echo "=== Rule 3: Velocity iterations ==="
$BASE_CMD "${rule3_targets[@]}" --cores 8

echo "=== Rule 4: Merge replicates and calculate CIs ==="
$BASE_CMD "${rule4_targets[@]}" --cores 1

echo "=== Rule 5: Gene ranking and delay ==="
$BASE_CMD "${rule5_targets[@]}" --cores 1

echo "=== Pipeline complete ==="
