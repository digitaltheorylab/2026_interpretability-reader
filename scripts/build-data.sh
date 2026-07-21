#!/usr/bin/env bash
#
# Build all data artifacts for the interpretability reader.
#
# Usage:
#   ./scripts/build-data.sh [OPTIONS]
#
# Options:
#   --force          Rebuild all outputs even if they exist
#   --skip-stories   Skip story generation (step 1); requires existing stories file
#   --model MODEL    Override the default model checkpoint
#
# Environment:
#   MODEL            Alternative way to set model (default: allenai/Olmo-3-7B-Instruct-SFT)
#
# Examples:
#   ./scripts/build-data.sh --force
#   ./scripts/build-data.sh --skip-stories
#   ./scripts/build-data.sh --model "meta-llama/Llama-3-8B-Instruct"
#
set -euo pipefail

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

MODEL="${MODEL:-allenai/Olmo-3-7B-Instruct-SFT}"
SEED=5167
DATA_DIR="data"
SRC_DIR="src"

# ------------------------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------------------------

FORCE=false
SKIP_STORIES=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE=true
            shift
            ;;
        --skip-stories)
            SKIP_STORIES=true
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            head -20 "$0" | tail -n +2 | sed 's/^#//' | sed 's/^ //'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

run_step() {
    local output="$1"
    shift
    if [[ "$FORCE" == true ]] || [[ ! -f "$output" ]]; then
        log "==> Building $output"
        "$@"
    else
        log "==> Skipping $output (exists, use --force to rebuild)"
    fi
}

# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------

mkdir -p "$DATA_DIR"
mkdir -p "$(dirname "$0")"

log "Model: $MODEL"
log "Data directory: $DATA_DIR"

# ------------------------------------------------------------------------------
# Step 1: Generate genre-conditioned stories
# ------------------------------------------------------------------------------

STORIES="$DATA_DIR/w2-d1_genre-stories.parquet"

if [[ "$SKIP_STORIES" == true ]]; then
    if [[ ! -f "$STORIES" ]]; then
        log "ERROR: --skip-stories specified but $STORIES does not exist"
        exit 1
    fi
    log "==> Skipping story generation (--skip-stories)"
else
    run_step "$STORIES" \
        pixi run python "$SRC_DIR/generate_stories.py" \
            "$STORIES" \
            -m "$MODEL" \
            -n 1000 \
            -b 32 \
            -s "$SEED"
fi

# ------------------------------------------------------------------------------
# Step 2: MCQA classification with logit lens + gradient attribution
# ------------------------------------------------------------------------------

MCQA_ATTRS="$DATA_DIR/w2-d2_genre-attributions.parquet"

run_step "$MCQA_ATTRS" \
    pixi run python "$SRC_DIR/classify_mcqa.py" \
        "$STORIES" \
        "$MCQA_ATTRS" \
        -m "$MODEL" \
        -n 100 \
        -s "$SEED"

# ------------------------------------------------------------------------------
# Step 3: Extract per-layer hidden states from MCQA outputs
# ------------------------------------------------------------------------------

HIDDEN_STATES="$DATA_DIR/w2-d3_genre-hidden-states.npz"

run_step "$HIDDEN_STATES" \
    pixi run python "$SRC_DIR/extract_hiddens.py" \
        "$MCQA_ATTRS" \
        "$HIDDEN_STATES" \
        -m "$MODEL"

# ------------------------------------------------------------------------------
# Step 4: Train linear probe / classifier head
# ------------------------------------------------------------------------------

CLASSIFIER="$DATA_DIR/w2-d3_linear-classifier.npz"

run_step "$CLASSIFIER" \
    pixi run python "$SRC_DIR/train_probe.py" \
        "$STORIES" \
        "$CLASSIFIER" \
        -m "$MODEL" \
        -n 100 \
        --save-states \
        -s "$SEED"

# ------------------------------------------------------------------------------
# Step 5: Token attributions using trained classifier head
# ------------------------------------------------------------------------------

CLF_ATTRS="$DATA_DIR/w2-d3_token-attributions.parquet"

run_step "$CLF_ATTRS" \
    pixi run python "$SRC_DIR/attribute_tokens.py" \
        "$STORIES" \
        "$CLASSIFIER" \
        "$CLF_ATTRS" \
        -m "$MODEL" \
        -n 100 \
        -s "$SEED"

# ------------------------------------------------------------------------------
# Step 6: Genre lift from MCQA attributions
# ------------------------------------------------------------------------------

MCQA_LIFT="$DATA_DIR/w2-d3_mcqa-genre-lift.parquet"

run_step "$MCQA_LIFT" \
    pixi run python "$SRC_DIR/calculate_lift.py" \
        "$MCQA_ATTRS" \
        "$MCQA_LIFT" \
        -m 10 \
        -n 250

# ------------------------------------------------------------------------------
# Step 7: Genre lift from classifier attributions
# ------------------------------------------------------------------------------

CLF_LIFT="$DATA_DIR/w2-d3_classifier-genre-lift.parquet"

run_step "$CLF_LIFT" \
    pixi run python "$SRC_DIR/calculate_lift.py" \
        "$CLF_ATTRS" \
        "$CLF_LIFT" \
        -m 10 \
        -n 250

# ------------------------------------------------------------------------------
# Done
# ------------------------------------------------------------------------------

log "==> Done"
