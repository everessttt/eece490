#!/bin/bash

ROOT="/home/everest/EECE490"
MODEL_ROOT="$ROOT/quantized_models/yolo11/det/yolo11n" # a directory or .pt file
#MODEL_ROOT="$ROOT/models/yolo11/det"
DATASET="coco128"
RUNS_ROOT="$ROOT/runs"
PARALLEL=$(nproc)

if [ -f "$MODEL_ROOT" ]; then
    models=("$MODEL_ROOT")
elif [ -d "$MODEL_ROOT" ]; then
    mapfile -t models < <(find "$MODEL_ROOT" -name "*.pt" | sort)
else
    echo "Error: $MODEL_ROOT is invalid"
    exit 1
fi

echo "Validating ${#models[@]} models in parallel"

run_val() {
    model="$1"
    ROOT="$2"
    DATASET="$3"
    RUNS_ROOT="$4"

    rel_path="${model#$ROOT/}"
    rel_dir=$(dirname "$rel_path")
    model_name=$(basename "$model" .pt)

    if [[ "$model_name" == *"-seg"* ]]; then
        task="segment"
    elif [[ "$model_name" == *"-pose"* ]]; then
        task="pose"
    else
        task="detect"
    fi

    out_dir="$RUNS_ROOT/$rel_dir/$model_name/$DATASET"
    if [ -f "$out_dir/val.log" ]; then
        echo "Skipping: $rel_path"
        return
    fi
    mkdir -p "$out_dir"

    yolo val \
        model="$model" \
        data=$DATASET.yaml \
        task=$task \
        name="$DATASET" \
        project="$RUNS_ROOT/$rel_dir/$model_name" \
        half=False \
        save_json=True \
        save_conf=True \
        save_txt=True \
        plots=True \
        verbose=False \
        exist_ok=True \
        2>&1 | tee "$out_dir/val.log"
}

export -f run_val

printf '%s\n' "${models[@]}" | xargs -P $PARALLEL -I {} bash -c \
    'run_val "$@"' _ {} "$ROOT" "$DATASET" "$RUNS_ROOT"