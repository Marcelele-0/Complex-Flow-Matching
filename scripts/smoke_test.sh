#!/bin/bash
set -e

echo "Starting Hydra Matrix Smoke Test..."

for ds in skm_tea fastmri_local; do
    for mod in c_unet; do
        echo "Testing dataset: $ds | model: $mod"
        uv run src/cfm/train.py dataset=$ds model=$mod training.epochs=1
    done
done

echo "Matrix Smoke test completed successfully!"
