#!/bin/bash

# Check if CONDA_PREFIX is set and not empty
if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: CONDA_PREFIX is not set. Please set the CONDA_PREFIX environment variable."
    echo "If you are on Polaris, this can be done via 'module load conda/2023-10-04; conda activate [MARKER_ENV_NAME]'"
    exit 1
fi

# Set the default file path
output_file="local.env"

# If an argument is provided, use it as the file path
if [ ! -z "$1" ]; then
    output_file="$1"
fi

# Proceed with the rest of your script
echo "TESSDATA_PREFIX=$(find $CONDA_PREFIX -name tessdata)" > "$output_file"
echo "TORCH_DEVICE=cuda" >> "$output_file"
INFERENCE_RAM=$(nvidia-smi | grep MiB | sed "s/|//g" | awk '{print $9 / 1000}' | tail -1 | cut --delimiter="." --fields=1)
echo "INFERENCE_RAM=${INFERENCE_RAM}" >> "$output_file"

echo "Successfully created local.env file at $output_file"
