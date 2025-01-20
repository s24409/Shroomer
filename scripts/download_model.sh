#!/bin/bash

# Check if HF_MODEL_ID is provided
if [ -z "$1" ]; then
    echo "Usage: ./download_model.sh <HF_MODEL_ID>"
    exit 1
fi

HF_MODEL_ID=$1
OUTPUT_DIR="app/models/fungi-classifier"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Download model using transformers-cli
python -c "from transformers import AutoModelForImageClassification, AutoFeatureExtractor; \
           model = AutoModelForImageClassification.from_pretrained('$HF_MODEL_ID'); \
           feature_extractor = AutoFeatureExtractor.from_pretrained('$HF_MODEL_ID'); \
           model.save_pretrained('$OUTPUT_DIR'); \
           feature_extractor.save_pretrained('$OUTPUT_DIR')"

echo "Model downloaded successfully to $OUTPUT_DIR"
