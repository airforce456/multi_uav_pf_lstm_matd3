#!/bin/bash
# A simple script to start the training process.

echo "Starting PF-LSTM-MATD3 Training..."

# Run the trainer as a module from the project root.
# This ensures that all imports starting with 'src.' work correctly.
python -m src.train.trainer --config_default configs/default.yaml --config_exp configs/experiment1.yaml