#!/bin/bash
# This is an example of a more complex experiment script.
# It could run multiple seeds or configurations sequentially.

echo "Running experiment with config from experiment1.yaml"
python src/train/trainer.py --config_default configs/default.yaml --config_exp configs/experiment1.yaml

echo "Experiment finished. Check logs for results."