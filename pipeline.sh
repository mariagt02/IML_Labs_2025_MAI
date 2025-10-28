#!/bin/bash

# Path to your virtual environment
VENV_PATH="./.venv"

# Check if python is available
if ! command -v python &> /dev/null
then
    echo "Python not found, attempting to activate virtual environment..."
    if [ -f "$VENV_PATH/bin/activate" ]; then
        source "$VENV_PATH/bin/activate"
        echo "Virtual environment activated."
    else
        echo "Virtual environment not found at $VENV_PATH."
        exit 1
    fi
else
    echo "Python is available."
fi


python --version

echo "Introduction to Machine Learning - Project 1: k-IBL Pipeline"

# Pipeline steps:
# 1. Data Preprocessing
echo "Starting Data Preprocessing..."
echo "NOT IMPLEMENTED YET"
# 2. Model Training
echo "Starting Model Training..."
echo "NOT IMPLEMENTED YET"
# 3. Statistical Tests for Model Selection
echo "Starting Statistical Tests for Model Selection..."

python stats.py --alpha 0.1 --datasets credit-a pen-based --test_name k-IBL_hyperparameters_2_datasets
python stats.py --alpha 0.1 --datasets all --test_name k-IBL_hyperparameters_all_datasets
