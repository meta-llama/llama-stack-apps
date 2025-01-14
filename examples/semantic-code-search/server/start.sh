#!/bin/bash

# Change to the script's directory
cd "$(dirname "$0")"

# Function to check if conda environment exists
check_conda_env() {
    conda env list | grep -q "semantic-code-search"
}

# Check if conda command exists
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Create or activate conda environment
if ! check_conda_env; then
    echo "Creating conda environment 'semantic-code-search'..."
    conda create -y -n semantic-code-search python=3.10
fi

# Activate conda environment
echo "Activating conda environment 'semantic-code-search'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate semantic-code-search

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Run the application
echo "Starting the application..."
python app.py
