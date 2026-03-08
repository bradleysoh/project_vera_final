#!/bin/bash

echo "🚀 Starting Project VERA..."

# Start Ollama if not running
if ! pgrep -x "ollama" > /dev/null
then
    echo "Starting Ollama..."
    ollama serve &
    sleep 3
else
    echo "Ollama already running"
fi

# Activate conda environment
echo "Activating environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vera || conda activate base

# Start backend
echo "Starting VERA backend..."
python app.py &

# Wait a moment
sleep 3

# Launch Streamlit UI
echo "Launching UI..."
streamlit run streamlit_app.py

echo "VERA started successfully!"
