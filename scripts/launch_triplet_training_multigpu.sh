#!/bin/bash

# Training launcher for EmbeddingGemma-300M triplet fine-tuning
# Multi-GPU setup (2x H100)

set -e

# Configuration
SESSION_NAME="triplet-training"
PROJECT_DIR="/home/azureuser/localfiles/finetune-embedding-norwegian"
SCRIPT="scripts/train_triplet.py"
CONFIG="configs/training_config_triplet_embeddinggemma.yaml"
LOG_FILE="triplet_training.log"
NUM_GPUS=2

echo "=========================================="
echo "EmbeddingGemma Triplet Fine-tuning (Multi-GPU)"
echo "=========================================="
echo "Project: $PROJECT_DIR"
echo "Config: $CONFIG"
echo "GPUs: $NUM_GPUS"
echo ""

# Set MLflow tracking
export MLFLOW_TRACKING_URI=file:./mlruns

# Kill existing session (if any)
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
sleep 1

# Start new tmux session
echo "Starting tmux session: $SESSION_NAME"
tmux new-session -d -s "$SESSION_NAME"
tmux send-keys -t "$SESSION_NAME" "cd $PROJECT_DIR" Enter
tmux send-keys -t "$SESSION_NAME" "export MLFLOW_TRACKING_URI=file:./mlruns" Enter

# Launch with torchrun for distributed training
tmux send-keys -t "$SESSION_NAME" "torchrun --nproc_per_node=$NUM_GPUS $SCRIPT --config $CONFIG 2>&1 | tee $LOG_FILE" Enter

echo ""
echo "✓ Training started on $NUM_GPUS GPUs!"
echo ""
echo "Commands:"
echo "  tmux attach -t $SESSION_NAME    # View live output"
echo "  tail -f $LOG_FILE              # Follow logs"
echo "  tmux kill-session -t $SESSION_NAME  # Stop training"
echo ""
echo "GPU Monitoring:"
echo "  watch -n 1 nvidia-smi"
echo ""
