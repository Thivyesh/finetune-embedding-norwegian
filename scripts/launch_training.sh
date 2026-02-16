#!/bin/bash

# Training launcher for EmbeddingGemma-300M domain adaptation
# Single GPU setup

set -e

# Configuration
SESSION_NAME="embedding-training"
PROJECT_DIR="/home/azureuser/localfiles/finetune-embedding-norwegian"
SCRIPT="scripts/train_retrieval.py"
CONFIG="configs/training_config_retrieval_embeddinggemma.yaml"
LOG_FILE="training.log"

echo "=========================================="
echo "EmbeddingGemma Domain Adaptation Training"
echo "=========================================="
echo "Project: $PROJECT_DIR"
echo "Config: $CONFIG"
echo "GPU: Single GPU"
echo ""

# Kill existing session (if any)
export MLFLOW_TRACKING_URI=file:./mlruns
echo "Cleaning up existing tmux session..."
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
sleep 1

# Start new tmux session
echo "Starting tmux session: $SESSION_NAME"
tmux new-session -d -s "$SESSION_NAME"

# Navigate to project and start training
tmux send-keys -t "$SESSION_NAME" "cd $PROJECT_DIR" Enter
sleep 1

# Run training with uv and log output
tmux send-keys -t "$SESSION_NAME" "uv run python $SCRIPT --config $CONFIG 2>&1 | tee >(tail -n 1000 > $LOG_FILE)" Enter

echo ""
echo "=========================================="
echo "✓ Training started in tmux session!"
echo "=========================================="
echo ""
echo "Commands:"
echo "  tmux attach -t $SESSION_NAME       # View live training output"
echo "  tail -f $LOG_FILE                 # Follow training logs"
echo "  tail -50 $LOG_FILE                # View last 50 lines"
echo "  grep -i error $LOG_FILE           # Check for errors"
echo "  grep -i 'eval_dev' $LOG_FILE      # View evaluation metrics"
echo "  tmux kill-session -t $SESSION_NAME # Stop training"
echo ""
echo "GPU Monitoring (in another terminal):"
echo "  watch -n 1 nvidia-smi            # Watch GPU in real-time"
echo "  nvidia-smi -l 1                   # Loop nvidia-smi every 1 sec"
echo ""
echo "TensorBoard (in another terminal):"
echo "  cd $PROJECT_DIR"
echo "  tensorboard --logdir models/embeddinggemma-300m-domain-adapted/logs"
echo ""
echo "Training directory:"
echo "  $PROJECT_DIR/models/embeddinggemma-300m-domain-adapted/"
echo ""
