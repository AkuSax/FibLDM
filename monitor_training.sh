#!/bin/bash

# Training monitoring script

# Find the most recent training log
LATEST_LOG=$(ls -t logs/training_*.log 2>/dev/null | head -n1)
LATEST_PID=$(ls -t logs/training_*.pid 2>/dev/null | head -n1)

if [ -z "$LATEST_LOG" ]; then
    echo "No training logs found. Make sure training is running."
    exit 1
fi

echo "=== Training Monitor ==="
echo "Latest log: $LATEST_LOG"

if [ -n "$LATEST_PID" ]; then
    PID=$(cat "$LATEST_PID")
    echo "Process ID: $PID"
    
    # Check if process is still running
    if ps -p $PID > /dev/null 2>&1; then
        echo "Status: RUNNING"
        echo ""
        echo "=== Recent Log Output ==="
        tail -n 20 "$LATEST_LOG"
        echo ""
        echo "=== GPU Usage ==="
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
        echo ""
        echo "=== Commands ==="
        echo "Monitor live: tail -f $LATEST_LOG"
        echo "Stop training: kill $PID"
        echo "Check process: ps aux | grep $PID"
    else
        echo "Status: STOPPED"
        echo ""
        echo "=== Final Log Output ==="
        tail -n 50 "$LATEST_LOG"
    fi
else
    echo "No PID file found."
    echo ""
    echo "=== Recent Log Output ==="
    tail -n 20 "$LATEST_LOG"
fi 