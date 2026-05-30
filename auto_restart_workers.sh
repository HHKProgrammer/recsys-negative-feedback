#!/bin/bash

LOG_DIR="logs"
TIMEOUT=1200   # 20 Minuten
PATTERN="hpt_parallel_.*w.*\.log"

echo "Checking worker health..."

for f in $LOG_DIR/$PATTERN; do
    last_line=$(tail -n 1 "$f")
    timestamp=$(stat -c %Y "$f")
    now=$(date +%s)
    age=$((now - timestamp))

    # Worker hängt → kein Update seit 20 Minuten
    if [ $age -gt $TIMEOUT ]; then
        echo ""!!!!  Worker hängt: $f (last update $age seconds ago)"
        
        # PID extrahieren
        pid=$(grep -oP 'PID \K[0-9]+' "$f")
        if [ ! -z "$pid" ]; then
            echo "→ Killing PID $pid"
            kill -9 $pid
        fi

        echo "→ Restarting worker"
        bash run_hpt_parallel.sh <dataset> <config> <workers> <trials> &
    fi
done
