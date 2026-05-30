#!/bin/bash

check_and_start() {
    cfg=$1
    dataset=$2
    target=200

    done=$(python scripts/summarize_hpt_arm_d.py | grep "$cfg" | grep "$dataset" | awk '{print $5}')
    if [ "$done" -ge "$target" ]; then
        echo " $cfg $dataset fertig → Worker stoppen"
        pkill -f "$cfg"
        return 0
    fi
    return 1
}

while true; do
    echo "[Scheduler] Checking…"

    # Reihenfolge: p4_n1 → p5_n1 → p3_n2
    check_and_start "p4_n1" "ML-20M" && {
        echo "→ Starte p5_n1 Worker"
        nohup ./run_hpt_parallel.sh movielens_20m p5_n1 4 200 &
    }

    check_and_start "p5_n1" "ML-20M" && {
        echo "→ Starte p3_n2 Worker"
        nohup ./run_hpt_parallel.sh movielens_20m p3_n2 4 200 &
    }

    sleep 300
done

