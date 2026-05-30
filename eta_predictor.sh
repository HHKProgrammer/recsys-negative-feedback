#!/bin/bash

echo "=== ML-20M ETA PREDICTOR ==="

# 1) Extract summary once
summary=$(python scripts/summarize_hpt_arm_d.py)

# 2) Extract trial counts (ASCII-safe)
get_done() {
    echo "$summary" | grep "$1" | grep "ML-20M" | awk '{print $5}'
}

p4=$(get_done "p4_n1")
p5=$(get_done "p5_n1")
p3=$(get_done "p3_n2")

# 3) Remaining trials
r4=$((200 - p4))
r5=$((200 - p5))
r3=$((200 - p3))

# 4) Estimate trial time from last 10 logs (if exist)
avg_time() {
    logs=$(ls logs/hpt_parallel_*movielens_20m_$1_w*.log 2>/dev/null)
    if [ -z "$logs" ]; then echo 1200; return; fi
    tail -n 200 $logs | grep trial_time | tail -n 10 | \
        sed -E 's/.*trial_time=([0-9.]+)s.*/\1/' | \
        awk '{sum+=$1} END {if (NR>0) print sum/NR; else print 1200}'
}

t4=$(avg_time "p4_n1")
t5=$(avg_time "p5_n1")
t3=$(avg_time "p3_n2")

# 5) Count active workers
workers() {
    pgrep -af "movielens_20m" | grep "$1" | wc -l
}

w4=$(workers "p4_n1")
w5=$(workers "p5_n1")
w3=$(workers "p3_n2")

# 6) ETA calculation (Python handles floats)
eta() {
    python - <<EOF
rem=$1
t=$2
w=$3
if w == 0:
    print(0)
else:
    print(rem * t / w)
EOF
}

eta4=$(eta $r4 $t4 $w4)
eta5=$(eta $r5 $t5 $w5)
eta3=$(eta $r3 $t3 $w3)

echo "p4_n1: $r4 trials left, ETA ≈ $eta4 sec"
echo "p5_n1: $r5 trials left, ETA ≈ $eta5 sec"
echo "p3_n2: $r3 trials left, ETA ≈ $eta3 sec"

# 7) Total ETA = max
total=$(python - <<EOF
print(max($eta4, $eta5, $eta3))
EOF
)

hours=$(python - <<EOF
print($total/3600)
EOF
)

echo
echo "=== TOTAL PARALLEL ETA ==="
echo "ML-20M fully done in ≈ $total seconds"
echo "≈ $hours hours"
