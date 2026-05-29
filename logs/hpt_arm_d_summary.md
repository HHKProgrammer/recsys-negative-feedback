
---

## Run: 2026-05-29 21:39:10

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2422 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 50 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 50 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 177 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 1 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-29 21:35:54

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2422 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 50 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 50 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 177 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 1 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-29 21:31:22

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2422 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 50 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 50 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 172 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 1 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-29 21:26:34

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2422 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 50 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 50 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 169 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 1 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-29 21:26:27

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2422 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 50 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 50 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 169 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 1 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-29 21:21:28

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2422 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 50 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 50 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 169 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 1 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-29 21:02:43

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2422 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 50 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 50 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 159 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 1 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-29 21:02:35

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2422 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 50 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 50 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 159 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 1 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-29 21:01:46

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2422 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 50 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 50 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 159 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 1 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-29 19:31:36

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2422 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 50 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 50 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 89 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 1 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-29 19:31:22

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2422 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 50 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 50 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 89 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 1 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-29 19:27:43

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2422 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 50 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 50 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 89 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 1 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-29 18:39:37

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2422 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 50 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 50 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 81 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 1 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-29 18:39:16

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2422 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 50 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 50 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 80 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 1 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-29 18:03:50

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2422 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 50 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 50 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 70 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 1 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-29 18:00:00

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2422 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 50 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 50 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 60 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 1 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-29 17:59:29

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2422 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 50 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 50 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 60 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 1 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-29 17:54:04

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2422 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 50 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 50 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 48 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 1 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-29 17:53:10

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2422 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 50 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 50 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 48 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 1 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-29 17:52:48

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2422 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 50 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 50 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 48 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 1 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-29 17:51:01

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2422 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 50 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 50 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 48 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 1 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-29 17:50:55

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2422 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 50 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 50 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 48 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 1 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-29 17:45:38

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2422 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 50 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 50 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 48 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 1 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-29 17:21:01

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | 0.2371 | **0.2534** | 0.2534 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2422 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 50 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 20 | **0.2371** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 50 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 30 | **0.2839** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 170 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 80 | **0.2223** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |


