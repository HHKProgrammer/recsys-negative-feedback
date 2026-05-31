
---

## Run: 2026-05-31 21:22:38

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.3009** | 0.2600 | 0.3009 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | f=10 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | f=10 ep=20 lr=0.01 reg=0.05 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | f=10 ep=30 lr=0.001 reg=0.1 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 204 | **0.3009** | RS | f=10 ep=20 lr=0.001 reg=0.05 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 21:22:02

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.3009** | 0.2600 | 0.3009 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | f=10 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | f=10 ep=20 lr=0.01 reg=0.05 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | f=10 ep=30 lr=0.001 reg=0.1 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 204 | **0.3009** | RS | f=10 ep=20 lr=0.001 reg=0.05 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 21:21:06

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.3009** | 0.2600 | 0.3009 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | f=10 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | f=10 ep=20 lr=0.01 reg=0.05 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | f=10 ep=30 lr=0.001 reg=0.1 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 204 | **0.3009** | RS | f=10 ep=20 lr=0.001 reg=0.05 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 21:21:01

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.3009** | 0.2600 | 0.3009 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | f=10 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | f=10 ep=20 lr=0.01 reg=0.05 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | f=10 ep=30 lr=0.001 reg=0.1 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 204 | **0.3009** | RS | f=10 ep=20 lr=0.001 reg=0.05 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 19:41:44

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.3009** | 0.2600 | 0.3009 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 204 | **0.3009** | RS | f=10 ep=20 lr=0.001 reg=0.05 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 19:21:37

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.3009** | 0.2600 | 0.3009 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 204 | **0.3009** | RS | f=10 ep=20 lr=0.001 reg=0.05 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 19:18:43

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.3009** | 0.2600 | 0.3009 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 202 | **0.3009** | RS | f=10 ep=20 lr=0.001 reg=0.05 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 18:59:59

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.3009** | 0.2600 | 0.3009 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 198 | **0.3009** | RS | f=10 ep=20 lr=0.001 reg=0.05 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 18:51:33

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.3009** | 0.2600 | 0.3009 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 196 | **0.3009** | RS | f=10 ep=20 lr=0.001 reg=0.05 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 18:43:16

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.3009** | 0.2600 | 0.3009 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 196 | **0.3009** | RS | f=10 ep=20 lr=0.001 reg=0.05 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 18:15:38

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.3009** | 0.2600 | 0.3009 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 193 | **0.3009** | RS | f=10 ep=20 lr=0.001 reg=0.05 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 17:52:29

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.3009** | 0.2600 | 0.3009 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 190 | **0.3009** | RS | f=10 ep=20 lr=0.001 reg=0.05 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 17:26:04

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.3009** | 0.2600 | 0.3009 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 184 | **0.3009** | RS | f=10 ep=20 lr=0.001 reg=0.05 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 16:59:19

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.3009** | 0.2600 | 0.3009 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 178 | **0.3009** | RS | f=10 ep=20 lr=0.001 reg=0.05 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 16:54:55

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.3009** | 0.2600 | 0.3009 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 178 | **0.3009** | RS | f=10 ep=20 lr=0.001 reg=0.05 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 16:35:50

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.3009** | 0.2600 | 0.3009 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 172 | **0.3009** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 16:33:17

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.3009** | 0.2600 | 0.3009 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 172 | **0.3009** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 16:28:07

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.3009** | 0.2600 | 0.3009 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 169 | **0.3009** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 16:20:52

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.3009** | 0.2600 | 0.3009 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 166 | **0.3009** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 15:54:21

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.3009** | 0.2600 | 0.3009 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 164 | **0.3009** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 14:58:05

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.2930** | 0.2600 | 0.2930 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 147 | **0.2930** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 14:13:44

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.2863** | 0.2600 | 0.2863 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 136 | **0.2863** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 13:58:32

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.2863** | 0.2600 | 0.2863 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 135 | **0.2863** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 13:57:58

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.2863** | 0.2600 | 0.2863 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 135 | **0.2863** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 13:56:57

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.2863** | 0.2600 | 0.2863 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 134 | **0.2863** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 13:44:00

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.2863** | 0.2600 | 0.2863 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 131 | **0.2863** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 13:17:19

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.2863** | 0.2600 | 0.2863 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 124 | **0.2863** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 13:17:11

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.2863** | 0.2600 | 0.2863 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 124 | **0.2863** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 12:37:48

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.2863** | 0.2600 | 0.2863 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 115 | **0.2863** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 11:05:56

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.2863** | 0.2600 | 0.2863 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 93 | **0.2863** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 10:57:32

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2614 | **0.2863** | 0.2600 | 0.2863 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 93 | **0.2863** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 10:51:20

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | **0.2614** | 0.2592 | 0.2600 | 0.2614 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 90 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 10:30:16

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | **0.2614** | 0.2592 | 0.2600 | 0.2614 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 89 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 10:30:11

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | **0.2614** | 0.2592 | 0.2600 | 0.2614 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 89 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 10:23:25

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | **0.2614** | 0.2592 | 0.2600 | 0.2614 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 89 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 10:19:14

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | **0.2614** | 0.2592 | 0.2600 | 0.2614 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 89 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 10:19:07

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | **0.2614** | 0.2592 | 0.2600 | 0.2614 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 89 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 10:01:39

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | **0.2614** | 0.2592 | 0.2600 | 0.2614 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 87 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 09:48:13

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | **0.2614** | 0.2592 | 0.2600 | 0.2614 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 86 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 09:47:33

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | **0.2614** | 0.2592 | 0.2600 | 0.2614 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 86 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 09:45:13

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | **0.2614** | 0.2592 | 0.2600 | 0.2614 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 85 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 09:43:34

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | **0.2614** | 0.2592 | 0.2600 | 0.2614 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 85 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 09:43:14

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | **0.2614** | 0.2592 | 0.2600 | 0.2614 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 330 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 85 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 09:41:32

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | **0.2614** | 0.2592 | 0.2600 | 0.2614 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 328 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 85 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-31 09:41:04

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | **0.2614** | 0.2592 | 0.2600 | 0.2614 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 328 | **0.2614** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 85 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 22:15:20

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 94 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 22:08:32

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 92 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 22:01:05

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 89 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 21:56:16

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 87 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 21:55:39

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 87 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 21:45:54

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 87 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 21:33:36

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 85 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 21:28:45

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 84 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 21:27:08

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 83 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 21:23:41

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 82 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 21:15:36

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 78 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 21:00:06

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 74 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 20:59:57

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 74 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 20:59:42

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 74 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 20:57:51

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 72 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 20:38:09

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 66 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 20:37:56

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 66 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 20:37:16

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 66 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 20:35:27

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 66 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

<<<<<<< HEAD
## Run: 2026-05-30 17:21:02
=======
## Run: 2026-05-30 19:21:09
>>>>>>> 43b1a16 (Fix resume logic + update logs)

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
<<<<<<< HEAD
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | 0.2371 | **0.2534** | 0.2534 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2422 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
=======
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
>>>>>>> 43b1a16 (Fix resume logic + update logs)
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
<<<<<<< HEAD
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
=======
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 50 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 19:20:47

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 50 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 19:17:22

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 50 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 19:10:50

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 50 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 19:02:20

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 49 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 19:01:59

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 49 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 18:58:48

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 48 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 18:49:11

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 45 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 18:44:57

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 43 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 18:36:28

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 41 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 18:02:31

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 17:58:37

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 17:57:30

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 17:50:41

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 17:47:37

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 17:47:20

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 17:46:33

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 16:56:40

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
>>>>>>> 43b1a16 (Fix resume logic + update logs)
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 16:53:22

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 16:49:59

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 16:37:49

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 16:29:15

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 16:17:49

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 16:16:19

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 16:13:27

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 16:10:06

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 16:05:42

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 15:58:14

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 15:54:43

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 15:51:43

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 15:23:00

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 15:19:57

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 15:16:27

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:58:36

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:57:37

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:48:03

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:48:00

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:47:49

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:45:43

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:45:36

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:45:26

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:45:16

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:44:47

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:44:30

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 206 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:41:55

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 204 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:39:45

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 197 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:39:40

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2224 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 189 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 197 | **0.2224** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:36:09

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 188 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 179 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:35:02

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 188 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 174 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:34:53

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 188 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 174 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:34:22

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 188 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 173 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:33:58

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 188 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 173 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:33:02

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 188 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 168 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:31:19

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 188 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 162 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:28:21

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 188 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 161 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:26:56

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 188 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 159 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:26:09

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 185 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 159 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:25:27

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 184 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 159 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:23:37

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 184 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 157 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:22:43

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 184 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 155 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:22:09

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 184 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 155 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:20:44

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 184 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 153 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:20:30

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 184 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 153 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:20:01

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 184 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 153 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:18:51

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 184 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 153 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:18:08

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 184 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 153 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:17:22

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 184 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 153 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:17:03

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 184 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 153 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:10:43

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 184 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 153 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:09:34

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 182 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 153 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:03:24

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 150 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:02:29

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 150 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:02:20

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 149 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:02:09

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 149 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:01:57

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 149 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:01:47

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 149 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:01:02

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 149 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:00:59

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 149 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:00:54

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 149 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:00:48

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 149 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:00:45

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 149 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:00:43

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 149 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:00:41

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 149 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:00:36

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 148 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 14:00:16

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 145 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:59:46

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 145 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:59:08

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 145 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:58:56

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 145 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 34 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:56:11

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 141 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 33 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:55:55

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 141 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 33 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:55:55

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 141 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 33 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:55:54

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 141 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 33 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:55:53

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 141 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 33 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:55:51

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 141 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 33 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:55:42

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 140 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 33 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:55:05

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 137 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 33 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:53:37

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 136 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 33 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:52:39

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 135 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 33 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:52:22

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 135 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 33 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:51:04

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 135 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 33 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:49:46

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 135 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 33 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:49:22

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 134 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 33 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 31 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:47:56

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=30 ep=30 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 226 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 133 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 33 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 29 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:43:38

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 216 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 125 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 33 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 29 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:43:37

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 215 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 125 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 33 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 29 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:43:36

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 215 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 125 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 33 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 29 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:43:33

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 215 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 125 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 33 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 29 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:43:26

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 215 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 125 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 33 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 29 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:40:34

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 203 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 119 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 33 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 29 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:39:44

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 201 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 117 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 33 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 29 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:39:40

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 201 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 117 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 33 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 29 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:38:52

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 201 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 116 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 33 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 29 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:37:51

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 200 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 113 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 32 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 29 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:35:00

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 192 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 110 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 32 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 28 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:31:38

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 183 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 109 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 31 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 28 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:31:20

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 181 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 109 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 31 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 28 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:30:33

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 180 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 109 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 31 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 28 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:30:26

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 180 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 109 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 31 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 28 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:28:37

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 175 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 107 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 31 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 28 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:28:37

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 175 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 107 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 31 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 28 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:27:46

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 173 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 106 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 31 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 28 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:26:36

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 169 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 104 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 31 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 28 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:25:54

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 167 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 102 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 31 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 28 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:25:51

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 167 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 102 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 31 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 28 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:24:42

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 165 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 101 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 31 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 28 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:24:37

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 165 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 101 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 28 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:24:06

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 164 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 100 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 29 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 28 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:23:09

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 162 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 98 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 29 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 28 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:22:13

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 161 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 96 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 29 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 28 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:21:25

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 158 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 94 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 29 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 28 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:20:46

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 157 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 93 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 29 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 28 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:20:07

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 154 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 92 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 29 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 28 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:18:10

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 151 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 88 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 28 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 28 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:17:18

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 150 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 86 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 27 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 28 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:13:30

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 146 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 83 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 27 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 28 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:11:41

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 143 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 81 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 27 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 28 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:05:01

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 129 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 79 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 25 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 26 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:01:20

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 125 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 77 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 23 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 24 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 13:00:09

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 121 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 74 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 23 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 24 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 12:55:50

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 118 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 73 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 22 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 24 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 12:48:49

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 106 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 69 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 21 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 24 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 12:45:43

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 97 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 68 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 20 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 24 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 12:41:52

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 87 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 68 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 19 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 24 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 12:38:10

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 80 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 65 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 18 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 24 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 12:36:20

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 79 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 64 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 17 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 24 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 12:34:11

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 77 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 61 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 16 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 24 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 12:32:50

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 77 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 61 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 15 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 24 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 12:32:21

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 76 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 61 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 15 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 24 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 12:28:44

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 67 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 57 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 14 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 24 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 12:17:04

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 52 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 51 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 12 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 12:08:53

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 32 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 47 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 8 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 12:06:50

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 31 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 44 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 8 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 12:05:29

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 30 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 43 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 8 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 12:02:19

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 28 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 40 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 8 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 11:58:34

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 26 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 39 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 6 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 11:54:11

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 26 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 37 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 4 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 11:51:51

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 25 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 37 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 4 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 11:49:44

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 24 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 36 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 4 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 11:48:34

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 23 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 36 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 4 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 11:44:41

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 22 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 32 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 4 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 11:43:37

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 22 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 32 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 4 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 11:40:52

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 21 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 31 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 4 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 11:40:16

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 21 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 30 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 4 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 11:38:01

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 19 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 28 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 11:35:36

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 17 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 24 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 11:29:41

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 14 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 11:25:42

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 9 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 11:23:00

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 5 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 11:21:30

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 5 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 11:21:19

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 5 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 11:18:59

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 5 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 11:15:40

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 5 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 11:14:44

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | f=10 ep=30 lr=0.002 reg=0.1 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
| `p4_n1` pos≥4 / neg≤1 | Spotify | 200 | **0.2913** | RS | f=150 ep=100 lr=0.02 reg=0.1 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-1M | 200 | **0.0981** | RS | f=10 ep=20 lr=0.002 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | ML-10M | 5 | **0.1919** | RS | — |
| `p5_n1` pos≥5 / neg≤1 | ML-20M | 20 | **0.2223** | RS | f=10 ep=20 lr=0.001 reg=0.005 bias=False |
| `p5_n1` pos≥5 / neg≤1 | Spotify | 200 | **0.2228** | RS | f=75 ep=75 lr=0.02 reg=0.2 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-1M | 200 | **0.1043** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p3_n2` pos≥3 / neg≤2 | ML-10M | 30 | **0.2575** | RS | — |
| `p3_n2` pos≥3 / neg≤2 | ML-20M | 20 | **0.2592** | RS | f=10 ep=100 lr=0.002 reg=0.01 bias=False |
| `p3_n2` pos≥3 / neg≤2 | Spotify | 200 | **0.2600** | RS | f=200 ep=50 lr=0.02 reg=0.1 bias=False |




---

## Run: 2026-05-30 11:12:06

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 11:09:50

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 11:08:27

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 11:05:54

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 11:02:36

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 11:01:04

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 11:00:28

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 10:59:21

# Arm D HPT Results

## Overview

| Config            | ML-1M   | ML-10M  | ML-20M  | Spotify | Best  |
|-------------------|---------|---------|---------|---------|-------|
| `p4_n2` pos≥4 / neg≤2 | 0.1058 | 0.2165 | **0.2574** | 0.2534 | 0.2574 |
| `p4_n1` pos≥4 / neg≤1 | 0.1103 | 0.2469 | 0.2839 | **0.2913** | 0.2913 |
| `p5_n1` pos≥5 / neg≤1 | 0.0981 | 0.1919 | 0.2223 | **0.2228** | 0.2228 |
| `p3_n2` pos≥3 / neg≤2 | 0.1043 | 0.2575 | 0.2592 | **0.2600** | 0.2600 |

## Detailed Results

| Config | Dataset | Trials | Best nDCG@10 | Engine | Key Hyperparameters |
|--------|---------|--------|------------|--------|---------------------|
| `p4_n2` pos≥4 / neg≤2 | ML-1M | 200 | **0.1058** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 423 | **0.2469** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:44:17

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 43 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:43:46

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 43 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:42:19

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 43 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:42:11

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 43 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:34:10

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:32:42

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:32:06

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:32:05

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:31:59

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:30:34

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:29:35

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:29:01

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:28:32

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:27:54

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:27:51

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:27:27

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:26:45

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:26:02

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:25:29

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:22:35

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:22:16

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:22:05

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 192 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:21:41

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 191 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:20:10

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 186 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:19:38

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 184 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:18:37

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 177 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:18:15

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 174 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:18:13

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 174 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:18:12

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 174 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:18:11

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 174 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:18:10

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 174 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:18:08

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 174 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:17:58

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 173 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:17:50

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 173 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:16:56

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 171 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:16:13

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 168 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:16:12

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 168 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:16:10

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 168 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:16:08

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 167 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:15:57

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 166 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:15:46

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 166 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:15:42

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 166 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:15:32

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 166 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:15:15

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 165 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:14:26

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 160 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:13:52

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 157 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:13:27

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 150 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:12:48

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 143 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:12:27

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 140 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:12:03

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 139 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:11:52

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 138 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:11:33

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 137 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:11:19

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 137 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:11:04

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 137 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:10:34

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 135 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:10:22

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 134 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:09:55

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 133 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:09:37

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 131 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:09:16

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 130 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:08:57

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 127 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:08:37

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 127 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:08:16

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 126 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:05:25

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 107 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:04:53

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 105 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:01:47

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 94 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:01:21

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 94 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:01:19

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 94 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:00:40

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 94 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 01:00:16

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 94 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:50:04

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 82 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:49:37

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 82 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:47:37

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 78 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:47:08

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 77 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:45:33

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 62 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:45:23

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 59 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:44:58

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 59 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:42:32

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 59 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:41:56

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 59 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:40:12

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 56 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:40:11

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 56 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:40:08

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 56 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:39:19

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 56 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:38:27

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 56 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:37:02

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 56 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:36:52

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 56 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:36:11

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 56 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:36:08

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 56 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:35:15

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 56 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:34:29

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 55 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:33:30

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 52 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:32:48

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 39 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:32:15

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 39 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:31:52

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 39 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:31:51

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 39 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:31:50

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 39 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:31:49

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 39 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:31:47

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 39 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:31:46

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 39 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:31:39

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 39 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:29:31

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 39 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:28:23

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 35 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:28:21

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 35 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:28:20

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 35 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:28:19

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 34 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:28:18

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 34 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:28:17

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 34 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:28:16

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 34 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:28:15

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 33 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:28:13

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 33 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:28:07

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 32 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:27:53

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 30 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:27:44

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 30 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-30 00:24:56

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 25 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-29 23:55:30

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 6 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-29 23:55:08

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 6 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 29 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-29 23:40:12

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 6 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 20 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-29 23:36:49

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 6 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 20 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-29 23:35:52

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 6 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 20 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-29 23:35:41

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 6 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 20 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-29 23:35:38

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 6 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 20 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-29 23:34:32

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 6 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 20 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-29 23:33:18

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 4 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 20 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-29 23:28:17

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 4 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 20 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-29 23:23:32

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 2 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 20 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-29 23:21:59

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 2 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 19 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-29 23:12:57

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 1 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 10 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-29 23:12:35

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
| `p4_n2` pos≥4 / neg≤2 | ML-10M | 1 | **0.2165** | RS | — |
| `p4_n2` pos≥4 / neg≤2 | ML-20M | 200 | **0.2574** | RS | f=20 ep=20 lr=0.005 reg=0.03 bias=False |
| `p4_n2` pos≥4 / neg≤2 | Spotify | 200 | **0.2534** | RS | f=100 ep=30 lr=0.02 reg=0.2 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-1M | 200 | **0.1103** | RS | f=20 ep=20 lr=0.001 reg=0.03 bias=False |
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 10 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-29 23:11:10

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
| `p4_n1` pos≥4 / neg≤1 | ML-10M | 9 | **0.2422** | RS | — |
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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

## Run: 2026-05-29 22:46:59

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
| `p4_n1` pos≥4 / neg≤1 | ML-20M | 180 | **0.2839** | RS | f=10 ep=50 lr=0.002 reg=0.05 bias=False |
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


