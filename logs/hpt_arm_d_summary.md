
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


