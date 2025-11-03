# Recsys Negative Feedback

# Recsys Negative Feedback

Goal:
Build a recommender system that learns not just what users like, but also what they don’t want.
I use the MovieLens dataset and treat low ratings as negative feedback.

---

##  1. Setup (Ubuntu / WSL)

This project runs in Ubuntu (WSL) on Windows 10/11.

Requirements

WSL2 with Ubuntu 20.04 or newer

Visual Studio Code

Remote - WSL extension

Python extension

Jupyter extension

Miniconda installed inside WSL

2. Install Miniconda

Run the following commands in your Ubuntu terminal:

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh

# Initialize Conda
~/miniconda3/bin/conda init bash
exec bash


Accept Conda terms (required for new versions):

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

3. Create Environment
conda create -n recsys-conda python=3.8 -y
conda activate recsys-conda

4. Install Required Packages

Use Conda (recommended):

conda install -c conda-forge pandas numpy pyarrow jupyter scikit-learn scikit-surprise scipy matplotlib seaborn ipykernel -y
#conda install -c conda-forge matplotlib -y
#conda install -c conda-forge matplotlib scikit-learn -y
# To recreate environment from file:
conda env create -f environment.yml

Or use pip:

pip install --upgrade pip
pip install pandas numpy pyarrow jupyter scikit-learn scikit-surprise scipy matplotlib seaborn

5. Register Kernel for VS Code

Make the environment visible in Jupyter and VS Code:

python -m ipykernel install --user --name recsys-conda --display-name "Python 3.8 (recsys)"


Verify:

cat /home/helin/.local/share/jupyter/kernels/recsys-conda/kernel.json


It should contain a line similar to:

"argv": ["/home/helin/miniconda3/envs/recsys-conda/bin/python", ...]
"display_name": "Python 3.8 (recsys)"

6. Open the Project in VS Code (WSL Mode)
conda activate recsys-conda
cd /mnt/c/Users/Helin/OneDrive/Dokumente/BachelorThesis/code/srcCode/recsys-negative-feedback
code .


In VS Code:

Make sure the bottom-left corner shows WSL: Ubuntu.

When you open a notebook, click Select Kernel (top-right).

Choose Python 3.8 (recsys).

7. Verify Installation

Run this in a new notebook cell:

import sys
from surprise import SVD
print("Python:", sys.version)
print("Executable:", sys.executable)
print("Surprise OK:", SVD)


Expected output:

Python: 3.8.x
Executable: /home/helin/miniconda3/envs/recsys-conda/bin/python
Surprise OK: <class 'surprise.prediction_algorithms.matrix_factorization.SVD'>

8. Daily Workflow

Every time you work:

conda activate recsys-conda
cd /mnt/c/Users/Helin/OneDrive/Dokumente/BachelorThesis/code/srcCode/recsys-negative-feedback
code .

9. Notebooks Overview

02_movielensEdaAndSplit.ipynb

Loads raw MovieLens data (u.data)

Columns: user, item, rating, timestamp

Filters users and items with at least 5 ratings

Maps IDs to integers

Sorts by time

Splits into train/test (last event per user = test)

Saves:

data/processed/movielens/train.parquet

data/processed/movielens/test.parquet

10. Running Models

Run baseline collaborative filtering:

python -m src.eval.evaluateAlsMovielens


Run thesis comparison (baseline vs. negative model):

python -m src.eval.generate_thesis_results

11. Output Files

Generated under reports/:

reports/
  figures/
    learned_factors.png
    learning_curves.png
    metric_comparison.png
  tables/
    baseline_metrics.csv
    negative_feedback_metrics.csv
    statistical_tests.csv