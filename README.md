# Recsys Negative Feedback

# Recsys Negative Feedback

Goal:
Build a recommender system that learns not just what users like, but also what they don’t want.
I use the MovieLens dataset and treat low ratings as negative feedback.

---

##  1. Setup (Ubuntu / WSL)

Make sure you’re in Ubuntu (WSL), **not PowerShell**.(just for me)

```bash
cd "/mnt/c/Users/Helin/OneDrive/Dokumente/BachelorThesis/code/srcCode/recsys-negative-feedback"


installed packages:
sudo apt update
sudo apt install -y python3 python3-venv python3-pip

##create and activate virtual enviorment:
python3 -m venv .venv
source .venv/bin/activate

python libs:
pip install --upgrade pip
pip install pandas numpy pyarrow jupyter


python3 --version
pip --version

python3 --version
Python 3.10.16
helin@DESKTOP-AE9GKSP:/mnt/c/Users/Helin/OneDrive/Dokumente/BachelorThesis/code/srcCode/recsys-negative-feedback$ pip3 --version
pip 25.3 from /home/helin/.local/lib/python3.10/site-packages/pip (python 3.10) 


venv active start repo:
jupyter notebook --no-browser --ip=127.0.0.1

URL it prints (starts with http://127.0.0.1:8888/...) and open it in your Windows browser


3. What the notebook 02_movielensEdaAndSplit does

Load raw MovieLens data (u.data)

Columns: user, item, rating, timestamp

Each row = one event (user-item interaction)

Filter sparse users/items
minUserEvents, minItemEvents = 5, 5

Keep only users and items with ≥5 ratings.

Map IDs to 0..N-1
Creates dense integer IDs for ML models.

Sort by time
So we know what “last” means for each user.

Split train/test
Each user’s last event = test
All previous = train
→ simulates “can we predict the next thing?”

Save processed data

data/processed/movielens/train.parquet
data/processed/movielens/test.parquet


