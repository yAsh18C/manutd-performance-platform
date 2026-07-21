# Manchester United Performance Intelligence Platform

An advanced football analytics system built on real StatsBomb event data from the Premier League 2015/16 season. This project covers five analytical modules that mirror the actual work done inside professional football club data teams — not basic stats, but the kind of questions that actually influence decisions.

Built by Yash Chabukswar, MSc Data Science, University of Manchester (2025/26).

---

## What This Project Does

Most football analytics projects stop at goals and assists. This one goes further. Each module answers a question that a real Head of Analytics would ask, using methods that clubs actually use in practice.

**Module 1 — Expected Goals (xG) Model**
A logistic regression xG model built entirely from scratch using StatsBomb freeze frame data. Every shot has features extracted including distance to goal, shot angle, body part used, whether the player was under pressure, the goalkeeper's position off the line, how many defenders were in the shooting cone, and the nearest defender's distance. The model reaches an AUC of 0.794 — competitive with published academic xG models which typically sit between 0.76 and 0.82.

Key finding: Anthony Martial scored 5 more goals than his xG predicted across the season. That gap is statistically significant enough to suggest genuine clinical ability rather than luck, particularly when combined with consistent shot location data.

**Module 2 — Player Influence Network**
Passing patterns modelled as a weighted directed graph using NetworkX. Nodes are players, edges are weighted by pass volume, and each player receives a PageRank influence score and betweenness centrality value. The network is computed separately for wins and losses to reveal how Man Utd's passing structure changes when things go wrong.

Key finding: Juan Mata had the highest influence score (0.097) and was involved in 41 pre-shot sequences — more than any other player. That finding challenges the narrative around his role and is one the traditional stats would never surface.

**Module 3 — Pressing Effectiveness**
Tracking not just how many pressures a player applies, but whether those pressures actually won the ball back within five subsequent events. Volume alone tells you nothing about pressing value. All effectiveness rates are presented with 95% Wilson score confidence intervals so you can immediately see which findings are reliable and which need more data.

Key finding: The team's overall pressing effectiveness was 18.5%. Morgan Schneiderlin led in volume (510 pressures) with a reliable CI of 15.5% to 22.2%. Timothy Fosu-Mensah had the highest raw rate at 31.1%, but with only 45 pressures the CI runs from 19.5% to 45.7% — meaning it warrants monitoring rather than conclusion.

**Module 4 — Recruitment Analytics**
Multi-dimensional player profiling using cosine similarity across 10 performance metrics. Build a profile for any Man Utd player and the system searches the full comparison dataset for statistically similar profiles at other clubs. The output is a ranked shortlist with similarity scores and comparative charts.

Key findings: Riyad Mahrez (Leicester) was the closest statistical match to Martial across all ten dimensions. N'Golo Kanté (Leicester) emerged as the closest match to Schneiderlin — a signing Man Utd never made. Mesut Özil was statistically comparable to Mata in creative output metrics.

**Module 5 — Tactical Spatial Analysis**
KDE heatmaps built from shot and pressure location data, broken down by match result. Where does Man Utd attack in wins versus losses? Where do they press? Where do they lose the ball? These questions need spatial data to answer properly, and StatsBomb's coordinate system makes that possible.

Key finding: In wins, pressing is concentrated and structured in the middle third. In losses, pressing becomes scattered across the pitch — visible evidence of tactical breakdown that would be invisible in aggregate stats.

---

## How the Data Pipeline Works

The project runs in four steps. Each step feeds into the next and nothing is hardcoded — everything is generated fresh from the raw StatsBomb data.

Step one pulls raw event data from the StatsBomb open data API for all 38 Man Utd Premier League matches plus 60 additional matches from comparison clubs. Step two transforms those raw events through an ETL layer into 13 clean, structured datasets. Step three runs all five analytical modules against those datasets. Step four builds the dashboard and launches the Streamlit app.

The data lives in two folders: `data/raw/` for the original cached API responses and `data/processed/` for everything that comes out of the transformation and analysis stages.

---

## How to Run This (Step by Step in VS Code)

**Prerequisites**

You need Python 3.10 or above and Git installed. You can check by opening a terminal and running `python --version` and `git --version`.

**Step 1 — Clone the repo**

Open VS Code, press Ctrl+Shift+P (Cmd+Shift+P on Mac), type "Git: Clone", and paste the repo URL. Choose a folder on your machine to clone into. VS Code will open the project automatically.

**Step 2 — Open a terminal in VS Code**

Go to Terminal in the top menu and click New Terminal. This opens a terminal inside VS Code pointing at the project folder.

**Step 3 — Create a virtual environment**

In the terminal, run:

```
python -m venv venv
```

Then activate it:

On Windows: `venv\Scripts\activate`
On Mac/Linux: `source venv/bin/activate`

You should see `(venv)` appear at the start of your terminal prompt. This keeps all the project's packages separate from the rest of your machine.

**Step 4 — Install all dependencies**

```
pip install -r requirements.txt
```

This installs everything the project needs. It takes two to three minutes the first time.

**Step 5 — Fetch the data**

```
python ingestion/fetch_data.py
```

This pulls all Man Utd match events from StatsBomb and saves them to `data/raw/`. It fetches 38 Man Utd matches plus 60 comparison matches from other PL clubs. Takes about two to three minutes depending on your internet connection.

**Step 6 — Transform the data**

```
python transformation/transform.py
```

Cleans and structures the raw events into analysis-ready tables in `data/processed/`.

**Step 7 — Run all five analytical modules**

```
python analysis/module1_xg.py
python analysis/module2_pass_network.py
python analysis/module3_pressing.py
python analysis/module4_recruitment.py
python analysis/module5_spatial.py
```

Each module prints its key findings as it runs. You will see the xG model AUC, the top network nodes, pressing effectiveness rates, and so on in the terminal output.

**Step 8 — Launch the Streamlit app**

```
streamlit run app.py
```

This opens the interactive dashboard in your browser automatically at `http://localhost:8501`. Use the sidebar to navigate between the six pages.

**Step 9 — Build the static dashboard (optional)**

```
python dashboards/build_dashboard.py
```

Generates the 15-panel matplotlib dashboard image in `dashboards/`.

---

## Deploying to Streamlit Cloud (Free, Public Link)

Once your code is on GitHub, deploying takes about ten minutes and gives you a live URL you can share with anyone.

Go to share.streamlit.io and sign in with your GitHub account. Click "New app", select your repository, set the branch to main, and set the main file path to `app.py`. Click deploy. Streamlit Cloud installs everything from `requirements.txt` automatically and your app will be live at a URL like `your-username-manutd-platform.streamlit.app` within five minutes.

Make sure your `data/processed/` folder is committed to GitHub before deploying so the app has data to load without needing to re-run the pipeline on startup.

---

## Project Structure

```
manutd-performance-platform/
    app.py                      the interactive Streamlit dashboard
    requirements.txt            all Python dependencies
    setup_data.py               runs the full pipeline in one command
    README.md                   this file

    ingestion/
        fetch_data.py           pulls raw data from StatsBomb API

    transformation/
        transform.py            ETL pipeline, cleans and structures raw events

    analysis/
        module1_xg.py           expected goals model with freeze frame features
        module2_pass_network.py player influence network and graph analytics
        module3_pressing.py     pressing effectiveness with confidence intervals
        module4_recruitment.py  player similarity and recruitment profiling
        module5_spatial.py      tactical heatmaps and spatial zone analysis
        run_all.py              runs all five modules in sequence

    database/
        schema.sql              PostgreSQL schema with indexes and analytical views
        load_data.py            loads processed CSVs into PostgreSQL

    dashboards/
        build_dashboard.py      builds the static 15-panel matplotlib dashboard

    data/
        raw/                    cached StatsBomb API responses (CSV)
        processed/              13 analysis-ready output tables (CSV)
```

---

## Numbers at a Glance

The project processes 38 Premier League matches, 134,053 Man Utd events, and 209,995 comparison events from other clubs. The xG model was trained on 2,394 shots with 11 features and reaches an AUC of 0.794. The pass network identifies influence scores across every player with 10 or more passes. The pressing module tracks 5,656 pressure events. The recruitment profiler runs cosine similarity across 80+ player profiles built from the comparison dataset.

---

## Tech Stack

The project uses Python as its core language. Data ingestion uses StatsBombPy. Transformation and analysis use Pandas and NumPy. The xG model uses scikit-learn (LogisticRegression with StandardScaler pipeline). Graph analytics use NetworkX. Spatial analysis uses SciPy KDE. Visualisation uses Plotly for the interactive app and Matplotlib for the static dashboard. The database schema is written for PostgreSQL with SQLAlchemy as the connection layer. The app is built in Streamlit.

---

## A Note on Confidence Intervals

Every effectiveness rate in this project is shown with 95% Wilson score confidence intervals. This matters because small sample sizes produce wide intervals. Fosu-Mensah's 31.1% pressing effectiveness rate is the highest in the squad, but with 45 pressures the CI runs from 19.5% to 45.7% — which means it is an interesting signal to monitor, not a conclusion. Schneiderlin's 18.6% rate at 510 pressures has a CI of 15.5% to 22.2% — that is a reliable number. The project distinguishes between findings and reliable findings throughout.

---

## Author

Yash Chabukswar
MSc Data Science, University of Manchester, 2025 to 2026
Global Futures Scholar
yash18chabukswar@gmail.com
linkedin.com/in/yash-chabukswar

Data source: StatsBomb Open Data at github.com/statsbomb/open-data, used under the StatsBomb Open Data Licence.
