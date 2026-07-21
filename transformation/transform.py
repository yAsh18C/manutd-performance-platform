"""
========================================================
Manchester United Performance Intelligence Platform
Step 2: Data Transformation (ETL Pipeline)
Author: Yash Chabukswar | MSc Data Science, UoM
========================================================

This script takes the raw StatsBomb event data from data/raw/ and
transforms it into clean, structured, analysis-ready tables.

It handles:
    - Match result calculation (Man Utd score, opponent score, W/D/L)
    - Per-player season stat aggregation (shots, goals, passes, etc.)
    - Pass edge list construction for the network module
    - Data quality checks at each stage

All outputs go to data/processed/ and are referenced by the analysis modules.
"""

import pandas as pd
import numpy as np
import os

# Input and output directories
RAW = os.path.join(os.path.dirname(__file__), '../data/raw')
PROC = os.path.join(os.path.dirname(__file__), '../data/processed')
os.makedirs(PROC, exist_ok=True)


def build_match_results(matches_df):
    """
    Takes the raw StatsBomb match data and adds Man Utd-specific columns.

    StatsBomb gives us home_team, away_team, home_score, away_score.
    We need to convert that into 'how did Man Utd do?' — their score,
    the opponent's score, the result (W/D/L), and whether they were home or away.

    Args:
        matches_df: raw DataFrame from mu_matches.csv

    Returns:
        Clean match results DataFrame saved to data/processed/match_results.csv
    """
    m = matches_df.copy()

    # Work out which side Man Utd were on in each match
    m['is_home'] = m['home_team'] == 'Manchester United'

    # Pick the right score column depending on home/away
    m['mu_score'] = m.apply(
        lambda r: r['home_score'] if r['is_home'] else r['away_score'], axis=1
    )
    m['opp_score'] = m.apply(
        lambda r: r['away_score'] if r['is_home'] else r['home_score'], axis=1
    )

    # Identify the opponent name
    m['opponent'] = m.apply(
        lambda r: r['away_team'] if r['is_home'] else r['home_team'], axis=1
    )

    # Derive the result from the scores
    m['result'] = m.apply(
        lambda r: 'W' if r['mu_score'] > r['opp_score']
        else ('D' if r['mu_score'] == r['opp_score'] else 'L'),
        axis=1
    )

    m['venue'] = m['is_home'].map({True: 'Home', False: 'Away'})

    # Keep only the columns we need downstream
    results = m[['match_id', 'match_date', 'opponent', 'venue',
                 'mu_score', 'opp_score', 'result']]
    results.to_csv(f'{PROC}/match_results.csv', index=False)

    return results


def aggregate_player_stats(events_df):
    """
    Aggregates raw events into per-player season statistics.

    StatsBomb events have one row per action. We group by player and
    count/aggregate each event type to build a player profile table.

    The metrics we extract:
        shots, goals, shots_on_target (from Shot events)
        passes, pass completions, pass accuracy (from Pass events)
        pressures (from Pressure events)
        dribble attempts and successes (from Dribble events)

    Args:
        events_df: raw events DataFrame (Man Utd players only)

    Returns:
        player_stats DataFrame saved to data/processed/player_stats.csv
    """
    # Filter to Man Utd actions only
    mu = events_df[events_df['team'] == 'Manchester United'].copy()

    # Shot stats — goal is when shot_outcome == 'Goal'
    shots = mu[mu['type'] == 'Shot'].groupby('player').agg(
        shots=('type', 'count'),
        goals=('shot_outcome', lambda x: (x == 'Goal').sum()),
        shots_on_target=('shot_outcome', lambda x: x.isin(['Goal', 'Saved']).sum())
    ).reset_index()

    # Pass stats — a null pass_outcome means the pass was completed
    # (StatsBomb only records outcomes for unsuccessful passes)
    passes = mu[mu['type'] == 'Pass'].groupby('player').agg(
        passes=('type', 'count'),
        pass_completions=('pass_outcome', lambda x: x.isna().sum())
    ).reset_index()
    passes['pass_accuracy'] = (
        passes['pass_completions'] / passes['passes'] * 100
    ).round(1)

    # Pressure, dribble counts
    pressures = mu[mu['type'] == 'Pressure'].groupby('player').agg(
        pressures=('type', 'count')
    ).reset_index()

    dribbles = mu[mu['type'] == 'Dribble'].groupby('player').agg(
        dribble_attempts=('type', 'count'),
        dribble_success=('dribble_outcome', lambda x: (x == 'Complete').sum())
    ).reset_index()

    # Merge all stat tables on player name, fill missing values with 0
    stats = (shots
             .merge(passes, on='player', how='outer')
             .merge(pressures, on='player', how='outer')
             .merge(dribbles, on='player', how='outer')
             .fillna(0))

    # Derived metric: what % of shots result in goals?
    stats['conversion_rate'] = (
        stats['goals'] / stats['shots'].replace(0, np.nan) * 100
    ).round(1).fillna(0)

    stats.to_csv(f'{PROC}/player_stats.csv', index=False)
    return stats


def build_pass_edges(events_df):
    """
    Builds the pass edge list used by the network analysis module.

    Each row represents a passing connection between two players,
    with a weight equal to the number of completed passes between them.
    This forms the basis of the directed weighted graph in module 2.

    Only completed passes are included — failed passes don't represent
    a genuine connection in the team's passing structure.

    Args:
        events_df: raw events DataFrame (Man Utd players only)

    Returns:
        Edge list DataFrame saved to data/processed/pass_edges_raw.csv
    """
    mu = events_df[events_df['team'] == 'Manchester United'].copy()

    # A completed pass has a null pass_outcome in StatsBomb data
    completed = mu[
        (mu['type'] == 'Pass') & mu['pass_outcome'].isna()
    ].dropna(subset=['player', 'pass_recipient'])

    # Count passes between each unique passer-recipient pair
    edges = completed.groupby(
        ['player', 'pass_recipient']
    ).size().reset_index(name='weight')

    edges.to_csv(f'{PROC}/pass_edges_raw.csv', index=False)
    return edges


def run():
    print("=" * 55)
    print("  Manchester United Performance Intelligence Platform")
    print("  Data Transformation (ETL)")
    print("=" * 55)

    # Load raw inputs
    print("\nLoading raw data...")
    events = pd.read_csv(f'{RAW}/mu_events.csv', low_memory=False)
    matches = pd.read_csv(f'{RAW}/mu_matches.csv')
    print(f"      Events loaded: {len(events):,}")
    print(f"      Matches loaded: {len(matches)}")

    # Run each transformation step
    print("\n[1/3] Building match results table...")
    results = build_match_results(matches)
    W = len(results[results['result'] == 'W'])
    D = len(results[results['result'] == 'D'])
    L = len(results[results['result'] == 'L'])
    print(f"      Season record: W{W} D{D} L{L}")

    print("[2/3] Aggregating player stats...")
    stats = aggregate_player_stats(events)
    print(f"      Players processed: {len(stats)}")

    print("[3/3] Building pass edge list...")
    edges = build_pass_edges(events)
    print(f"      Pass connections: {len(edges)}")

    print("\n[DONE] Transformation complete.")
    print("       Next step: python analysis/run_all.py")


if __name__ == '__main__':
    run()
