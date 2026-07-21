"""
========================================================
Manchester United Performance Intelligence Platform
Step 1: Data Ingestion Pipeline
Author: Yash Chabukswar | MSc Data Science, UoM
========================================================

This script pulls all Man Utd Premier League 2015/16 match events
from the StatsBomb open data API and saves them locally as CSV files.

It also fetches a set of comparison matches from other top-six clubs
so the recruitment module has a wider dataset to search against.

StatsBomb open data is free and requires no authentication.
All data is saved to data/raw/ for use by the transformation step.
"""

from statsbombpy import sb
import pandas as pd
import os

# All raw data goes here — transformation reads from this folder
RAW = os.path.join(os.path.dirname(__file__), '../data/raw')
os.makedirs(RAW, exist_ok=True)


def fetch_competitions():
    """
    Pulls the list of all available competitions from StatsBomb.
    Not strictly required for the pipeline, but useful for reference
    and for checking what seasons are available.
    """
    print("[1/3] Fetching available competitions...")
    comps = sb.competitions()
    comps.to_csv(f'{RAW}/competitions.csv', index=False)
    print(f"      Saved {len(comps)} competitions to data/raw/competitions.csv")
    return comps


def fetch_mu_matches():
    """
    Pulls all Premier League 2015/16 matches and filters to those
    involving Manchester United.

    StatsBomb uses competition_id=2 for the Premier League
    and season_id=27 for the 2015/16 season.

    Returns both the full PL match list (needed for comparison clubs)
    and the Man Utd subset.
    """
    print("[2/3] Fetching Premier League 2015/16 matches...")
    all_matches = sb.matches(competition_id=2, season_id=27)

    # Filter to Man Utd home and away games
    mu = all_matches[
        (all_matches['home_team'] == 'Manchester United') |
        (all_matches['away_team'] == 'Manchester United')
    ].copy()

    # Save both — full list needed for recruitment comparison
    all_matches.to_csv(f'{RAW}/all_pl_matches.csv', index=False)
    mu.to_csv(f'{RAW}/mu_matches.csv', index=False)

    print(f"      Man Utd matches: {len(mu)} | All PL matches: {len(all_matches)}")
    return all_matches, mu


def fetch_events(match_ids, output_filename):
    """
    Fetches all events for a list of match IDs and combines them
    into a single DataFrame.

    Each event represents one action in the match — a pass, shot,
    pressure, dribble, carry, and so on. StatsBomb events also include
    freeze frame data for shots, which we use in the xG model.

    Args:
        match_ids: list of StatsBomb match IDs to fetch
        output_filename: where to save the combined events CSV

    Returns:
        Combined DataFrame of all events across all matches
    """
    all_events = []
    for i, mid in enumerate(match_ids, 1):
        try:
            ev = sb.events(match_id=mid)
            ev['match_id'] = mid  # Tag each event with its match ID
            all_events.append(ev)
            if i % 10 == 0:
                print(f"      {i}/{len(match_ids)} matches loaded...")
        except Exception as e:
            print(f"      Warning: could not load match {mid} — {e}")

    df = pd.concat(all_events, ignore_index=True)
    df.to_csv(f'{RAW}/{output_filename}', index=False)
    print(f"      Saved {len(df):,} events to data/raw/{output_filename}")
    return df


def run():
    print("=" * 55)
    print("  Manchester United Performance Intelligence Platform")
    print("  Data Ingestion Pipeline")
    print("=" * 55)

    fetch_competitions()
    all_matches, mu = fetch_mu_matches()

    print("\n[3/3] Fetching match events...")
    print("      Fetching Man Utd events (38 matches)...")
    fetch_events(mu['match_id'].tolist(), 'mu_events.csv')

    # For recruitment: fetch 60 matches from top-six comparison clubs.
    # We use a subset rather than all 380 PL matches to keep runtime manageable.
    comparison_clubs = ['Leicester City', 'Arsenal', 'Tottenham Hotspur',
                        'Manchester City', 'Liverpool', 'Chelsea']
    comparison_ids = all_matches[
        (all_matches['home_team'].isin(comparison_clubs)) |
        (all_matches['away_team'].isin(comparison_clubs))
    ]['match_id'].tolist()[:60]

    print(f"      Fetching comparison club events ({len(comparison_ids)} matches)...")
    fetch_events(comparison_ids, 'comparison_events.csv')

    print("\n[DONE] Ingestion complete.")
    print("       Next step: python transformation/transform.py")


if __name__ == '__main__':
    run()
