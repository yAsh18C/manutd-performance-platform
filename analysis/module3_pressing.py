"""
========================================================
Manchester United Performance Intelligence Platform
Module 3: Pressing Effectiveness Analysis
Author: Yash Chabukswar | MSc Data Science, UoM
========================================================

Most pressing metrics count how many times a player presses.
That tells you about work rate, but not about value.

This module asks a better question: when a player presses,
does it actually win the ball back?

To answer that, we track every Pressure event and look at
the next five events in the same match to see whether
Man Utd regained possession. If a Ball Recovery, Interception,
or Duel follows the pressure before the opponent makes a sustained
Carry, we credit the press as successful.

All effectiveness rates are reported with 95% Wilson score
confidence intervals. This matters because small sample sizes
produce wide intervals — a player with 30 pressures and a 30%
rate has a CI from roughly 16% to 49%. We distinguish between
findings that are reliable and ones that need more data.

Wilson score interval is preferred over the simple Wald interval
for proportions because it performs better at values near 0 or 1
and with small samples.
"""

import pandas as pd
import numpy as np
import ast
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

RAW = os.path.join(os.path.dirname(__file__), '../data/raw')
PROC = os.path.join(os.path.dirname(__file__), '../data/processed')
os.makedirs(PROC, exist_ok=True)


def parse_location(loc):
    """Parses StatsBomb coordinate strings into [x, y] lists."""
    try:
        return ast.literal_eval(loc) if isinstance(loc, str) else loc
    except:
        return [None, None]


def wilson_ci(successes, n, confidence=0.95):
    """
    Computes the Wilson score confidence interval for a proportion.

    The Wilson interval is more accurate than the simple p +/- z*SE
    formula (Wald interval) especially when n is small or p is near 0 or 1.
    It is the standard choice for bounded proportions in analytics.

    Args:
        successes: number of positive outcomes (balls won)
        n: total trials (pressures applied)
        confidence: confidence level, default 0.95 for 95% CI

    Returns:
        (lower, upper) as percentages, rounded to 1 decimal place
    """
    if n == 0:
        return 0.0, 0.0

    z = stats.norm.ppf((1 + confidence) / 2)
    p = successes / n

    # Wilson score formula
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denominator

    lower = round(max(0, centre - margin) * 100, 1)
    upper = round(min(1, centre + margin) * 100, 1)
    return lower, upper


def assign_pitch_zone(x):
    """
    Assigns a pitch zone based on x-coordinate.

    StatsBomb pitches are 120 units long, with 0 at the defensive end
    and 120 at the attacking goal. We split into thirds:
        Defensive Third:  x < 40
        Middle Third:    40 <= x < 80
        Attacking Third:  x >= 80
    """
    if x is None:
        return 'Unknown'
    if x >= 80:
        return 'Attacking Third'
    elif x >= 40:
        return 'Middle Third'
    else:
        return 'Defensive Third'


def analyse_pressing(events_df, team='Manchester United'):
    """
    For every pressure event by Man Utd, look at the next 5 events
    in that match and determine whether possession was regained.

    We define a successful press as one where the next action
    involving Man Utd is a Ball Recovery, Interception, or winning
    a Duel — before the opponent completes a Carry (which signals
    they escaped the press successfully).

    Args:
        events_df: full events DataFrame sorted by match_id and event index
        team: the team whose pressing we are evaluating

    Returns:
        DataFrame with one row per pressure event and a 'won_ball' boolean
    """
    # Sort so we can reliably look at "next events"
    events_df = events_df.sort_values(['match_id', 'index']).reset_index(drop=True)

    # Filter to this team's pressure events
    pressures = events_df[(events_df['type'] == 'Pressure') &
                          (events_df['team'] == team)].copy()

    results = []
    for _, press in pressures.iterrows():
        # Get all events in this match
        match_events = events_df[
            events_df['match_id'] == press['match_id']
        ].reset_index(drop=True)

        # Find the position of this pressure in the match event sequence
        press_positions = match_events[match_events['index'] == press['index']].index

        if len(press_positions) == 0:
            continue

        press_pos = press_positions[0]

        # Look at the next 5 events
        next_events = match_events.iloc[press_pos + 1: press_pos + 6]

        # Check each subsequent event for possession change
        won_ball = False
        for _, ev in next_events.iterrows():
            if ev['team'] == team and ev['type'] in ['Ball Recovery', 'Interception', 'Duel']:
                # Man Utd won the ball back
                won_ball = True
                break
            if ev['team'] != team and ev['type'] == 'Carry':
                # Opponent carried the ball away — press failed
                break

        # Extract location and zone
        loc = parse_location(press.get('location'))
        x = loc[0] if loc else None
        y = loc[1] if loc else None

        results.append({
            'player': press['player'],
            'match_id': press['match_id'],
            'won_ball': won_ball,
            'x': x,
            'y': y,
            'zone': assign_pitch_zone(x)
        })

    return pd.DataFrame(results)


def compute_player_stats(press_df, min_pressures=20):
    """
    Aggregates pressing effectiveness per player and adds confidence intervals.

    Only players with at least min_pressures pressures are included
    to avoid misleadingly precise stats from tiny samples.

    Args:
        press_df: output from analyse_pressing()
        min_pressures: minimum pressures to include a player

    Returns:
        Player-level stats DataFrame with CIs and reliability labels
    """
    stats = press_df.groupby('player').agg(
        total_pressures=('won_ball', 'count'),
        balls_won=('won_ball', 'sum'),
    ).reset_index()

    # Point estimate for effectiveness rate
    stats['effectiveness_rate'] = (
        stats['balls_won'] / stats['total_pressures'] * 100
    ).round(1)

    # 95% Wilson score confidence intervals
    stats['ci_lower'] = stats.apply(
        lambda r: wilson_ci(int(r['balls_won']), int(r['total_pressures']))[0],
        axis=1
    )
    stats['ci_upper'] = stats.apply(
        lambda r: wilson_ci(int(r['balls_won']), int(r['total_pressures']))[1],
        axis=1
    )

    # Reliability label — this is the key analytical caveat:
    # small samples mean wide CIs and conclusions should be tentative
    stats['reliability'] = stats['total_pressures'].apply(
        lambda n: 'High (n>100)' if n > 100
        else ('Medium (n=50-100)' if n >= 50 else 'Low (n<50)')
    )

    return stats[stats['total_pressures'] >= min_pressures].sort_values(
        'effectiveness_rate', ascending=False
    )


def run():
    print("[MODULE 3] Analysing Pressing Effectiveness...")

    events = pd.read_csv(f'{RAW}/mu_events.csv', low_memory=False)

    # Run the pressing analysis
    press_df = analyse_pressing(events)
    press_df.to_csv(f'{PROC}/pressing_effectiveness.csv', index=False)

    # Compute player-level stats with CIs
    player_stats = compute_player_stats(press_df)
    player_stats.to_csv(f'{PROC}/pressing_player_stats.csv', index=False)

    # Zone-level breakdown
    zone_stats = press_df.groupby(['zone', 'won_ball']).size().unstack(
        fill_value=0
    ).reset_index()
    zone_stats.to_csv(f'{PROC}/pressing_zone_stats.csv', index=False)

    overall_rate = press_df['won_ball'].mean() * 100
    print(f"\n  Pressures analysed: {len(press_df):,}")
    print(f"  Team effectiveness: {overall_rate:.1f}%")
    print(f"\n  Player effectiveness (with 95% CIs):")
    print(player_stats[['player', 'total_pressures', 'effectiveness_rate',
                          'ci_lower', 'ci_upper', 'reliability']].to_string(index=False))

    print("\n[MODULE 3] Complete. Output: data/processed/pressing_player_stats.csv")


if __name__ == '__main__':
    run()
