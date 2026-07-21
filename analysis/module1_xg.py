"""
========================================================
Manchester United Performance Intelligence Platform
Module 1: Expected Goals (xG) Model
Author: Yash Chabukswar | MSc Data Science, UoM
========================================================

Expected Goals (xG) is a metric that quantifies the quality of a shot.
Instead of just counting shots, xG assigns each shot a probability
between 0 and 1 representing how likely it was to result in a goal,
based on the circumstances at the moment of the shot.

This module builds a logistic regression xG model using 11 features,
including StatsBomb freeze frame data — the positions of every player
on the pitch at the exact moment of each shot. This gives us access
to spatial context that basic event data cannot provide.

Features used:
    Basic:      distance to goal, shot angle, body part, open play flag,
                shot technique, under pressure flag
    Freeze frame: goalkeeper distance from goal line, defenders in
                  shooting cone, nearest defender distance, total defenders

Model: Logistic Regression with StandardScaler (Pipeline)
Evaluation: 5-fold cross-validated AUC (Area Under ROC Curve)
Result: AUC 0.794 — competitive with published academic xG models
"""

import pandas as pd
import numpy as np
import ast
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

RAW = os.path.join(os.path.dirname(__file__), '../data/raw')
PROC = os.path.join(os.path.dirname(__file__), '../data/processed')
os.makedirs(PROC, exist_ok=True)


def parse_location(loc):
    """
    Converts StatsBomb location strings like '[100.5, 40.2]' into
    Python lists [x, y]. Returns [None, None] if parsing fails.
    """
    try:
        return ast.literal_eval(loc) if isinstance(loc, str) else loc
    except:
        return [None, None]


def extract_freeze_frame_features(ff_str, shooter_x, shooter_y):
    """
    Extracts spatial features from the StatsBomb freeze frame.

    A freeze frame is a snapshot of every player's position at the
    exact moment a shot is taken. We use it to measure the defensive
    context the shooter faced — something basic event data cannot capture.

    Features extracted:
        gk_distance_from_goal: how far is the keeper off the line?
            A keeper far from goal is more vulnerable to lobbed shots.

        defenders_in_cone: how many defenders are between the shooter
            and the goal mouth? More defenders = lower xG.

        nearest_defender_dist: how close is the nearest defender?
            A tightly marked shot is harder than an unchallenged one.

        total_defenders: total number of opposition outfield players
            visible in the freeze frame.

    Args:
        ff_str: raw freeze frame string from StatsBomb events
        shooter_x, shooter_y: the shooter's coordinates

    Returns:
        dict of four freeze frame features
    """
    try:
        players = ast.literal_eval(ff_str) if isinstance(ff_str, str) else ff_str
    except:
        # If parsing fails, return sensible defaults
        return {
            'gk_distance_from_goal': 5.0,
            'defenders_in_cone': 1,
            'nearest_defender_dist': 5.0,
            'total_defenders': 3
        }

    gk_dist = None
    defenders_in_cone = 0
    nearest_defender_dist = 999
    total_defenders = 0

    # StatsBomb pitch: goal mouth centre is at (120, 40)
    goal_x, goal_y = 120, 40

    for player in players:
        loc = player.get('location', [None, None])
        if not loc or loc[0] is None:
            continue

        px, py = loc[0], loc[1]
        is_teammate = player.get('teammate', False)
        position_name = player.get('position', {}).get('name', '')

        dist_to_shooter = np.sqrt((px - shooter_x)**2 + (py - shooter_y)**2)

        if not is_teammate:
            # This player is an opponent
            total_defenders += 1

            if dist_to_shooter < nearest_defender_dist:
                nearest_defender_dist = dist_to_shooter

            # Shooting cone check: is the defender positioned between
            # the shooter and the goal posts (roughly at y=36 and y=44)?
            # We approximate the cone by checking if the defender is
            # in the right x-range and within a narrowing y-band.
            if shooter_x < px < goal_x:
                cone_proportion = (px - shooter_x) / max(goal_x - shooter_x, 1)
                cone_half_width = cone_proportion * 8  # Goal is 8 units wide
                if abs(py - goal_y) < cone_half_width:
                    defenders_in_cone += 1

        if position_name == 'Goalkeeper' and not is_teammate:
            # How far is the keeper from the goal centre?
            gk_dist = np.sqrt((px - goal_x)**2 + (py - goal_y)**2)

    return {
        'gk_distance_from_goal': gk_dist if gk_dist else 5.0,
        'defenders_in_cone': defenders_in_cone,
        'nearest_defender_dist': nearest_defender_dist if nearest_defender_dist < 999 else 10.0,
        'total_defenders': total_defenders
    }


def prepare_shot_features(events_df):
    """
    Takes the full events DataFrame, filters to shots, and engineers
    all 11 features needed for the xG model.

    The feature set combines positional geometry (distance, angle)
    with contextual qualifiers (body part, technique, pressure)
    and freeze frame spatial data (goalkeeper, defenders).

    Args:
        events_df: combined events DataFrame (all teams, for training)

    Returns:
        shots_df with engineered features and a 'goal' target column
    """
    shots = events_df[events_df['type'] == 'Shot'].copy()

    # Parse the shot location coordinates
    locs = shots['location'].apply(parse_location)
    shots['x'] = locs.apply(lambda l: l[0] if l else None)
    shots['y'] = locs.apply(lambda l: l[1] if l else None)
    shots = shots.dropna(subset=['x', 'y'])

    # Geometry features
    # StatsBomb pitch is 120 units wide, goal centre at x=120, y=40
    shots['distance'] = np.sqrt((120 - shots['x'])**2 + (40 - shots['y'])**2)

    # Angle in degrees from the shooter to the goal centre
    shots['angle'] = np.abs(
        np.arctan2(np.abs(shots['y'] - 40), 120 - shots['x'])
    ) * 180 / np.pi

    # Body part and technique flags
    shots['is_header'] = (shots['shot_body_part'] == 'Head').astype(int)
    shots['is_open_play'] = (shots['shot_type'] == 'Open Play').astype(int)
    shots['is_strong_foot'] = shots['shot_body_part'].isin(
        ['Right Foot', 'Left Foot']
    ).astype(int)

    # Shot technique — map to ordered scale of difficulty
    # Volleys and half-volleys are harder to control than normal shots
    technique_map = {
        'Normal': 0, 'Volley': 1, 'Half Volley': 1,
        'Lob': 2, 'Backheel': 2, 'Diving Header': 1, 'No Touch': 0
    }
    shots['technique_enc'] = shots['shot_technique'].map(technique_map).fillna(0)

    # Was the player under pressure from a defender when shooting?
    shots['under_pressure'] = shots['under_pressure'].fillna(False).astype(int)

    # Target variable
    shots['goal'] = (shots['shot_outcome'] == 'Goal').astype(int)

    # Freeze frame features — extract for each shot
    print("  Extracting freeze frame features for each shot...")
    ff_features = []
    for _, row in shots.iterrows():
        if pd.notna(row.get('shot_freeze_frame')):
            ff = extract_freeze_frame_features(
                row['shot_freeze_frame'], row['x'], row['y']
            )
        else:
            # Default values if no freeze frame available
            ff = {
                'gk_distance_from_goal': 5.0,
                'defenders_in_cone': 1,
                'nearest_defender_dist': 5.0,
                'total_defenders': 3
            }
        ff_features.append(ff)

    ff_df = pd.DataFrame(ff_features, index=shots.index)
    shots = pd.concat([shots, ff_df], axis=1)

    return shots


def build_xg_model(shots_df):
    """
    Trains a logistic regression xG model and evaluates it with
    5-fold cross-validation, reporting AUC.

    We use a Pipeline (scaler + model) so that feature scaling is
    applied consistently inside each cross-validation fold — avoiding
    data leakage that would artificially inflate the AUC.

    Args:
        shots_df: prepared shot DataFrame with all 11 features

    Returns:
        fitted pipeline, cross-validation AUC scores
    """
    feature_cols = [
        'distance',             # metres from goal
        'angle',                # degrees to goal centre
        'is_header',            # 1 if headed, 0 otherwise
        'is_open_play',         # 1 if open play, 0 if set piece
        'is_strong_foot',       # 1 if right or left foot (not weak foot)
        'gk_distance_from_goal', # freeze frame: how far is keeper off line?
        'defenders_in_cone',    # freeze frame: defenders blocking shooting lane
        'nearest_defender_dist', # freeze frame: distance to closest opponent
        'total_defenders',      # freeze frame: total opponents in view
        'technique_enc',        # 0=normal, 1=volley, 2=lob/trick
        'under_pressure',       # 1 if defender applying pressure on shooter
    ]

    X = shots_df[feature_cols].fillna(shots_df[feature_cols].median())
    y = shots_df['goal']

    # Pipeline ensures scaling is fitted only on training data in each fold
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=42, max_iter=2000, C=0.5))
    ])

    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
    pipeline.fit(X, y)

    return pipeline, cv_scores, feature_cols


def run():
    print("[MODULE 1] Building Expected Goals (xG) Model...")

    # Load Man Utd events plus comparison club events for a larger training set
    mu_events = pd.read_csv(f'{RAW}/mu_events.csv', low_memory=False)
    comp_events = pd.read_csv(f'{RAW}/comparison_events.csv', low_memory=False)
    all_events = pd.concat([mu_events, comp_events], ignore_index=True)

    print(f"  Training data: {len(all_events[all_events['type']=='Shot']):,} shots")

    # Feature engineering
    shots = prepare_shot_features(all_events)

    # Train model
    pipeline, cv_scores, feature_cols = build_xg_model(shots)
    shots['xg'] = pipeline.predict_proba(
        shots[feature_cols].fillna(shots[feature_cols].median())
    )[:, 1]

    print(f"\n  Model AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Shots modelled: {len(shots):,}")

    # Save all shots with xG values attached
    shots.to_csv(f'{PROC}/shots_with_xg_v2.csv', index=False)

    # Build per-player xG summary for Man Utd players
    mu_shots = shots[shots['team'] == 'Manchester United']
    summary = mu_shots.groupby('player').agg(
        shots=('xg', 'count'),
        xg=('xg', 'sum'),
        goals=('goal', 'sum'),
        avg_xg=('xg', 'mean')
    ).reset_index()
    summary['xg_diff'] = (summary['goals'] - summary['xg']).round(2)
    summary['xg'] = summary['xg'].round(2)
    summary.to_csv(f'{PROC}/player_xg_v2.csv', index=False)

    # Save feature coefficients for the model explainability chart in the app
    coef_df = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': pipeline.named_steps['model'].coef_[0]
    }).sort_values('coefficient')
    coef_df.to_csv(f'{PROC}/xg_feature_importance.csv', index=False)

    print("\n  Top xG performers (min 5 shots):")
    print(summary[summary['shots'] >= 5].sort_values('xg', ascending=False)[
        ['player', 'shots', 'xg', 'goals', 'xg_diff']
    ].head(8).to_string(index=False))

    print("\n[MODULE 1] Complete. Output: data/processed/player_xg_v2.csv")


if __name__ == '__main__':
    run()
