"""
MODULE 4: Recruitment Analytics
Builds multi-dimensional player profiles and identifies
comparable players from other PL teams.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

def build_player_profile(events, team=None):
    if team:
        ev = events[events['team'] == team].copy()
    else:
        ev = events.copy()

    # Per-player metrics
    shots = ev[ev['type']=='Shot'].groupby(['player','team']).agg(
        shots=('type','count'),
        goals=('shot_outcome', lambda x:(x=='Goal').sum()),
        shots_on_target=('shot_outcome', lambda x: x.isin(['Goal','Saved']).sum())
    ).reset_index()

    passes = ev[ev['type']=='Pass'].groupby(['player','team']).agg(
        passes=('type','count'),
        pass_completions=('pass_outcome', lambda x: x.isna().sum())
    ).reset_index()
    passes['pass_accuracy'] = (passes['pass_completions']/passes['passes']*100).round(1)

    pressures = ev[ev['type']=='Pressure'].groupby(['player','team']).agg(
        pressures=('type','count')
    ).reset_index()

    dribbles = ev[ev['type']=='Dribble'].groupby(['player','team']).agg(
        dribble_attempts=('type','count'),
        dribble_success=('dribble_outcome', lambda x:(x=='Complete').sum())
    ).reset_index()
    dribbles['dribble_rate'] = (dribbles['dribble_success']/dribbles['dribble_attempts']*100).round(1)

    ball_recoveries = ev[ev['type']=='Ball Recovery'].groupby(['player','team']).agg(
        ball_recoveries=('type','count')
    ).reset_index()

    # Merge
    profile = shots.merge(passes, on=['player','team'], how='outer')
    profile = profile.merge(pressures, on=['player','team'], how='outer')
    profile = profile.merge(dribbles, on=['player','team'], how='outer')
    profile = profile.merge(ball_recoveries, on=['player','team'], how='outer')
    profile = profile.fillna(0)

    profile['conversion_rate'] = (profile['goals']/profile['shots'].replace(0,np.nan)*100).round(1).fillna(0)
    profile['shot_accuracy'] = (profile['shots_on_target']/profile['shots'].replace(0,np.nan)*100).round(1).fillna(0)

    return profile

def find_similar_players(target_player, target_team, all_profiles, top_n=5):
    feature_cols = ['shots','goals','passes','pass_accuracy','pressures',
                    'dribble_attempts','dribble_rate','ball_recoveries',
                    'conversion_rate','shot_accuracy']

    target = all_profiles[
        (all_profiles['player'] == target_player) &
        (all_profiles['team'] == target_team)
    ]
    if len(target) == 0:
        return pd.DataFrame()

    # Exclude Man Utd from comparison
    comparison = all_profiles[all_profiles['team'] != target_team].copy()
    comparison = comparison[comparison['passes'] >= 100]  # Minimum game time filter

    combined = pd.concat([target, comparison], ignore_index=True)
    X = combined[feature_cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    target_vec = X_scaled[0].reshape(1, -1)
    comp_vecs = X_scaled[1:]

    similarities = cosine_similarity(target_vec, comp_vecs)[0]
    comparison = comparison.reset_index(drop=True)
    comparison['similarity'] = similarities

    return comparison.sort_values('similarity', ascending=False).head(top_n)[
        ['player','team','similarity','goals','shots','passes',
         'pass_accuracy','pressures','conversion_rate']
    ]


if __name__ == '__main__':
    import os
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW = os.path.join(BASE, 'data', 'raw')
    PROC = os.path.join(BASE, 'data', 'processed')
    os.makedirs(PROC, exist_ok=True)

    print("[MODULE 4] Building Recruitment Profiler...")
    mu_events = pd.read_csv(os.path.join(RAW, 'mu_events.csv'), low_memory=False)
    comp_events = pd.read_csv(os.path.join(RAW, 'comparison_events.csv'), low_memory=False)

    all_events = pd.concat([mu_events, comp_events], ignore_index=True)
    all_profiles = build_player_profile(all_events)
    all_profiles.to_csv(os.path.join(PROC, 'all_player_profiles.csv'), index=False)

    mu_profile = all_profiles[all_profiles['team']=='Manchester United']
    mu_profile.to_csv(os.path.join(PROC, 'mu_player_profiles.csv'), index=False)

    targets = [
        ('Anthony Martial', 'Manchester United'),
        ('Wayne Mark Rooney', 'Manchester United'),
        ('Juan Manuel Mata García', 'Manchester United'),
        ('Morgan Schneiderlin', 'Manchester United'),
    ]

    all_comps = []
    for player, team in targets:
        print(f"\nPlayers similar to {player}:")
        comps = find_similar_players(player, team, all_profiles)
        if len(comps) > 0:
            comps['target_player'] = player
            all_comps.append(comps)
            print(comps[['player','team','similarity','goals','passes','pass_accuracy']].to_string(index=False))

    if all_comps:
        pd.concat(all_comps).to_csv(os.path.join(PROC, 'recruitment_comps.csv'), index=False)

    print("\n[MODULE 4] Complete.")
