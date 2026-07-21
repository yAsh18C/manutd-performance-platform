"""
MODULE 5: Tactical Spatial Analysis
KDE heatmaps of attack, pressure, and ball loss zones.
Zone-based danger maps by match result.
"""
import pandas as pd
import numpy as np
import ast
import warnings
warnings.filterwarnings('ignore')

def parse_location(loc):
    try:
        return ast.literal_eval(loc) if isinstance(loc, str) else loc
    except:
        return [None, None]

def extract_locations(events, event_type, team=None, result_filter=None, matches_df=None):
    ev = events[events['type'] == event_type].copy()
    if team:
        ev = ev[ev['team'] == team]
    if result_filter is not None and matches_df is not None:
        ids = matches_df[matches_df['result'] == result_filter]['match_id'].tolist()
        ev = ev[ev['match_id'].isin(ids)]
    locs = ev['location'].apply(parse_location)
    ev = ev.copy()
    ev['x'] = locs.apply(lambda l: l[0] if l and l[0] else None)
    ev['y'] = locs.apply(lambda l: l[1] if l and l[1] else None)
    return ev.dropna(subset=['x','y'])[['x','y','match_id','player']]

def zone_analysis(locations_df):
    """Divide pitch into 6 zones and count events per zone."""
    def assign_zone(row):
        x, y = row['x'], row['y']
        col = 'Left' if y < 27 else ('Right' if y > 53 else 'Centre')
        row_z = 'Attacking' if x >= 80 else ('Middle' if x >= 40 else 'Defensive')
        return f"{row_z} {col}"
    locations_df = locations_df.copy()
    locations_df['zone'] = locations_df.apply(assign_zone, axis=1)
    return locations_df.groupby('zone').size().reset_index(name='count').sort_values('count', ascending=False)


if __name__ == '__main__':
    import os
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW = os.path.join(BASE, 'data', 'raw')
    PROC = os.path.join(BASE, 'data', 'processed')
    os.makedirs(PROC, exist_ok=True)

    print("[MODULE 5] Building Tactical Spatial Analysis...")
    events = pd.read_csv(os.path.join(RAW, 'mu_events.csv'), low_memory=False)
    matches = pd.read_csv(os.path.join(RAW, 'mu_matches.csv'))

    matches['result'] = matches.apply(lambda r: 'W' if
        (r['home_team']=='Manchester United' and r['home_score']>r['away_score']) or
        (r['away_team']=='Manchester United' and r['away_score']>r['home_score'])
        else ('D' if r['home_score']==r['away_score'] else 'L'), axis=1)

    # Shot locations by result
    for result in ['W', 'D', 'L']:
        shots = extract_locations(events, 'Shot', 'Manchester United', result, matches)
        shots.to_csv(os.path.join(PROC, f'shots_locs_{result}.csv'), index=False)
        print(f"  Shots in {result}: {len(shots)}")

    all_shots = extract_locations(events, 'Shot', 'Manchester United')
    all_shots.to_csv(os.path.join(PROC, 'shots_locs_all.csv'), index=False)

    # Pressure locations by result
    for result in ['W', 'D', 'L']:
        press = extract_locations(events, 'Pressure', 'Manchester United', result, matches)
        press.to_csv(os.path.join(PROC, f'press_locs_{result}.csv'), index=False)

    all_press = extract_locations(events, 'Pressure', 'Manchester United')
    all_press.to_csv(os.path.join(PROC, 'press_locs_all.csv'), index=False)

    # Ball loss locations
    ball_loss = events[
        (events['type'].isin(['Dispossessed', 'Miscontrol'])) &
        (events['team'] == 'Manchester United')
    ].copy()
    locs = ball_loss['location'].apply(parse_location)
    ball_loss['x'] = locs.apply(lambda l: l[0] if l else None)
    ball_loss['y'] = locs.apply(lambda l: l[1] if l else None)
    ball_loss = ball_loss.dropna(subset=['x', 'y'])
    ball_loss[['x', 'y', 'match_id', 'player']].to_csv(
        os.path.join(PROC, 'ball_loss_locs.csv'), index=False)

    # Zone analysis
    shot_zones = zone_analysis(all_shots)
    press_zones = zone_analysis(all_press)
    shot_zones.to_csv(os.path.join(PROC, 'shot_zones.csv'), index=False)
    press_zones.to_csv(os.path.join(PROC, 'press_zones.csv'), index=False)

    print("\nDangerous Shot Zones:")
    print(shot_zones.head(6).to_string(index=False))
    print("\nPressing Zones:")
    print(press_zones.head(6).to_string(index=False))
    print("\n[MODULE 5] Complete.")
