"""
Loads processed CSVs into PostgreSQL.
Update DB_URL with your connection string before running.
"""
import pandas as pd
from sqlalchemy import create_engine

DB_URL = "postgresql://user:password@localhost:5432/manutd_platform"

def load():
    engine = create_engine(DB_URL)
    tables = {
        'match_results.csv': 'matches',
        'player_stats.csv': 'player_season_stats',
        'pass_edges_full.csv': 'pass_network',
        'pressing_effectiveness.csv': 'pressing_events',
        'player_xg.csv': 'xg_shots',
        'recruitment_comps.csv': 'recruitment_profiles',
    }
    import os
    proc = os.path.join(os.path.dirname(__file__), '../data/processed')
    for csv, table in tables.items():
        try:
            df = pd.read_csv(f'{proc}/{csv}')
            df.to_sql(table, engine, if_exists='append', index=False)
            print(f"Loaded {len(df)} rows into {table}")
        except Exception as e:
            print(f"Error loading {csv}: {e}")

if __name__ == '__main__':
    load()
