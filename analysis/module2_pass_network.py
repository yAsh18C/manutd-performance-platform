"""
MODULE 2: Player Influence Network
Graph analytics on passing patterns.
Identifies key connectors, partnerships, and network changes by result.
"""
import pandas as pd
import numpy as np
import networkx as nx
import ast
import warnings
warnings.filterwarnings('ignore')

def parse_location(loc):
    try:
        return ast.literal_eval(loc) if isinstance(loc, str) else loc
    except:
        return [None, None]

def build_pass_network(events, team='Manchester United', result_filter=None, matches_df=None):
    passes = events[(events['type'] == 'Pass') & (events['team'] == team)].copy()

    if result_filter and matches_df is not None:
        filtered_ids = matches_df[matches_df['result'] == result_filter]['match_id'].tolist()
        passes = passes[passes['match_id'].isin(filtered_ids)]

    # Only completed passes
    passes = passes[passes['pass_outcome'].isna()].copy()
    passes = passes.dropna(subset=['player', 'pass_recipient'])

    # Build edge list
    edges = passes.groupby(['player', 'pass_recipient']).agg(
        weight=('type', 'count')
    ).reset_index()
    edges.columns = ['source', 'target', 'weight']
    edges = edges[edges['weight'] >= 3]

    # Build graph
    G = nx.DiGraph()
    for _, row in edges.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])

    return G, edges, passes

def network_metrics(G):
    if len(G.nodes) == 0:
        return pd.DataFrame()
    metrics = []
    betweenness = nx.betweenness_centrality(G, weight='weight')
    in_degree = dict(G.in_degree(weight='weight'))
    out_degree = dict(G.out_degree(weight='weight'))
    try:
        pagerank = nx.pagerank(G, weight='weight')
    except:
        pagerank = {n: 0 for n in G.nodes}

    for node in G.nodes:
        metrics.append({
            'player': node,
            'betweenness': round(betweenness.get(node, 0), 4),
            'passes_received': in_degree.get(node, 0),
            'passes_made': out_degree.get(node, 0),
            'influence_score': round(pagerank.get(node, 0), 4)
        })
    return pd.DataFrame(metrics).sort_values('influence_score', ascending=False)

def key_partnerships(edges, top_n=10):
    undirected = edges.copy()
    undirected['pair'] = undirected.apply(
        lambda r: tuple(sorted([r['source'], r['target']])), axis=1
    )
    partnerships = undirected.groupby('pair')['weight'].sum().reset_index()
    partnerships['player_1'] = partnerships['pair'].apply(lambda x: x[0])
    partnerships['player_2'] = partnerships['pair'].apply(lambda x: x[1])
    return partnerships[['player_1','player_2','weight']].sort_values('weight', ascending=False).head(top_n)

def passes_leading_to_shots(events, team='Manchester United'):
    shots = events[(events['type'] == 'Shot') & (events['team'] == team)][['match_id','index']].copy()
    shots['shot_index'] = shots['index']
    passes = events[(events['type'] == 'Pass') & (events['team'] == team)].copy()

    results = []
    for _, shot in shots.iterrows():
        # Look at the 3 passes before each shot in the same match
        pre = passes[
            (passes['match_id'] == shot['match_id']) &
            (passes['index'] < shot['shot_index']) &
            (passes['index'] >= shot['shot_index'] - 3)
        ]
        for _, p in pre.iterrows():
            results.append({
                'passer': p['player'],
                'recipient': p.get('pass_recipient', None)
            })

    df = pd.DataFrame(results).dropna()
    return df.groupby('passer').size().reset_index(name='pre_shot_passes').sort_values('pre_shot_passes', ascending=False)


if __name__ == '__main__':
    import os
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW = os.path.join(BASE, 'data', 'raw')
    PROC = os.path.join(BASE, 'data', 'processed')
    os.makedirs(PROC, exist_ok=True)

    print("[MODULE 2] Building Pass Network...")
    events = pd.read_csv(os.path.join(RAW, 'mu_events.csv'), low_memory=False)
    matches = pd.read_csv(os.path.join(RAW, 'mu_matches.csv'))

    mu = matches.copy()
    mu['result'] = mu.apply(lambda r: 'W' if
        (r['home_team']=='Manchester United' and r['home_score']>r['away_score']) or
        (r['away_team']=='Manchester United' and r['away_score']>r['home_score'])
        else ('D' if r['home_score']==r['away_score'] else 'L'), axis=1)

    # Full season network
    G_full, edges_full, passes_full = build_pass_network(events)
    metrics_full = network_metrics(G_full)
    metrics_full.to_csv(os.path.join(PROC, 'network_metrics_full.csv'), index=False)
    edges_full.to_csv(os.path.join(PROC, 'pass_edges_full.csv'), index=False)

    # Win vs Loss networks
    G_win, edges_win, _ = build_pass_network(events, matches_df=mu, result_filter='W')
    G_loss, edges_loss, _ = build_pass_network(events, matches_df=mu, result_filter='L')
    metrics_win = network_metrics(G_win)
    metrics_loss = network_metrics(G_loss)
    metrics_win.to_csv(os.path.join(PROC, 'network_metrics_win.csv'), index=False)
    metrics_loss.to_csv(os.path.join(PROC, 'network_metrics_loss.csv'), index=False)

    # Key partnerships and pre-shot passers
    partners = key_partnerships(edges_full)
    partners.to_csv(os.path.join(PROC, 'key_partnerships.csv'), index=False)

    pre_shot = passes_leading_to_shots(events)
    pre_shot.to_csv(os.path.join(PROC, 'pre_shot_passers.csv'), index=False)

    print("\nTop 8 Most Influential Players:")
    print(metrics_full.head(8)[['player','influence_score','betweenness','passes_made']].to_string(index=False))
    print("\nTop 5 Partnerships:")
    print(partners.head(5).to_string(index=False))
    print("\n[MODULE 2] Complete.")
