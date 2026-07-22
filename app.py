"""
========================================================
Manchester United Performance Intelligence Platform
Interactive Streamlit Dashboard
Author: Yash Chabukswar | MSc Data Science, UoM
========================================================
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import ast
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Man Utd Performance Intelligence",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)

MU_RED = '#DA291C'
MU_YELLOW = '#FBE122'
BG = '#0d0d1a'
PANEL = '#141428'

st.markdown("""
<style>
    .stApp { background-color: #0d0d1a; color: #ffffff; }
    .stSidebar { background-color: #141428; }
    .metric-card {
        background: #1e1e35;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #DA291C;
        margin: 5px 0;
    }
    h1, h2, h3 { color: #ffffff; }
    .stSelectbox label, .stMultiSelect label { color: #ccccdd; }
    div[data-testid="stMetric"] {
        background: #1e1e35;
        padding: 10px;
        border-radius: 8px;
        border-left: 3px solid #DA291C;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    base = '.'
    d = {}
    files = {
        'xg': 'data/processed/player_xg_v2.csv',
        'shots': 'data/processed/shots_with_xg_v2.csv',
        'matches': 'data/raw/mu_matches.csv',
        'network': 'data/processed/network_metrics_full.csv',
        'network_win': 'data/processed/network_metrics_win.csv',
        'network_loss': 'data/processed/network_metrics_loss.csv',
        'edges': 'data/processed/pass_edges_full.csv',
        'partners': 'data/processed/key_partnerships.csv',
        'pre_shot': 'data/processed/pre_shot_passers.csv',
        'pressing_player': 'data/processed/pressing_player_stats.csv',
        'pressing_ev': 'data/processed/pressing_effectiveness.csv',
        'recruitment': 'data/processed/recruitment_comps.csv',
        'profiles': 'data/processed/all_player_profiles.csv',
        'shots_W': 'data/processed/shots_locs_W.csv',
        'shots_L': 'data/processed/shots_locs_L.csv',
        'shots_all': 'data/processed/shots_locs_all.csv',
        'press_W': 'data/processed/press_locs_W.csv',
        'press_L': 'data/processed/press_locs_L.csv',
        'press_all': 'data/processed/press_locs_all.csv',
        'ball_loss': 'data/processed/ball_loss_locs.csv',
        'xg_coef': 'data/processed/xg_feature_importance.csv',
    }
    for key, path in files.items():
        try:
            d[key] = pd.read_csv(f'{base}/{path}', low_memory=False)
        except:
            d[key] = pd.DataFrame()

    # Process matches
    if len(d['matches']) > 0:
        m = d['matches']
        m['is_home'] = m['home_team'] == 'Manchester United'
        m['mu_score'] = m.apply(lambda r: r['home_score'] if r['is_home'] else r['away_score'], axis=1)
        m['opp_score'] = m.apply(lambda r: r['away_score'] if r['is_home'] else r['home_score'], axis=1)
        m['opponent'] = m.apply(lambda r: r['away_team'] if r['is_home'] else r['home_team'], axis=1)
        m['result'] = m.apply(lambda r: 'W' if r['mu_score']>r['opp_score'] else ('D' if r['mu_score']==r['opp_score'] else 'L'), axis=1)
        m['venue'] = m['is_home'].map({True:'Home',False:'Away'})
        d['matches'] = m
    return d

def pitch_shape(color='#1a2a1a', line_color='rgba(100,150,100,0.7)'):
    """Returns Plotly shapes for a football pitch."""
    shapes = [
        # Outline
        dict(type='rect', x0=0,y0=0,x1=120,y1=80, line=dict(color=line_color,width=2), fillcolor=color),
        # Halfway
        dict(type='line', x0=60,y0=0,x1=60,y1=80, line=dict(color=line_color,width=1.5)),
        # Centre circle
        dict(type='circle', x0=50,y0=30,x1=70,y1=50, line=dict(color=line_color,width=1.5)),
        # Penalty areas
        dict(type='rect', x0=0,y0=18,x1=18,y1=62, line=dict(color=line_color,width=1.5), fillcolor='rgba(0,0,0,0)'),
        dict(type='rect', x0=102,y0=18,x1=120,y1=62, line=dict(color=line_color,width=1.5), fillcolor='rgba(0,0,0,0)'),
        # 6-yard boxes
        dict(type='rect', x0=0,y0=30,x1=6,y1=50, line=dict(color=line_color,width=1), fillcolor='rgba(0,0,0,0)'),
        dict(type='rect', x0=114,y0=30,x1=120,y1=50, line=dict(color=line_color,width=1), fillcolor='rgba(0,0,0,0)'),
        # Goals
        dict(type='line', x0=0,y0=36,x1=0,y1=44, line=dict(color='white',width=4)),
        dict(type='line', x0=120,y0=36,x1=120,y1=44, line=dict(color='white',width=4)),
    ]
    return shapes

# ============================================================
# LOAD
# ============================================================
data = load_data()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown(f"""
<div style="text-align:center; padding:15px; background:{MU_RED}; border-radius:8px; margin-bottom:20px;">
    <h2 style="color:white; margin:0;">🔴 MU Analytics</h2>
    <p style="color:{MU_YELLOW}; margin:0; font-size:12px;">Performance Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigate", [
    "📊 Season Overview",
    "⚽ xG Model",
    "🕸️ Pass Network",
    "🏃 Pressing Analysis",
    "🔍 Recruitment",
    "🗺️ Tactical Spatial"
], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="font-size:11px; color:#888; padding:10px;">
    <b style="color:#ccc;">Data Source</b><br>
    StatsBomb Open Data<br>
    Premier League 2015/16<br><br>
    <b style="color:#ccc;">Coverage</b><br>
    38 matches | 134,053 events<br><br>
    <b style="color:#ccc;">Author</b><br>
    Yash Chabukswar<br>
    MSc Data Science, UoM
</div>
""", unsafe_allow_html=True)

# ============================================================
# PAGE 1: SEASON OVERVIEW
# ============================================================
if page == "📊 Season Overview":
    st.markdown(f"<h1 style='color:{MU_RED};'>Manchester United — Season Overview</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#888;'>Premier League 2015/16 | Full season analysis</p>", unsafe_allow_html=True)

    matches = data['matches']
    if len(matches) > 0:
        W = len(matches[matches['result']=='W'])
        D = len(matches[matches['result']=='D'])
        L = len(matches[matches['result']=='L'])
        gf = int(matches['mu_score'].sum())
        ga = int(matches['opp_score'].sum())

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Wins", W)
        c2.metric("Draws", D)
        c3.metric("Losses", L)
        c4.metric("Goals Scored", gf)
        c5.metric("Goals Conceded", ga)
        c6.metric("Goal Diff", gf-ga)

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(data=[go.Pie(
                labels=['Wins','Draws','Losses'],
                values=[W,D,L],
                marker_colors=[MU_RED,'#888',MU_YELLOW],
                hole=0.4,
                textfont=dict(color='white', size=14)
            )])
            fig.update_layout(title='Season Results', paper_bgcolor=PANEL,
                            plot_bgcolor=PANEL, font_color='white', height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            matches_sorted = matches.sort_values('match_id').reset_index(drop=True)
            matches_sorted['match_num'] = range(1, len(matches_sorted)+1)
            res_colors = {'W': MU_RED, 'D': MU_YELLOW, 'L': '#888'}

            fig = go.Figure()
            fig.add_trace(go.Bar(x=matches_sorted['match_num'], y=matches_sorted['mu_score'],
                                 name='Goals Scored', marker_color=MU_RED, opacity=0.8))
            fig.add_trace(go.Bar(x=matches_sorted['match_num'], y=matches_sorted['opp_score'],
                                 name='Goals Conceded', marker_color='#4fc3f7', opacity=0.8))
            fig.update_layout(title='Goals Per Match', barmode='group',
                            paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                            font_color='white', height=350, xaxis_title='Match')
            st.plotly_chart(fig, use_container_width=True)

        # Results table
        st.subheader("Match Results")
        disp = matches[['match_date','opponent','venue','mu_score','opp_score','result']].copy()
        disp.columns = ['Date','Opponent','Venue','MU Goals','Opp Goals','Result']
        result_filter = st.selectbox("Filter by result", ["All","W","D","L"])
        if result_filter != "All":
            disp = disp[disp['Result']==result_filter]
        st.dataframe(disp.reset_index(drop=True), use_container_width=True, height=400)

# ============================================================
# PAGE 2: xG MODEL
# ============================================================
elif page == "⚽ xG Model":
    st.markdown(f"<h1 style='color:{MU_RED};'>Expected Goals (xG) Model</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#888;'>Logistic regression model trained on 11 features including StatsBomb freeze frame data</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Model AUC", "0.794")
    col2.metric("Features Used", "11")
    col3.metric("Shots Modelled", "2,394")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["xG vs Goals", "Shot Map", "Model Features"])

    xg = data['xg']
    shots = data['shots']

    with tab1:
        if len(xg) > 0:
            xg_f = xg[xg['shots']>=5].sort_values('xg', ascending=False)
            player_sel = st.multiselect("Select players", xg_f['player'].str.split().str[-1].tolist(),
                                        default=xg_f['player'].str.split().str[-1].tolist()[:12])
            xg_f2 = xg_f[xg_f['player'].str.split().str[-1].isin(player_sel)] if player_sel else xg_f

            fig = go.Figure()
            fig.add_trace(go.Bar(x=xg_f2['player'].str.split().str[-1], y=xg_f2['xg'],
                                 name='xG', marker_color='#4fc3f7', opacity=0.85))
            fig.add_trace(go.Bar(x=xg_f2['player'].str.split().str[-1], y=xg_f2['goals'],
                                 name='Actual Goals', marker_color=MU_RED, opacity=0.85))
            fig.update_layout(barmode='group', paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                            font_color='white', height=400, title='xG vs Actual Goals')
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("xG Over/Underperformance")
            st.markdown("*Positive = scoring more than model predicts (clinical finisher) | Negative = underperforming xG*")
            xg_diff = xg[xg['shots']>=5].sort_values('xg_diff')
            colors = [MU_RED if v<0 else '#4caf50' for v in xg_diff['xg_diff']]
            fig2 = go.Figure(go.Bar(
                x=xg_diff['xg_diff'], y=xg_diff['player'].str.split().str[-1],
                orientation='h', marker_color=colors, opacity=0.85
            ))
            fig2.add_vline(x=0, line_color='white', line_width=1, opacity=0.5)
            fig2.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                             font_color='white', height=400, xaxis_title='Goals - xG')
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        if len(shots) > 0:
            mu_shots = shots[shots['team']=='Manchester United'].copy()
            result_f = st.selectbox("Filter by match result", ["All","W","D","L"])
            matches = data['matches']

            if result_f != "All" and len(matches) > 0:
                ids = matches[matches['result']==result_f]['match_id'].tolist()
                mu_shots = mu_shots[mu_shots['match_id'].isin(ids)]

            goals_s = mu_shots[mu_shots['goal']==1]
            non_goals = mu_shots[mu_shots['goal']==0]

            fig = go.Figure()
            shapes = pitch_shape()

            fig.add_trace(go.Scatter(
                x=non_goals['x'], y=non_goals['y'],
                mode='markers',
                marker=dict(size=non_goals['xg']*30+5, color=non_goals['xg'],
                            colorscale='YlOrRd', cmin=0, cmax=0.5,
                            colorbar=dict(title=dict(text='xG', font=dict(color='white')), tickfont=dict(color='white')),
                            opacity=0.7, line=dict(color='white',width=0.5)),
                name='Shot (no goal)', hovertemplate='xG: %{marker.color:.3f}<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=goals_s['x'], y=goals_s['y'],
                mode='markers',
                marker=dict(symbol='star', size=18, color=MU_YELLOW,
                            line=dict(color='white',width=1)),
                name='⭐ Goal', hovertemplate='GOAL | xG: %{text}<extra></extra>',
                text=goals_s['xg'].round(3).astype(str) if 'xg' in goals_s else ['']
            ))

            fig.update_layout(
                shapes=shapes,
                xaxis=dict(range=[-2,122], visible=False),
                yaxis=dict(range=[-2,82], visible=False, scaleanchor='x', scaleratio=1),
                paper_bgcolor=PANEL, plot_bgcolor='#0a1a0a',
                height=500, title=f'Shot Map — {result_f} matches',
                font_color='white', legend=dict(bgcolor=PANEL)
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        coef = data['xg_coef']
        if len(coef) > 0:
            clean = coef.copy()
            clean['feature_clean'] = clean['feature'].str.replace('_',' ').str.title()
            clean['direction'] = ['Reduces xG' if v<0 else 'Increases xG' for v in clean['coefficient']]
            colors = [MU_RED if v<0 else '#4caf50' for v in clean['coefficient']]
            fig = go.Figure(go.Bar(
                x=clean['coefficient'], y=clean['feature_clean'],
                orientation='h', marker_color=colors, opacity=0.85
            ))
            fig.add_vline(x=0, line_color='white', line_width=1, opacity=0.5)
            fig.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                            font_color='white', height=400,
                            title='Model Feature Coefficients',
                            xaxis_title='Coefficient (effect on xG)')
            st.plotly_chart(fig, use_container_width=True)
            st.info("Distance and number of defenders have the strongest negative effect on xG. When the goalkeeper is off their line (far from goal), xG increases — as expected.")

# ============================================================
# PAGE 3: PASS NETWORK
# ============================================================
elif page == "🕸️ Pass Network":
    st.markdown(f"<h1 style='color:{MU_RED};'>Player Influence Network</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#888;'>Graph analytics on passing patterns — who connects the team?</p>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Pass Network Graph", "Win vs Loss Analysis"])

    with tab1:
        edges = data['edges']
        network = data['network']
        min_passes = st.slider("Minimum passes for connection", 10, 100, 40)

        if len(edges) > 0:
            top_edges = edges[edges['weight']>=min_passes]
            G = nx.DiGraph()
            for _, row in top_edges.iterrows():
                G.add_edge(row['source'], row['target'], weight=row['weight'])

            if len(G.nodes) > 0:
                pos = nx.spring_layout(G, seed=42, k=2.5)
                inf = network.set_index('player')['influence_score'].to_dict()

                edge_x, edge_y = [], []
                for u, v in G.edges():
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]

                node_x = [pos[n][0] for n in G.nodes]
                node_y = [pos[n][1] for n in G.nodes]
                node_sizes = [inf.get(n, 0.01)*8000+15 for n in G.nodes]
                node_colors = [inf.get(n, 0) for n in G.nodes]
                node_labels = [n.split()[-1] for n in G.nodes]
                node_text = [f"{n.split()[-1]}<br>Influence: {inf.get(n,0):.4f}" for n in G.nodes]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                        line=dict(color='rgba(79,195,247,0.3)', width=1),
                                        hoverinfo='none'))
                fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text',
                                        marker=dict(size=[s/50 for s in node_sizes],
                                                    color=node_colors, colorscale='RdYlGn',
                                                    colorbar=dict(title=dict(text='Influence', font=dict(color='white')), tickfont=dict(color='white')),
                                                    line=dict(color='white',width=1)),
                                        text=node_labels, textposition='top center',
                                        textfont=dict(color='white', size=10),
                                        hovertext=node_text, hoverinfo='text'))
                fig.update_layout(showlegend=False, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                                 xaxis=dict(visible=False), yaxis=dict(visible=False),
                                 height=550, title='Player Pass Network (node size/colour = influence score)',
                                 font_color='white')
                st.plotly_chart(fig, use_container_width=True)

        partners = data['partners']
        if len(partners) > 0:
            st.subheader("Top Passing Partnerships")
            partners['pair'] = partners['player_1'].str.split().str[-1] + ' ↔ ' + partners['player_2'].str.split().str[-1]
            fig = go.Figure(go.Bar(x=partners.head(10)['weight'], y=partners.head(10)['pair'],
                                  orientation='h', marker_color=MU_RED, opacity=0.85))
            fig.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL, font_color='white',
                            height=350, xaxis_title='Combined Pass Exchanges')
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        nw = data['network_win']
        nl = data['network_loss']
        if len(nw) > 0 and len(nl) > 0:
            top8 = data['network'].sort_values('influence_score', ascending=False).head(8)['player'].tolist()
            win_s = nw.set_index('player')['influence_score'].reindex(top8).fillna(0)
            loss_s = nl.set_index('player')['influence_score'].reindex(top8).fillna(0)
            names = [p.split()[-1] for p in top8]

            fig = go.Figure()
            fig.add_trace(go.Bar(x=names, y=win_s, name='Wins', marker_color='#4caf50', opacity=0.85))
            fig.add_trace(go.Bar(x=names, y=loss_s, name='Losses', marker_color=MU_RED, opacity=0.85))
            fig.update_layout(barmode='group', paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                            font_color='white', height=400,
                            title='Player Influence: Wins vs Losses',
                            yaxis_title='PageRank Influence Score')
            st.plotly_chart(fig, use_container_width=True)
            st.info("Players whose influence score is notably higher in wins are the key performers. A drop in influence during losses suggests they are being successfully neutralised by opponents.")

# ============================================================
# PAGE 4: PRESSING
# ============================================================
elif page == "🏃 Pressing Analysis":
    st.markdown(f"<h1 style='color:{MU_RED};'>Pressing Effectiveness Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#888;'>Not just how much they press — how effectively they win the ball back</p>", unsafe_allow_html=True)

    pp = data['pressing_player']
    pe = data['pressing_ev']

    if len(pp) > 0:
        overall_rate = (pe['won_ball'].sum() / len(pe) * 100) if len(pe) > 0 else 0
        c1,c2,c3 = st.columns(3)
        c1.metric("Total Pressures", f"{len(pe):,}")
        c2.metric("Balls Won", f"{pe['won_ball'].sum():,}")
        c3.metric("Team Effectiveness", f"{overall_rate:.1f}%")

    tab1, tab2 = st.tabs(["Player Effectiveness", "Spatial Pressing Map"])

    with tab1:
        if len(pp) > 0:
            pp_sorted = pp.sort_values('effectiveness_rate', ascending=True)
            colors = ['#4caf50' if v>=22 else '#4fc3f7' if v>=18 else MU_RED for v in pp_sorted['effectiveness_rate']]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pp_sorted['effectiveness_rate'],
                y=pp_sorted['player'].str.split().str[-1],
                mode='markers',
                marker=dict(color=colors, size=10, opacity=0.9,
                           line=dict(color='white', width=1)),
                error_x=dict(
                    type='data',
                    symmetric=False,
                    array=pp_sorted['ci_upper']-pp_sorted['effectiveness_rate'] if 'ci_upper' in pp_sorted else [0]*len(pp_sorted),
                    arrayminus=pp_sorted['effectiveness_rate']-pp_sorted['ci_lower'] if 'ci_lower' in pp_sorted else [0]*len(pp_sorted),
                    color='rgba(255,255,255,0.4)', thickness=2, width=6
                ),
                hovertemplate='<b>%{y}</b><br>Rate: %{x:.1f}%<br>95% CI: [%{error_x.arrayminus:.1f}%, %{error_x.array:.1f}%]<extra></extra>'
            ))
            fig.add_vline(x=overall_rate, line_color=MU_YELLOW, line_dash='dash',
                         annotation_text=f'Team avg: {overall_rate:.1f}%',
                         annotation_font_color=MU_YELLOW)
            fig.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL, font_color='white',
                            height=520, title='Ball Recovery Rate per Player with 95% Confidence Intervals',
                            xaxis_title='Ball Recovery Rate (%)')
            st.plotly_chart(fig, use_container_width=True)
            st.info("Error bars show 95% Wilson score confidence intervals. Players with fewer than 50 pressures (Low reliability) should be interpreted with caution — their point estimates have wide intervals.")
            if 'reliability' in pp_sorted.columns:
                st.dataframe(pp_sorted[['player','total_pressures','balls_won','effectiveness_rate','ci_lower','ci_upper','reliability']].rename(columns={'player':'Player','total_pressures':'Pressures','balls_won':'Balls Won','effectiveness_rate':'Rate %','ci_lower':'CI Lower','ci_upper':'CI Upper','reliability':'Reliability'}).reset_index(drop=True), use_container_width=True)

    with tab2:
        result_sel = st.selectbox("Show pressing map for", ["All matches","Wins only","Losses only"])
        if result_sel == "All matches":
            press_df = data['press_all']
        elif result_sel == "Wins only":
            press_df = data['press_W']
        else:
            press_df = data['press_L']

        if len(press_df) > 0:
            fig = go.Figure()
            fig.update_layout(
                shapes=pitch_shape(),
                xaxis=dict(range=[-2,122], visible=False),
                yaxis=dict(range=[-2,82], visible=False, scaleanchor='x', scaleratio=1),
                paper_bgcolor=PANEL, plot_bgcolor='#0a1a0a',
                height=480, title=f'Pressing Heatmap — {result_sel}',
                font_color='white'
            )

            # KDE
            if len(press_df) >= 10:
                try:
                    x, y = press_df['x'].values, press_df['y'].values
                    xi = np.linspace(0, 120, 100)
                    yi = np.linspace(0, 80, 80)
                    Xi, Yi = np.meshgrid(xi, yi)
                    kde = gaussian_kde(np.vstack([x,y]), bw_method=0.1)
                    Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
                    fig.add_trace(go.Contour(x=xi, y=yi, z=Zi,
                                            colorscale='Reds', opacity=0.7,
                                            showscale=False, contours=dict(coloring='fill')))
                except:
                    fig.add_trace(go.Scatter(x=press_df['x'], y=press_df['y'],
                                            mode='markers', marker=dict(color=MU_RED, size=4, opacity=0.3)))
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 5: RECRUITMENT
# ============================================================
elif page == "🔍 Recruitment":
    st.markdown(f"<h1 style='color:{MU_RED};'>Recruitment Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#888;'>Multi-dimensional player profiling using cosine similarity across 10 performance metrics</p>", unsafe_allow_html=True)

    rec = data['recruitment']
    if len(rec) > 0:
        target = st.selectbox("Select Man Utd player to find comparables for",
                             rec['target_player'].unique())
        comps = rec[rec['target_player']==target].head(5)

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Bar(
                x=comps['similarity'],
                y=comps['player'].str.split().str[-1] + ' (' + comps['team'].str.split().str[-1] + ')',
                orientation='h',
                marker_color=[MU_RED,'#4fc3f7','#4caf50','#ff9800','#9c27b0'],
                opacity=0.85,
                text=comps['similarity'].round(3).astype(str),
                textposition='outside', textfont=dict(color='white')
            ))
            fig.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL, font_color='white',
                            height=350, title=f'Similarity to {target.split()[-1]}',
                            xaxis_title='Cosine Similarity Score', xaxis_range=[0.7, 1.0])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            metrics = ['goals','passes','pass_accuracy','conversion_rate']
            metric_labels = ['Goals','Passes (÷20)','Pass Acc %','Conv Rate %']

            fig = go.Figure()
            colors_r = [MU_RED,'#4fc3f7','#4caf50','#ff9800','#9c27b0']
            for i, (_, row) in enumerate(comps.iterrows()):
                vals = [row['goals'], row['passes']/20 if 'passes' in row else 0,
                        row.get('pass_accuracy',0), row.get('conversion_rate',0)]
                fig.add_trace(go.Scatter(x=metric_labels, y=vals, mode='lines+markers',
                                        name=row['player'].split()[-1],
                                        line=dict(color=colors_r[i], width=2),
                                        marker=dict(size=8)))
            fig.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL, font_color='white',
                            height=350, title='Performance Profile Comparison')
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Full Shortlist")
        disp = comps[['player','team','similarity','goals','passes','pass_accuracy']].copy()
        disp['similarity'] = disp['similarity'].round(3)
        disp.columns = ['Candidate','Club','Similarity','Goals','Passes','Pass Acc %']
        st.dataframe(disp.reset_index(drop=True), use_container_width=True)

# ============================================================
# PAGE 6: TACTICAL SPATIAL
# ============================================================
elif page == "🗺️ Tactical Spatial":
    st.markdown(f"<h1 style='color:{MU_RED};'>Tactical Spatial Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#888;'>Where does Man Utd attack, press, and lose the ball?</p>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Shot Zones", "Pressing Zones", "Ball Loss Danger"])

    def draw_pitch_traces(fig):
        lc = 'rgba(120,180,120,0.9)'
        # Pitch outline and lines
        for xs, ys in [
            ([0,120,120,0,0],[0,0,80,80,0]),
            ([60,60],[0,80]),
            ([0,18,18,0],[18,18,62,62]),
            ([120,102,102,120],[18,18,62,62]),
            ([0,6,6,0],[30,30,50,50]),
            ([114,120,120,114],[30,30,50,50]),
        ]:
            fig.add_trace(go.Scatter(x=xs,y=ys,mode='lines',
                line=dict(color=lc,width=1.5),hoverinfo='skip',showlegend=False))
        th = np.linspace(0,2*np.pi,60)
        fig.add_trace(go.Scatter(x=60+10*np.cos(th),y=40+10*np.sin(th),mode='lines',
            line=dict(color=lc,width=1.5),hoverinfo='skip',showlegend=False))
        for ys in [[36,44]]:
            fig.add_trace(go.Scatter(x=[0,0],y=ys,mode='lines',
                line=dict(color='white',width=4),hoverinfo='skip',showlegend=False))
            fig.add_trace(go.Scatter(x=[120,120],y=ys,mode='lines',
                line=dict(color='white',width=4),hoverinfo='skip',showlegend=False))
        return fig

    def plotly_pitch_kde(df, title, colorscale='Reds'):
        fig = go.Figure()

        if len(df) >= 5:
            try:
                from scipy.ndimage import gaussian_filter
                x, y = df['x'].values, df['y'].values
                xi = np.linspace(0, 120, 35)
                yi = np.linspace(0, 80, 25)
                H, xe, ye = np.histogram2d(x, y, bins=[xi, yi])
                H_sm = gaussian_filter(H.T, sigma=2.0).astype(float)
                H_sm[H_sm < 0.05] = np.nan
                xc = (xe[:-1]+xe[1:])/2
                yc = (ye[:-1]+ye[1:])/2
                fig.add_trace(go.Heatmap(
                    x=xc, y=yc, z=H_sm,
                    colorscale=colorscale, opacity=0.85,
                    showscale=False, zsmooth='best', hoverinfo='skip'
                ))
            except Exception:
                fig.add_trace(go.Scatter(
                    x=df['x'], y=df['y'], mode='markers',
                    marker=dict(color='red',size=6,opacity=0.5),
                    hoverinfo='skip', showlegend=False
                ))

        fig = draw_pitch_traces(fig)

        fig.update_layout(
            xaxis=dict(range=[-2,122], visible=False, fixedrange=True),
            yaxis=dict(range=[-2,82], visible=False, scaleanchor='x', scaleratio=1, fixedrange=True),
            paper_bgcolor=PANEL, plot_bgcolor='#0a1a0a',
            height=480,
            title=dict(text=title, font=dict(color='white', size=13)),
            font_color='white',
            margin=dict(l=10,r=10,t=50,b=10),
            showlegend=False
        )
        return fig

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = plotly_pitch_kde(data['shots_W'], 'Shot Zones — WINS', colorscale='Greens')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = plotly_pitch_kde(data['shots_L'], 'Shot Zones — LOSSES', colorscale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        st.info("In wins, Man Utd create more shots from central attacking areas. In losses, shot locations are wider and further from goal — indicating less penetration through the middle.")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = plotly_pitch_kde(data['press_W'], 'Pressing Map — WINS', colorscale='Greens')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = plotly_pitch_kde(data['press_L'], 'Pressing Map — LOSSES', colorscale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        st.info("In wins, pressing is more structured and concentrated in the middle third. In losses, pressing becomes scattered — a sign of tactical breakdown under pressure.")

    with tab3:
        fig = plotly_pitch_kde(data['ball_loss'], 'Ball Loss Danger Zones (Dispossessions & Miscontrols)', colorscale='Oranges')
        st.plotly_chart(fig, use_container_width=True)
        st.info("Areas where Man Utd most frequently lose possession. High-density zones in the attacking third are dangerous — turnovers near goal can lead to fast counter-attacks.")


# This file has been updated — see full version
