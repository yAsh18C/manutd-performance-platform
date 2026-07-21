"""
MASTER DASHBOARD
Manchester United Performance Intelligence Platform
Professional multi-panel analytical dashboard.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

MU_RED = '#DA291C'
MU_YELLOW = '#FBE122'
BG = '#0d0d1a'
PANEL_BG = '#141428'
GRID = '#2a2a4a'
WHITE = '#ffffff'
LIGHT = '#ccccdd'

def draw_pitch(ax, color='#1a2a1a', line_color='#4a6a4a', alpha=1.0):
    ax.set_facecolor(color)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 80)
    ax.set_aspect('equal')
    ax.axis('off')
    lw = 1.2
    # Pitch outline
    ax.plot([0,120,120,0,0],[0,0,80,80,0], color=line_color, lw=lw, alpha=alpha)
    # Halfway line
    ax.plot([60,60],[0,80], color=line_color, lw=lw, alpha=alpha)
    # Centre circle
    circle = plt.Circle((60,40), 10, color=line_color, fill=False, lw=lw, alpha=alpha)
    ax.add_patch(circle)
    ax.plot(60,40,'o', color=line_color, ms=2, alpha=alpha)
    # Penalty areas
    ax.plot([0,18,18,0],[18,18,62,62], color=line_color, lw=lw, alpha=alpha)
    ax.plot([120,102,102,120],[18,18,62,62], color=line_color, lw=lw, alpha=alpha)
    # 6-yard boxes
    ax.plot([0,6,6,0],[30,30,50,50], color=line_color, lw=lw, alpha=alpha)
    ax.plot([120,114,114,120],[30,30,50,50], color=line_color, lw=lw, alpha=alpha)
    # Goals
    ax.plot([0,0],[36,44], color='white', lw=3, alpha=alpha)
    ax.plot([120,120],[36,44], color='white', lw=3, alpha=alpha)

def kde_heatmap(ax, x, y, cmap_name='Reds', alpha=0.7, bandwidth=4):
    try:
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy, bw_method=bandwidth/100)
        xi = np.linspace(0, 120, 200)
        yi = np.linspace(0, 80, 150)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
        ax.contourf(Xi, Yi, Zi, levels=15, cmap=cmap_name, alpha=alpha, zorder=2)
    except Exception as e:
        pass

# ============================================================
# LOAD ALL DATA
# ============================================================

# Automatically detect the project root folder regardless of OS
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, 'data', 'processed')
RAW = os.path.join(BASE, 'data', 'raw')
DASH = os.path.join(BASE, 'dashboards')

xg = pd.read_csv(os.path.join(PROC, 'player_xg_v2.csv'))
shots_xg = pd.read_csv(os.path.join(PROC, 'shots_with_xg_v2.csv'), low_memory=False)
network_full = pd.read_csv(os.path.join(PROC, 'network_metrics_full.csv'))
network_win = pd.read_csv(os.path.join(PROC, 'network_metrics_win.csv'))
network_loss = pd.read_csv(os.path.join(PROC, 'network_metrics_loss.csv'))
edges = pd.read_csv(os.path.join(PROC, 'pass_edges_full.csv'))
partners = pd.read_csv(os.path.join(PROC, 'key_partnerships.csv'))
pre_shot = pd.read_csv(os.path.join(PROC, 'pre_shot_passers.csv'))
pressing_eff = pd.read_csv(os.path.join(PROC, 'pressing_player_stats.csv'))
pressing_ev = pd.read_csv(os.path.join(PROC, 'pressing_effectiveness.csv'))
recruitment = pd.read_csv(os.path.join(PROC, 'recruitment_comps.csv'))
shots_all = pd.read_csv(os.path.join(PROC, 'shots_locs_all.csv'))
shots_W = pd.read_csv(os.path.join(PROC, 'shots_locs_W.csv'))
shots_L = pd.read_csv(os.path.join(PROC, 'shots_locs_L.csv'))
press_all = pd.read_csv(os.path.join(PROC, 'press_locs_all.csv'))
press_W = pd.read_csv(os.path.join(PROC, 'press_locs_W.csv'))
press_L = pd.read_csv(os.path.join(PROC, 'press_locs_L.csv'))
ball_loss = pd.read_csv(os.path.join(PROC, 'ball_loss_locs.csv'))
matches = pd.read_csv(os.path.join(RAW, 'mu_matches.csv'))
matches['result'] = matches.apply(lambda r: 'W' if
    (r['home_team']=='Manchester United' and r['home_score']>r['away_score']) or
    (r['away_team']=='Manchester United' and r['away_score']>r['home_score'])
    else ('D' if r['home_score']==r['away_score'] else 'L'), axis=1)
matches['mu_score'] = matches.apply(lambda r: r['home_score'] if r['home_team']=='Manchester United' else r['away_score'], axis=1)
matches['opp_score'] = matches.apply(lambda r: r['away_score'] if r['home_team']=='Manchester United' else r['home_score'], axis=1)

# ============================================================
# FIGURE SETUP — 5 rows x 3 cols
# ============================================================
fig = plt.figure(figsize=(24, 32), facecolor=BG)

title_ax = fig.add_axes([0, 0.965, 1, 0.035])
title_ax.set_facecolor(MU_RED)
title_ax.axis('off')
title_ax.text(0.5, 0.5,
    'MANCHESTER UNITED  |  PERFORMANCE INTELLIGENCE PLATFORM  |  PREMIER LEAGUE 2015/16',
    ha='center', va='center', fontsize=16, fontweight='bold', color='white',
    fontfamily='monospace')

gs = fig.add_gridspec(5, 3, top=0.96, bottom=0.02,
                       left=0.04, right=0.97,
                       hspace=0.42, wspace=0.30)

def panel(row, col, colspan=1):
    ax = fig.add_subplot(gs[row, col:col+colspan])
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.tick_params(colors=LIGHT, labelsize=8)
    ax.xaxis.label.set_color(LIGHT)
    ax.yaxis.label.set_color(LIGHT)
    return ax

def title(ax, text, sub=None):
    ax.set_title(text, color=WHITE, fontweight='bold', fontsize=10.5, pad=8)
    if sub:
        ax.annotate(sub, xy=(0.5, 1.01), xycoords='axes fraction',
                    ha='center', fontsize=7.5, color=MU_YELLOW, style='italic')

# ============================================================
# ROW 0: xG Model outputs
# ============================================================

# --- 0A: xG vs Actual Goals (over/underperformers) ---
ax = panel(0, 0)
xg_f = xg[xg['shots'] >= 5].sort_values('xg', ascending=False).head(12)
short = xg_f['player'].str.split().str[-1]
x_pos = np.arange(len(xg_f))
ax.bar(x_pos - 0.2, xg_f['xg'], 0.38, label='xG', color='#4fc3f7', alpha=0.85)
ax.bar(x_pos + 0.2, xg_f['goals'], 0.38, label='Actual Goals', color=MU_RED, alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels(short, rotation=45, ha='right', fontsize=7.5, color=LIGHT)
ax.legend(facecolor=PANEL_BG, labelcolor=LIGHT, fontsize=8, loc='upper right')
ax.set_ylabel('Goals / xG', color=LIGHT, fontsize=9)
ax.spines['bottom'].set_color(GRID)
ax.spines['left'].set_color(GRID)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
title(ax, 'Expected Goals (xG) vs Actual Goals', sub='Model AUC: 0.784 — Built from shot distance, angle, body part & technique')

# --- 0B: xG difference (over/under performance) ---
ax = panel(0, 1)
xg_diff = xg[xg['shots'] >= 5].sort_values('xg_diff')
colors_diff = [MU_RED if v < 0 else '#4caf50' for v in xg_diff['xg_diff']]
bars = ax.barh(xg_diff['player'].str.split().str[-1], xg_diff['xg_diff'], color=colors_diff, alpha=0.85)
ax.axvline(0, color=LIGHT, lw=0.8, alpha=0.5)
for bar, val in zip(bars, xg_diff['xg_diff']):
    ax.text(val + (0.05 if val >= 0 else -0.05), bar.get_y() + bar.get_height()/2,
            f'{val:+.1f}', va='center', ha='left' if val >= 0 else 'right',
            color=WHITE, fontsize=8, fontweight='bold')
ax.spines['bottom'].set_color(GRID)
ax.spines['left'].set_color(GRID)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
title(ax, 'xG Over/Underperformance', sub='Green = scoring more than model predicts | Red = underperforming xG')

# --- 0C: Shot map coloured by xG ---
ax = fig.add_subplot(gs[0, 2])
draw_pitch(ax, color='#0a1a0a')
mu_shots = shots_xg[(shots_xg['team']=='Manchester United') & shots_xg['xg'].notna()].copy()
goals_s = mu_shots[mu_shots['goal']==1]
non_goals = mu_shots[mu_shots['goal']==0]
sc = ax.scatter(non_goals['x'], non_goals['y'], c=non_goals['xg'],
                cmap='YlOrRd', s=non_goals['xg']*300+15, alpha=0.6,
                vmin=0, vmax=0.5, edgecolors='white', linewidth=0.3, zorder=3)
ax.scatter(goals_s['x'], goals_s['y'], c='gold', s=120, marker='*',
           zorder=5, edgecolors='white', linewidth=0.5, label='Goal')
ax.legend(facecolor='#0a1a0a', labelcolor=WHITE, fontsize=8, loc='upper left')
cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
cbar.ax.yaxis.set_tick_params(color=LIGHT, labelsize=7)
cbar.set_label('xG Value', color=LIGHT, fontsize=8)
ax.set_title('Shot Map — Coloured by xG Value\n★ = Goals scored', color=WHITE, fontweight='bold', fontsize=10.5, pad=6)

# ============================================================
# ROW 1: Pass Network
# ============================================================

# --- 1A: Full Pass Network ---
ax = panel(1, 0, colspan=2)
G = nx.DiGraph()
top_edges = edges[edges['weight'] >= 30].copy()
for _, row in top_edges.iterrows():
    G.add_edge(row['source'], row['target'], weight=row['weight'])

if len(G.nodes) > 0:
    pos = nx.spring_layout(G, seed=42, k=2)
    inf_scores = network_full.set_index('player')['influence_score'].to_dict()
    node_sizes = [inf_scores.get(n, 0.01) * 8000 for n in G.nodes]
    node_colors = [MU_RED if inf_scores.get(n,0) > 0.06 else '#4fc3f7' for n in G.nodes]
    edge_weights = [G[u][v]['weight'] for u,v in G.edges]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [w/max_w*4+0.5 for w in edge_weights]

    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                           edge_color='#4fc3f7', alpha=0.3, arrows=True,
                           arrowsize=8, connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color=node_colors, alpha=0.9)
    labels = {n: n.split()[-1] for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7,
                            font_color=WHITE, font_weight='bold')

red_patch = mpatches.Patch(color=MU_RED, label='High Influence (top connector)')
blue_patch = mpatches.Patch(color='#4fc3f7', label='Standard node')
ax.legend(handles=[red_patch, blue_patch], facecolor=PANEL_BG, labelcolor=LIGHT,
          fontsize=8, loc='lower right')
title(ax, 'Player Influence Network — Pass Connections (min 30 passes)',
      sub='Node size = PageRank influence score | Edge thickness = pass volume | Red = key connectors')

# --- 1B: Influence comparison Win vs Loss ---
ax = panel(1, 2)
top_players = network_full.sort_values('influence_score', ascending=False).head(8)['player'].tolist()
win_scores = network_win.set_index('player')['influence_score'].reindex(top_players).fillna(0)
loss_scores = network_loss.set_index('player')['influence_score'].reindex(top_players).fillna(0)
short_names = [p.split()[-1] for p in top_players]
x_pos = np.arange(len(top_players))
ax.bar(x_pos - 0.2, win_scores, 0.38, color='#4caf50', alpha=0.85, label='Win')
ax.bar(x_pos + 0.2, loss_scores, 0.38, color=MU_RED, alpha=0.85, label='Loss')
ax.set_xticks(x_pos)
ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7.5, color=LIGHT)
ax.legend(facecolor=PANEL_BG, labelcolor=LIGHT, fontsize=8)
ax.set_ylabel('Influence Score', color=LIGHT, fontsize=9)
ax.spines['bottom'].set_color(GRID)
ax.spines['left'].set_color(GRID)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
title(ax, 'Network Influence: Wins vs Losses', sub='Which players become more/less central in wins?')

# ============================================================
# ROW 2: Pressing Effectiveness
# ============================================================

# --- 2A: Player pressing effectiveness ---
ax = panel(2, 0)
press_top = pressing_eff.sort_values('effectiveness_rate', ascending=True).tail(10)
colors_press = ['#4caf50' if v >= 22 else '#4fc3f7' if v >= 18 else MU_RED
                for v in press_top['effectiveness_rate']]
bars = ax.barh(press_top['player'].str.split().str[-1],
               press_top['effectiveness_rate'], color=colors_press, alpha=0.85)
ax.axvline(18.5, color=MU_YELLOW, lw=1.2, linestyle='--', alpha=0.7, label='Team avg (18.5%)')
for bar, val in zip(bars, press_top['effectiveness_rate']):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', color=WHITE, fontsize=8, fontweight='bold')
ax.legend(facecolor=PANEL_BG, labelcolor=LIGHT, fontsize=8)
ax.set_xlabel('Ball Recovery Rate (%)', color=LIGHT, fontsize=9)
ax.spines['bottom'].set_color(GRID)
ax.spines['left'].set_color(GRID)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
title(ax, 'Pressing Effectiveness by Player', sub='% of pressures that directly won the ball back within 5 seconds')

# --- 2B: Pressing heatmap ---
ax = fig.add_subplot(gs[2, 1])
draw_pitch(ax)
kde_heatmap(ax, press_all['x'], press_all['y'], cmap_name='Reds', alpha=0.75)
ax.set_title('Pressing Heatmap — Where Man Utd Press\n(Full Season)', color=WHITE, fontweight='bold', fontsize=10.5, pad=6)
ax.text(20, -5, '← Defending', color=LIGHT, fontsize=7, ha='center')
ax.text(100, -5, 'Attacking →', color=LIGHT, fontsize=7, ha='center')

# --- 2C: Pressing in wins vs losses ---
ax = fig.add_subplot(gs[2, 2])
draw_pitch(ax)
kde_heatmap(ax, press_W['x'], press_W['y'], cmap_name='Greens', alpha=0.6)
kde_heatmap(ax, press_L['x'], press_L['y'], cmap_name='Reds', alpha=0.4)
green_p = mpatches.Patch(color='#4caf50', alpha=0.7, label='Pressing in Wins')
red_p = mpatches.Patch(color=MU_RED, alpha=0.5, label='Pressing in Losses')
ax.legend(handles=[green_p, red_p], facecolor='#0a1a0a', labelcolor=WHITE,
          fontsize=7.5, loc='upper left')
ax.set_title('Pressing Location: Wins vs Losses', color=WHITE, fontweight='bold', fontsize=10.5, pad=6)

# ============================================================
# ROW 3: Recruitment Analytics
# ============================================================

# --- 3A: Recruitment radar for Martial ---
ax = panel(3, 0)
martial_comps = recruitment[recruitment['target_player']=='Anthony Martial'].head(5)
metrics = ['goals','passes','pressures','conversion_rate']
metric_labels = ['Goals', 'Passes (÷10)', 'Pressures (÷10)', 'Conv.%']
x_pos = np.arange(len(metrics))

# Normalise for display
def norm(series, col):
    if col in ['passes','pressures']:
        return series / 10
    return series

mu_vals = [11, 93.8, 37.2, 19.3]
colors_r = ['#4fc3f7','#4caf50','#ff9800','#e91e63','#9c27b0']
for i, (_, row) in enumerate(martial_comps.iterrows()):
    vals = [row['goals'], row['passes']/10, row['pressures']/10 if 'pressures' in row else 0, row['conversion_rate'] if 'conversion_rate' in row else 0]
    ax.plot(x_pos, vals, 'o-', color=colors_r[i], alpha=0.7, lw=1.5,
            label=f"{row['player'].split()[-1]} ({row['team'].split()[0]})", ms=5)
ax.plot(x_pos, mu_vals, 's-', color=MU_RED, lw=2.5, ms=8, label='Martial (MU)', zorder=5)
ax.set_xticks(x_pos)
ax.set_xticklabels(metric_labels, color=LIGHT, fontsize=8)
ax.legend(facecolor=PANEL_BG, labelcolor=LIGHT, fontsize=7, loc='upper right')
ax.spines['bottom'].set_color(GRID)
ax.spines['left'].set_color(GRID)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
title(ax, 'Recruitment — Similar to Anthony Martial', sub='Cosine similarity across 10 performance dimensions')

# --- 3B: Recruitment table ---
ax = panel(3, 1)
ax.axis('off')
rec_display = recruitment[['target_player','player','team','similarity','goals','passes','pass_accuracy']].copy()
rec_display['target'] = rec_display['target_player'].str.split().str[-1]
rec_display['candidate'] = rec_display['player'].str.split().str[-1]
rec_display['sim'] = rec_display['similarity'].round(3)
rec_display['club'] = rec_display['team'].str.split().str[-1]

table_data = []
for _, row in rec_display.iterrows():
    table_data.append([row['target'], row['candidate'], row['club'], f"{row['sim']:.3f}", int(row['goals']), f"{row['pass_accuracy']:.0f}%"])

col_labels = ['Profile', 'Candidate', 'Club', 'Similarity', 'Goals', 'Pass%']
t = ax.table(cellText=table_data, colLabels=col_labels,
             loc='center', cellLoc='center')
t.auto_set_font_size(False)
t.set_fontsize(7.5)
t.scale(1, 1.35)
for (r,c), cell in t.get_celld().items():
    if r == 0:
        cell.set_facecolor(MU_RED)
        cell.set_text_props(color=WHITE, fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#1e1e35')
        cell.set_text_props(color=LIGHT)
    else:
        cell.set_facecolor(PANEL_BG)
        cell.set_text_props(color=LIGHT)
    cell.set_edgecolor(GRID)
title(ax, 'Recruitment Shortlist — Multi-Dimensional Similarity', sub='Candidates ranked by cosine similarity across 10 performance metrics')

# --- 3C: Pre-shot passers ---
ax = panel(3, 2)
ps_top = pre_shot.head(8).sort_values('pre_shot_passes')
bars = ax.barh(ps_top['passer'].str.split().str[-1], ps_top['pre_shot_passes'],
               color=MU_YELLOW, alpha=0.85)
for bar, val in zip(bars, ps_top['pre_shot_passes']):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            str(int(val)), va='center', color=WHITE, fontsize=9, fontweight='bold')
ax.set_xlabel('Passes in Final 3 Before Shot', color=LIGHT, fontsize=9)
ax.spines['bottom'].set_color(GRID)
ax.spines['left'].set_color(GRID)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
title(ax, 'Shot Creation — Key Passers', sub='Who is involved in the build-up to shots? (final 3 passes before each shot)')

# ============================================================
# ROW 4: Tactical Spatial
# ============================================================

# --- 4A: Shot heatmap Win vs Loss ---
ax = fig.add_subplot(gs[4, 0])
draw_pitch(ax)
kde_heatmap(ax, shots_W['x'], shots_W['y'], cmap_name='Greens', alpha=0.65)
kde_heatmap(ax, shots_L['x'], shots_L['y'], cmap_name='Reds', alpha=0.45)
green_p = mpatches.Patch(color='#4caf50', alpha=0.7, label='Shots in Wins')
red_p = mpatches.Patch(color=MU_RED, alpha=0.5, label='Shots in Losses')
ax.legend(handles=[green_p, red_p], facecolor='#0a1a0a', labelcolor=WHITE, fontsize=7.5, loc='upper left')
ax.set_title('Shot Zones: Wins vs Losses\n(Green = wins, Red = losses)', color=WHITE, fontweight='bold', fontsize=10.5, pad=6)

# --- 4B: Ball loss heatmap ---
ax = fig.add_subplot(gs[4, 1])
draw_pitch(ax)
kde_heatmap(ax, ball_loss['x'], ball_loss['y'], cmap_name='Reds', alpha=0.75)
ax.set_title('Ball Loss Danger Zones\n(Dispossessions & Miscontrols)', color=WHITE, fontweight='bold', fontsize=10.5, pad=6)

# --- 4C: Season xG timeline ---
ax = panel(4, 2)
mu_shot_match = shots_xg[shots_xg['team']=='Manchester United'].groupby('match_id').agg(
    xg_total=('xg','sum'), goals=('goal','sum')).reset_index()
mu_shot_match = mu_shot_match.merge(matches[['match_id','result']], on='match_id')
mu_shot_match = mu_shot_match.sort_values('match_id').reset_index(drop=True)
mu_shot_match['match_num'] = range(1, len(mu_shot_match)+1)

result_colors = {'W':'#4caf50','D':MU_YELLOW,'L':MU_RED}
for _, row in mu_shot_match.iterrows():
    ax.bar(row['match_num']-0.2, row['xg_total'], 0.35,
           color='#4fc3f7', alpha=0.6)
    ax.bar(row['match_num']+0.2, row['goals'], 0.35,
           color=result_colors.get(row['result'], LIGHT), alpha=0.85)

xg_legend = mpatches.Patch(color='#4fc3f7', alpha=0.6, label='xG')
ax.legend(handles=[xg_legend,
    mpatches.Patch(color='#4caf50', label='Goals (W)'),
    mpatches.Patch(color=MU_YELLOW, label='Goals (D)'),
    mpatches.Patch(color=MU_RED, label='Goals (L)')],
    facecolor=PANEL_BG, labelcolor=LIGHT, fontsize=7, loc='upper right')
ax.set_xlabel('Match Number', color=LIGHT, fontsize=9)
ax.set_ylabel('xG / Goals', color=LIGHT, fontsize=9)
ax.spines['bottom'].set_color(GRID)
ax.spines['left'].set_color(GRID)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
title(ax, 'xG vs Goals — Season Timeline', sub='xG (blue) vs actual goals by match result')


plt.savefig(os.path.join(DASH, 'manutd_advanced_dashboard.png'),
            dpi=150, bbox_inches='tight', facecolor=BG)
print("Dashboard saved to dashboards/manutd_advanced_dashboard.png")
