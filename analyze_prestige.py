import sys
import os
import pandas as pd

sys.path.append(os.path.abspath('.'))
os.environ['LEAGUE_CONFIG'] = '10_team_private_half_ppr'
from app import generate_predictions

print("Loading player projections...")
draft_list = generate_predictions()

# Sort the overall draft list by VORP to calculate 'overall rank'
draft_list = draft_list.sort_values(by='vorp', ascending=False).reset_index(drop=True)
draft_list['overall_rank'] = draft_list.index + 1

# Find user's drafted team
my_team = [
    "Sam Darnold",
    "Jaxon Smith-Njigba",
    "Emeka Egbuka",
    "Saquon Barkley",
    "Cam Skattebo",
    "Tyler Warren",
    "Rico Dowdle",
    "Evan McPherson",
    "Carnell Tate",
    "Jonathon Brooks",
    "Tank Bigsby",
    "Stefon Diggs",
    "Rashid Shaheed",
    "Tyjae Spears"
]

print("\\n=== Your Team Analysis ===")
team_vorp = 0
drafted_data = []
for p in my_team:
    match = draft_list[draft_list['player_name'] == p]
    if len(match) == 0:
        # Fuzzy match
        for idx, row in draft_list.iterrows():
            if p.split()[-1] in row['player_name']:
                match = draft_list.iloc[[idx]]
                break
    if len(match) > 0:
        row = match.iloc[0]
        pts = row['predicted_pts']
        vorp = row['vorp']
        pos = row['position']
        team_vorp += vorp
        drafted_data.append({'Name': p, 'Pos': pos, 'Pts': pts, 'VORP': vorp, 'Rank': row['overall_rank']})

# Sort by VORP
drafted_data = sorted(drafted_data, key=lambda x: x['VORP'], reverse=True)
for d in drafted_data:
    print(f"{d['Name']} ({d['Pos']}): {d['Pts']:.1f} pts, {d['VORP']:.1f} VORP (Overall Rank: {d['Rank']})")
    
print(f"\\nTotal Team VORP: {team_vorp:.1f}")

# Let's find players you drafted that the model dislikes (low rank/negative vorp)
dislikes = [d for d in drafted_data if d['VORP'] < 10 or d['Rank'] > 100]
if dislikes:
    print("\n=== Biggest Discrepancies (Model hates these picks) ===")
    for d in dislikes:
        print(f"{d['Name']}: Ranked {d['Rank']}, VORP {d['VORP']:.1f}")

# Find players the model loved that you missed (Top overall players by VORP)
print("\n=== Top 10 Model Recommended Players ===")
for _, row in draft_list.head(10).iterrows():
    print(f"{row['player_name']} ({row['position']}): {row['predicted_pts']:.1f} pts, {row['vorp']:.1f} VORP")

# Analyze WR2s
print("\n=== Depth Chart Analysis (WR2 vs WR1) ===")
wr_df = draft_list[draft_list['position'] == 'WR'].copy()
from src.preprocessing import load_depth_charts
depth_map = load_depth_charts(2026)
wr_df['pos_rank'] = wr_df.apply(lambda r: depth_map.get((r['player_name'].lower().replace('.', '').replace('-', '').replace(' ', ''), 'WR'), 99), axis=1)

wr1s = wr_df[wr_df['pos_rank'] == 1].head(5)
wr2s = wr_df[wr_df['pos_rank'] == 2].head(5)

print("\nTop 5 Projected WR1s:")
for _, row in wr1s.iterrows():
    print(f"- {row['player_name']} ({row['team_predict']}): {row['predicted_pts']:.1f} pts")

print("\nTop 5 Projected WR2s (Compressed):")
for _, row in wr2s.iterrows():
    print(f"- {row['player_name']} ({row['team_predict']}): {row['predicted_pts']:.1f} pts")
