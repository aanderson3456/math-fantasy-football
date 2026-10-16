import sys
import os
import pandas as pd

sys.path.append(os.path.abspath('.'))
os.environ['LEAGUE_CONFIG'] = '10_team_private_half_ppr'
from app import generate_predictions

print("Loading player projections (which includes raw input features)...")
draft_list = generate_predictions()

players_to_inspect = ['Jonathon Brooks', 'Cam Skattebo', 'Stefon Diggs']

print("\\n=== NN INPUT FEATURES FOR SPECIFIC PLAYERS ===")
for p in players_to_inspect:
    match = draft_list[draft_list['player_name'] == p]
    if len(match) == 0:
        for idx, row in draft_list.iterrows():
            if p.split()[-1] in row['player_name']:
                match = draft_list.iloc[[idx]]
                break
    if len(match) > 0:
        print(f"\\n--- {p} ---")
        row = match.iloc[0]
        # Print only relevant numeric and string features, skip dummy variables or internal stuff if possible
        skip_cols = ['predicted_adp', 'predicted_pts', 'vorp', 'draft_rank']
        for col, val in row.items():
            if col not in skip_cols and pd.notna(val):
                print(f"{col}: {val}")
    else:
        print(f"\\n--- {p} ---")
        print("Player not found in draft list!")
